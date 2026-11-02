#!/usr/bin/env python3
"""Tencent HunyuanImage-3.0-Instruct & Instruct-Distil (83B Total / 13B Active 64-Expert MoE) TPU v7x / v5p Inference Suite.

Uses `tpu_single_host_setup.start_single_host_metadata_env` so each configuration runs in a clean single-client
multi-device `torch_tpu` (`tpu:0..N-1`) subprocess across all 8 TensorCores (`v7x`) or 4 chips (`v5p`) per host.

Upgrades:
  - `--stage-only` cooperative multi-host HF -> GCS staging (`wait_all=(rank == 0)`) for dot-free checkpoints
    `HunyuanImage-3-Instruct-Distil` and `HunyuanImage-3-Instruct`.
  - Evaluates `COMPILED_VECTORIZED_GROUPED_MM` (`torch.compile(..., dynamic=False)` with graceful fallback),
    `VECTORIZED_GROUPED_MM`, and `EAGER_MOE`.
  - Exercises hardware FP8 MXU acceleration via `torch._scaled_mm(..., out_dtype=torch.bfloat16)` on 128-aligned 2D
    projections with clean `.to(torch.bfloat16)` fallback.
  - Integrates `cpu_offload_client.benchmark_image_encode_ablation` (`1024x1024`, `1280x768`, `2048x2048`) and
    `benchmark_vision_preproc_ablation` (`TI2I-MultiImageEdit-2Refs-1024x1024`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/torchinductor")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_offload_client import (  # noqa: E402
    benchmark_image_encode_ablation,
    benchmark_vision_preproc_ablation,
    check_offload_health,
)
from gcs_stager import ensure_model_staged_to_gcs, gcs_upload_bytes, pull_gcs_prefix_to_local, utc_iso  # noqa: E402
from tpu_single_host_setup import start_single_host_metadata_env  # noqa: E402


def log(rank: int, msg: str) -> None:
    print(f"[{utc_iso()}][hunyuan-img3][host-{rank}] {msg}", flush=True)


def verify_dotfree_checkpoint(model_dir: Path) -> dict[str, Any]:
    from safetensors import safe_open

    assert "." not in model_dir.name, f"Directory name MUST NOT contain dots: {model_dir}"
    cfg = json.loads((model_dir / "config.json").read_text())
    shards = sorted(model_dir.glob("*.safetensors"))
    total_bytes = sum(p.stat().st_size for p in shards)
    idx_file = model_dir / "model.safetensors.index.json"
    if idx_file.exists():
        total_bytes = int(json.loads(idx_file.read_text()).get("metadata", {}).get("total_size", total_bytes))

    sample_shapes = {}
    if shards:
        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            for k in list(f.keys())[:16]:
                sample_shapes[k] = list(f.get_slice(k).get_shape())

    return {
        "dir_name": model_dir.name,
        "architectures": cfg.get("architectures"),
        "num_hidden_layers": cfg.get("num_hidden_layers", 32),
        "hidden_size": cfg.get("hidden_size", 4096),
        "num_experts": cfg.get("num_experts", 64),
        "max_position_embeddings": cfg.get("max_position_embeddings", 22800),
        "num_shards": len(shards),
        "total_weight_gib": round(total_bytes / (1024**3), 2),
        "sample_shapes": sample_shapes,
    }


def run_child_hy3_worker(
    model_dir: Path,
    variant_label: str,
    diff_steps: int,
    tp_ep_size: int,
    precision_mode: str,
    moe_kernel_mode: str,
    out_json: Path,
) -> None:
    import torch
    import torch.nn.functional as F
    from tpu_single_host_setup import _patch_torchtpu_export_multidevice
    _patch_torchtpu_export_multidevice()
    import torch_tpu  # noqa: F401
    from safetensors import safe_open

    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = False

    n_devs = torch.accelerator.device_count()
    devices = [torch.device(f"tpu:{i}") for i in range(n_devs)]
    dp_per_host = max(1, n_devs // tp_ep_size)

    cfg = json.loads((model_dir / "config.json").read_text())
    d_model = int(cfg.get("hidden_size", 4096))
    n_layers = int(cfg.get("num_hidden_layers", 32))
    n_heads = int(cfg.get("num_attention_heads", 32))
    n_kv = int(cfg.get("num_key_value_heads", 8))
    head_dim = int(cfg.get("attention_head_dim", 128))
    n_experts = int(cfg.get("num_experts", 64))
    topk = 8
    moe_inter = 3072
    local_experts = max(1, n_experts // tp_ep_size)
    local_q_heads = max(1, n_heads // tp_ep_size)
    local_kv_heads = max(1, n_kv // min(tp_ep_size, n_kv))
    gqa_rep = max(1, local_q_heads // local_kv_heads)
    local_topk = max(1, math.ceil(topk / tp_ep_size))
    shared_inter = max(128, ((moe_inter // tp_ep_size) // 128) * 128)

    use_fp8 = precision_mode == "FP8_DYNAMIC" and hasattr(torch, "float8_e4m3fn")
    use_compile = moe_kernel_mode == "COMPILED_VECTORIZED_GROUPED_MM"
    is_vectorized_moe = moe_kernel_mode in ("VECTORIZED_GROUPED_MM", "COMPILED_VECTORIZED_GROUPED_MM")

    shards = sorted(model_dir.glob("*.safetensors"))
    target_per_core = int((165.0 / tp_ep_size) * (1024**3))
    if use_fp8:
        target_per_core //= 2
    held: list[list[torch.Tensor]] = [[] for _ in range(n_devs)]
    per_core_bytes = [0 for _ in range(tp_ep_size)]

    t_l0 = time.perf_counter()
    for s_idx, sp in enumerate(shards):
        slot = s_idx % tp_ep_size
        if per_core_bytes[slot] >= target_per_core:
            continue
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                tc = f.get_tensor(k)
                for dp_g in range(dp_per_host):
                    d_i = dp_g * tp_ep_size + slot
                    if d_i < n_devs:
                        if use_fp8 and tc.ndim >= 2:
                            td = tc.to(torch.float8_e4m3fn).to(devices[d_i])
                        else:
                            td = tc.to(torch.bfloat16).to(devices[d_i])
                        held[d_i].append(td)
                        if dp_g == 0:
                            per_core_bytes[slot] += td.numel() * td.element_size()
                if per_core_bytes[slot] >= target_per_core:
                    break
    torch.accelerator.synchronize()
    load_sec = time.perf_counter() - t_l0

    # Probe hardware FP8 `torch._scaled_mm(..., out_dtype=torch.bfloat16)` support on 128-aligned shapes.
    scaled_mm_mode = "DISABLED"
    if use_fp8 and hasattr(torch, "_scaled_mm"):
        for cand_mode, scale_shape in (("SCALAR_SCALE", ()), ("TENSOR1D_SCALE", (1,))):
            try:
                pa = (torch.randn((128, 128), device=devices[0], dtype=torch.bfloat16) * 0.02).to(torch.float8_e4m3fn)
                pb_row = (torch.randn((128, 128), device=devices[0], dtype=torch.bfloat16) * 0.02).to(torch.float8_e4m3fn)
                ps = torch.ones(scale_shape, device=devices[0], dtype=torch.float32)
                pout = torch._scaled_mm(pa, pb_row.t(), scale_a=ps, scale_b=ps, out_dtype=torch.bfloat16)
                if isinstance(pout, tuple):
                    pout = pout[0]
                _ = pout.sum().item()
                torch.accelerator.synchronize()
                scaled_mm_mode = cand_mode
                break
            except Exception:
                continue

    scale_shape = () if scaled_mm_mode == "SCALAR_SCALE" else (1,)

    def _mk_w2d(n_out: int, k_in: int, dev: Any) -> torch.Tensor:
        w = torch.randn((n_out, k_in), device=dev, dtype=torch.bfloat16) * 0.02
        return w.to(torch.float8_e4m3fn).contiguous() if use_fp8 else w.contiguous()

    def _mk_w3d(n_exp: int, n_out: int, k_in: int, dev: Any) -> torch.Tensor:
        w = torch.randn((n_exp, n_out, k_in), device=dev, dtype=torch.bfloat16) * 0.02
        return w.to(torch.float8_e4m3fn).contiguous() if use_fp8 else w.contiguous()

    dp_ops: list[dict[str, Any]] = []
    for dp_g in range(dp_per_host):
        dev = devices[dp_g * tp_ep_size]
        dp_ops.append(
            {
                "dev": dev,
                "scale": torch.ones(scale_shape, device=dev, dtype=torch.float32),
                "wq": _mk_w2d(local_q_heads * head_dim, d_model, dev),
                "wk": _mk_w2d(local_kv_heads * head_dim, d_model, dev),
                "wv": _mk_w2d(local_kv_heads * head_dim, d_model, dev),
                "wo": _mk_w2d(d_model, local_q_heads * head_dim, dev),
                "rw": (torch.randn((local_experts, d_model), device=dev, dtype=torch.bfloat16) * 0.02).contiguous(),
                "egu": _mk_w3d(local_experts, 2 * moe_inter, d_model, dev),
                "edn": _mk_w3d(local_experts, d_model, moe_inter, dev),
                "sgu": _mk_w2d(2 * shared_inter, d_model, dev),
                "sdn": _mk_w2d(d_model, shared_inter, dev),
            }
        )

    def _proj2d(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if not use_fp8:
            return F.linear(x, w)
        n_out, k_in = w.shape
        m_rows = x.numel() // k_in
        if (
            scaled_mm_mode != "DISABLED"
            and m_rows % 128 == 0
            and k_in % 128 == 0
            and n_out % 128 == 0
        ):
            x_fp8 = x.reshape(m_rows, k_in).contiguous().to(torch.float8_e4m3fn)
            res = torch._scaled_mm(x_fp8, w.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
            if isinstance(res, tuple):
                res = res[0]
            return res.view(*x.shape[:-1], n_out)
        return F.linear(x, w.to(torch.bfloat16))

    def _hy3_core(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        rw: torch.Tensor,
        egu: torch.Tensor,
        edn: torch.Tensor,
        sgu: torch.Tensor,
        sdn: torch.Tensor,
        scale: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = _proj2d(x, wq, scale).view(b, s, local_q_heads, head_dim).transpose(1, 2)
        k = _proj2d(x, wk, scale).view(b, s, local_kv_heads, head_dim).transpose(1, 2).repeat_interleave(gqa_rep, dim=1)[:, :local_q_heads]
        v = _proj2d(x, wv, scale).view(b, s, local_kv_heads, head_dim).transpose(1, 2).repeat_interleave(gqa_rep, dim=1)[:, :local_q_heads]
        if s > 8192:
            at = torch.cat([F.scaled_dot_product_attention(qc, k, v, is_causal=False) for qc in q.split(4096, dim=2)], dim=2)
        else:
            at = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        h = x + _proj2d(at.transpose(1, 2).contiguous().view(b, s, -1), wo, scale)
        flat = h.view(-1, d_model)
        sg, su = _proj2d(flat, sgu, scale).chunk(2, dim=-1)
        out_m = _proj2d(F.silu(sg) * su, sdn, scale)
        if is_vectorized_moe:
            scores = F.softmax(F.linear(flat, rw), dim=-1)
            cap = max(16, ((flat.shape[0] * local_topk) // local_experts + 15) // 16 * 16)
            idx = torch.arange(local_experts * cap, device=x.device) % flat.shape[0]
            disp = flat.index_select(0, idx).view(local_experts, cap, d_model)
            egu_bf16 = egu.to(torch.bfloat16) if use_fp8 else egu
            edn_bf16 = edn.to(torch.bfloat16) if use_fp8 else edn
            g, u = torch.bmm(disp, egu_bf16.transpose(1, 2)).chunk(2, dim=-1)
            eo = torch.bmm(F.silu(g) * u, edn_bf16.transpose(1, 2)).view(-1, d_model)
            out_m = out_m + eo[: flat.shape[0]] * scores[:, :1]
        else:
            for e_i in range(local_experts):
                ge, ue = _proj2d(flat, egu[e_i], scale).chunk(2, dim=-1)
                out_m = out_m + _proj2d(F.silu(ge) * ue, edn[e_i], scale) * (1.0 / local_experts)
        return h + out_m.view(b, s, d_model)

    def _hy3_causal_eager(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        rw: torch.Tensor,
        egu: torch.Tensor,
        edn: torch.Tensor,
        sgu: torch.Tensor,
        sdn: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return _hy3_core(x, wq, wk, wv, wo, rw, egu, edn, sgu, sdn, scale, True)

    def _hy3_bidir_eager(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        rw: torch.Tensor,
        egu: torch.Tensor,
        edn: torch.Tensor,
        sgu: torch.Tensor,
        sdn: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return _hy3_core(x, wq, wk, wv, wo, rw, egu, edn, sgu, sdn, scale, False)

    causal_fn: Callable[..., torch.Tensor] = _hy3_causal_eager
    bidir_fn: Callable[..., torch.Tensor] = _hy3_bidir_eager
    compile_notes: dict[str, str] = {"ar": "EAGER", "diff": "EAGER"}

    if use_compile:
        try:
            causal_fn = torch.compile(_hy3_causal_eager, dynamic=False)
            compile_notes["ar"] = "COMPILED"
        except Exception as exc:
            compile_notes["ar"] = f"FALLBACK_EAGER({type(exc).__name__})"
        try:
            bidir_fn = torch.compile(_hy3_bidir_eager, dynamic=False)
            compile_notes["diff"] = "COMPILED"
        except Exception as exc:
            compile_notes["diff"] = f"FALLBACK_EAGER({type(exc).__name__})"

    def _hy3_layer(op: dict[str, Any], x: torch.Tensor, is_causal: bool) -> torch.Tensor:
        nonlocal causal_fn, bidir_fn
        if is_causal:
            try:
                return causal_fn(
                    x, op["wq"], op["wk"], op["wv"], op["wo"], op["rw"], op["egu"], op["edn"], op["sgu"], op["sdn"], op["scale"]
                )
            except Exception as exc:
                if causal_fn is not _hy3_causal_eager:
                    compile_notes["ar"] = f"FALLBACK_EAGER({type(exc).__name__})"
                    causal_fn = _hy3_causal_eager
                    return causal_fn(
                        x, op["wq"], op["wk"], op["wv"], op["wo"], op["rw"], op["egu"], op["edn"], op["sgu"], op["sdn"], op["scale"]
                    )
                raise
        try:
            return bidir_fn(
                x, op["wq"], op["wk"], op["wv"], op["wo"], op["rw"], op["egu"], op["edn"], op["sgu"], op["sdn"], op["scale"]
            )
        except Exception as exc:
            if bidir_fn is not _hy3_bidir_eager:
                compile_notes["diff"] = f"FALLBACK_EAGER({type(exc).__name__})"
                bidir_fn = _hy3_bidir_eager
                return bidir_fn(
                    x, op["wq"], op["wk"], op["wv"], op["wo"], op["rw"], op["egu"], op["edn"], op["sgu"], op["sdn"], op["scale"]
                )
            raise

    offload_health = check_offload_health()
    rows: list[dict[str, Any]] = []
    tasks = [
        ("T2I-Direct-1024x1024", "1024x1024", "image", 256, 0, 4096),
        ("T2I-ThinkRecaption-CoT-1024x1024", "1024x1024", "think_recaption", 512, 384, 4096),
        ("T2I-Wide-1280x768", "1280x768", "think_recaption", 512, 256, 3840),
        ("TI2I-MultiImageEdit-2Refs-1024x1024", "1024x1024", "think_recaption_ti2i", 4352, 384, 4096),
        ("T2I-UltraHD-2048x2048-LongSeq20K", "2048x2048", "think_recaption", 4096, 384, 16384),
    ]
    for task_name, res_str, bot_task, prompt_ctx_tok, cot_out_tok, img_lat_tok in tasks:
        if moe_kernel_mode == "EAGER_MOE" and task_name not in ("T2I-Direct-1024x1024", "T2I-ThinkRecaption-CoT-1024x1024"):
            continue
        try:
            x_ctxs = [torch.randn((1, prompt_ctx_tok, d_model), device=o["dev"], dtype=torch.bfloat16) * 0.02 for o in dp_ops]
            for o, xc in zip(dp_ops, x_ctxs):
                _ = _hy3_layer(o, xc, True)
            torch.accelerator.synchronize()

            t_ar0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=dp_per_host) as pool:

                def _pre(i: int) -> torch.Tensor:
                    hc = x_ctxs[i]
                    for _ in range(4):
                        hc = _hy3_layer(dp_ops[i], hc, True)
                    return hc

                list(pool.map(_pre, range(dp_per_host)))
            torch.accelerator.synchronize()
            cot_prefill_s = (time.perf_counter() - t_ar0) * (n_layers / 4.0)

            cot_decode_s = 0.0
            if cot_out_tok > 0:
                x_toks = [torch.randn((1, 16, d_model), device=o["dev"], dtype=torch.bfloat16) * 0.02 for o in dp_ops]
                for o, xt in zip(dp_ops, x_toks):
                    _ = _hy3_layer(o, xt, True)
                torch.accelerator.synchronize()
                t_d0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=dp_per_host) as pool:

                    def _dec(i: int) -> torch.Tensor:
                        hd = x_toks[i]
                        for _ in range(8):
                            hd = _hy3_layer(dp_ops[i], hd, True)
                        return hd

                    list(pool.map(_dec, range(dp_per_host)))
                torch.accelerator.synchronize()
                cot_decode_s = ((time.perf_counter() - t_d0) / 8.0) * n_layers * cot_out_tok

            cfg_batch = 1 if diff_steps <= 8 else 2
            x_imgs = [torch.randn((cfg_batch, img_lat_tok + 256, d_model), device=o["dev"], dtype=torch.bfloat16) * 0.02 for o in dp_ops]
            for o, xi in zip(dp_ops, x_imgs):
                _ = _hy3_layer(o, xi, False)
            torch.accelerator.synchronize()

            t_df0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=dp_per_host) as pool:

                def _dif(i: int) -> torch.Tensor:
                    hi = x_imgs[i]
                    for _ in range(4):
                        hi = _hy3_layer(dp_ops[i], hi, False)
                    return hi

                list(pool.map(_dif, range(dp_per_host)))
            torch.accelerator.synchronize()
            per_diff_step_s = (time.perf_counter() - t_df0) * (n_layers / 4.0)
            diff_total_s = per_diff_step_s * diff_steps
            e2e_s = cot_prefill_s + cot_decode_s + diff_total_s

            w_px, h_px = [int(z) for z in res_str.split("x")]
            img_ablation = benchmark_image_encode_ablation(width=w_px, height=h_px, batch=1, fmt="png")
            local_cpu_png_ms = float(img_ablation.get("local_cpu_png_ms", 0.0))
            png_offload_ok = bool(img_ablation.get("offload_ok", False))
            disagg_png_rtt_ms = float(img_ablation["disagg_cpu_rtt_ms"]) if png_offload_ok else None

            local_cpu_preproc_ms = 0.0
            disagg_preproc_rtt_ms: float | None = 0.0
            preproc_offload_ok = True
            if task_name == "TI2I-MultiImageEdit-2Refs-1024x1024":
                pre_ablation = benchmark_vision_preproc_ablation(
                    num_cams=2,
                    src_w=1024,
                    src_h=1024,
                    dst_w=512,
                    dst_h=512,
                )
                local_cpu_preproc_ms = float(pre_ablation.get("local_cpu_preproc_ms", 0.0))
                preproc_offload_ok = bool(pre_ablation.get("offload_ok", False))
                disagg_preproc_rtt_ms = float(pre_ablation["disagg_cpu_rtt_ms"]) if preproc_offload_ok else None

            offload_ok = png_offload_ok and preproc_offload_ok
            if offload_ok and disagg_png_rtt_ms is not None and disagg_preproc_rtt_ms is not None:
                disagg_cpu_rtt_ms: float | None = round(disagg_png_rtt_ms + disagg_preproc_rtt_ms, 2)
            else:
                disagg_cpu_rtt_ms = None

            total_local_cpu_ms = local_cpu_png_ms + local_cpu_preproc_ms
            eff_disagg_cpu_ms = disagg_cpu_rtt_ms if disagg_cpu_rtt_ms is not None else total_local_cpu_ms
            e2e_with_local_cpu_s = round(e2e_s + (total_local_cpu_ms / 1000.0), 3)
            e2e_with_disagg_cpu_s = round(e2e_s + (eff_disagg_cpu_ms / 1000.0), 3)

            rows.append(
                {
                    "variant": variant_label,
                    "moe_kernel": moe_kernel_mode,
                    "compiled": use_compile,
                    "compile_status": f"ar={compile_notes['ar']},diff={compile_notes['diff']}",
                    "fp8_mxu_mode": scaled_mm_mode if use_fp8 else "BF16",
                    "precision": precision_mode,
                    "tp_ep_degree": tp_ep_size,
                    "dp_per_host": dp_per_host,
                    "tpu_cores_detected": n_devs,
                    "task": task_name,
                    "bot_task": bot_task,
                    "resolution": res_str,
                    "total_sequence_tokens": prompt_ctx_tok + cot_out_tok + img_lat_tok,
                    "diff_infer_steps": diff_steps,
                    "weight_shard_gib_per_core": round(per_core_bytes[0] / (1024**3), 2),
                    "weight_load_s": round(load_sec, 2),
                    "stage1_cot_s": round(cot_prefill_s + cot_decode_s, 3),
                    "stage2_diff_step_ms": round(per_diff_step_s * 1000.0, 2),
                    "stage2_diff_total_s": round(diff_total_s, 3),
                    "e2e_latency_s": round(e2e_s, 3),
                    "offload_health_status": offload_health.get("status", "ok" if offload_ok else "unreachable"),
                    "offload_ok": offload_ok,
                    "local_cpu_png_ms": round(local_cpu_png_ms, 2),
                    "local_cpu_preproc_ms": round(local_cpu_preproc_ms, 2),
                    "disagg_png_rtt_ms": round(disagg_png_rtt_ms, 2) if disagg_png_rtt_ms is not None else None,
                    "disagg_preproc_rtt_ms": round(disagg_preproc_rtt_ms, 2) if disagg_preproc_rtt_ms is not None else None,
                    "disagg_cpu_rtt_ms": disagg_cpu_rtt_ms,
                    "e2e_with_local_cpu_s": e2e_with_local_cpu_s,
                    "e2e_with_disagg_cpu_s": e2e_with_disagg_cpu_s,
                    "images_per_min_replica": round(60.0 / max(e2e_s, 1e-6), 2),
                    "images_per_min_disagg_replica": round(60.0 / max(e2e_with_disagg_cpu_s, 1e-6), 2),
                    "disagg_stage2_only_images_per_min_replica": round(60.0 / max(diff_total_s, 1e-6), 2),
                    "status": "PASS",
                }
            )
            del x_ctxs, x_imgs
            torch.accelerator.synchronize()
        except Exception as exc:
            rows.append(
                {
                    "variant": variant_label,
                    "moe_kernel": moe_kernel_mode,
                    "compiled": use_compile,
                    "precision": precision_mode,
                    "tp_ep_degree": tp_ep_size,
                    "dp_per_host": dp_per_host,
                    "task": task_name,
                    "diff_infer_steps": diff_steps,
                    "status": f"FAIL({type(exc).__name__}: {str(exc)[:160]})",
                }
            )

    out_json.write_text(json.dumps(rows, indent=2))


def write_hy3_summary(
    out_dir: Path,
    bucket: str,
    slice_name: str,
    chip_type: str,
    num_hosts: int,
    cores_per_host: int,
    payload: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    js_bytes = json.dumps(payload, indent=2).encode("utf-8")
    (out_dir / "summary.json").write_bytes(js_bytes)

    offload_status = payload.get("cpu_offload_health", {}).get("status", "unknown")
    lines = [
        f"# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B Total / 13B Active 64-Expert MoE) — `{slice_name}` (`{chip_type}`)",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Hardware:** `{num_hosts}` hosts x `{cores_per_host}` TPU cores/host = `{num_hosts * cores_per_host}` total TPU cores",
        f"- **Disaggregated CPU Offload Health:** `{offload_status}`",
        "",
        "Compares `COMPILED_VECTORIZED_GROUPED_MM` (`torch.compile(..., dynamic=False)`), `VECTORIZED_GROUPED_MM`, and `EAGER_MOE` across BF16 and hardware FP8 MXU (`torch._scaled_mm`), alongside PNG / multi-reference vision CPU offload ablation (`multimodal-cpu-offload-svc`).",
        "",
        "| Checkpoint | MoE Kernel | Precision | TP/EP | Cluster DP | Task | Res | Steps | Stage-1 CoT (s) | Diff Step (ms) | TPU E2E (s) | Local PNG (ms) | Disagg RTT (ms) | E2E Local (s) | E2E Disagg (s) | Cluster img/min (Disagg) | Status |",
        "|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in payload["results"]:
        lines.append(
            f"| `{r.get('variant')}` | `{r.get('moe_kernel')}` | `{r.get('precision')}` | `{r.get('tp_ep_degree')}` | "
            f"`{r.get('cluster_dp_replicas', '-')}` | `{r.get('task')}` | `{r.get('resolution', '-')}` | "
            f"`{r.get('diff_infer_steps')}` | {r.get('stage1_cot_s', '-')} | "
            f"{r.get('stage2_diff_step_ms', '-')} | {r.get('e2e_latency_s', '-')} | "
            f"{r.get('local_cpu_png_ms', '-')} | {r.get('disagg_cpu_rtt_ms', '-')} | "
            f"{r.get('e2e_with_local_cpu_s', '-')} | {r.get('e2e_with_disagg_cpu_s', '-')} | "
            f"{r.get('cluster_images_per_min_disagg', '-')} | `{r.get('status')}` |"
        )

    md_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    (out_dir / "SUMMARY.md").write_bytes(md_bytes)
    gcs_upload_bytes(bucket, f"omni-world-results/hunyuan-image-3/{slice_name}/summary.json", js_bytes)
    gcs_upload_bytes(
        bucket,
        f"omni-world-results/hunyuan-image-3/{slice_name}/SUMMARY.md",
        md_bytes,
        content_type="text/markdown",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-only", action="store_true", help="Cooperatively stage both checkpoints to GCS and exit.")
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-model-dir", default="")
    parser.add_argument("--worker-variant", default="")
    parser.add_argument("--worker-steps", type=int, default=8)
    parser.add_argument("--worker-tp", type=int, default=4)
    parser.add_argument("--worker-precision", default="BF16")
    parser.add_argument(
        "--worker-moe",
        choices=["COMPILED_VECTORIZED_GROUPED_MM", "VECTORIZED_GROUPED_MM", "EAGER_MOE"],
        default="COMPILED_VECTORIZED_GROUPED_MM",
    )
    parser.add_argument("--worker-out", default="")
    parser.add_argument("--chip-type", choices=["v7x", "v5p"], default="v7x")
    parser.add_argument("--slice-name", default="v7x-8")
    parser.add_argument("--num-hosts", type=int, default=2)
    parser.add_argument("--cores-per-host", type=int, default=8)
    parser.add_argument("--bucket", default="YOUR_GCS_BUCKET")
    parser.add_argument("--use-shm", action="store_true")
    parser.add_argument("--output-dir", default="/models/omni-world-results/hunyuan-image-3")
    args = parser.parse_args()

    if args.worker_mode:
        run_child_hy3_worker(
            Path(args.worker_model_dir),
            args.worker_variant,
            args.worker_steps,
            args.worker_tp,
            args.worker_precision,
            args.worker_moe,
            Path(args.worker_out),
        )
        return 0

    rank = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    variants = [
        ("tencent/HunyuanImage-3.0-Instruct-Distil", "models/tencent/HunyuanImage-3-Instruct-Distil", "HunyuanImage-3-Instruct-Distil", 8),
        ("tencent/HunyuanImage-3.0-Instruct", "models/tencent/HunyuanImage-3-Instruct", "HunyuanImage-3-Instruct", 50),
    ]

    if args.stage_only:
        log(rank, f"Running cooperative HF -> GCS staging (--stage-only, wait_all={rank == 0}) on {args.slice_name}")
        for repo_id, gcs_pref, clean_name, _ in variants:
            assert "." not in clean_name
            ensure_model_staged_to_gcs(
                repo_id=repo_id,
                bucket=args.bucket,
                dst_prefix=gcs_pref,
                worker_rank=rank,
                num_workers=args.num_hosts,
                threads_per_worker=8,
                wait_all=(rank == 0),
            )
        log(rank, "Completed --stage-only step for both HunyuanImage-3.0 checkpoints.")
        return 0

    stage_infos = {}
    for repo_id, gcs_pref, clean_name, _ in variants:
        stage_infos[clean_name] = ensure_model_staged_to_gcs(
            repo_id=repo_id,
            bucket=args.bucket,
            dst_prefix=gcs_pref,
            worker_rank=rank,
            num_workers=args.num_hosts,
            threads_per_worker=8,
            wait_all=True,
        )
    if rank != 0:
        log(rank, "Cooperative GCS staging complete; exiting cleanly.")
        return 0

    offload_health = check_offload_health()
    log(rank, f"CPU offload service health: {offload_health}")

    child_env = start_single_host_metadata_env(args.chip_type)
    out_dir = Path(args.output_dir) / args.slice_name
    results: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}

    for _repo_id, gcs_pref, clean_name, diff_steps in variants:
        local_dir = (Path("/dev/shm") / clean_name) if args.use_shm else (Path("/models") / gcs_pref)
        try:
            if args.use_shm:
                pull_gcs_prefix_to_local(args.bucket, gcs_pref, local_dir, threads=16)

            manifests[clean_name] = verify_dotfree_checkpoint(local_dir)
            tp_list = [4, 8] if args.cores_per_host >= 8 else [4]
            for tp_ep in tp_list:
                combos: list[tuple[str, str]] = [
                    ("COMPILED_VECTORIZED_GROUPED_MM", "BF16"),
                    ("COMPILED_VECTORIZED_GROUPED_MM", "FP8_DYNAMIC"),
                    ("VECTORIZED_GROUPED_MM", "BF16"),
                    ("VECTORIZED_GROUPED_MM", "FP8_DYNAMIC"),
                ]
                if clean_name.endswith("Distil") and tp_ep == tp_list[-1]:
                    combos.append(("EAGER_MOE", "BF16"))
                for moe_k, prec in combos:
                    if clean_name == "HunyuanImage-3-Instruct" and prec == "FP8_DYNAMIC" and tp_ep == 8 and moe_k != "COMPILED_VECTORIZED_GROUPED_MM":
                        continue
                    out_js = Path(f"/dev/shm/hy3_{clean_name}_tp{tp_ep}_{prec}_{moe_k}.json")
                    out_js.unlink(missing_ok=True)
                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker-mode",
                        f"--worker-model-dir={local_dir}",
                        f"--worker-variant={clean_name}",
                        f"--worker-steps={diff_steps}",
                        f"--worker-tp={tp_ep}",
                        f"--worker-precision={prec}",
                        f"--worker-moe={moe_k}",
                        f"--worker-out={out_js}",
                    ]
                    log(rank, f"Launching {clean_name} (steps={diff_steps}) TP/EP={tp_ep} {moe_k} {prec}...")
                    subprocess.run(cmd, env=child_env, check=False, timeout=2400)
                    if out_js.exists():
                        rows = json.loads(out_js.read_text())
                        for row in rows:
                            dp_h = int(row.get("dp_per_host", max(1, args.cores_per_host // tp_ep)))
                            c_dp = dp_h * args.num_hosts
                            row["chip_type"] = args.chip_type
                            row["cluster_dp_replicas"] = c_dp
                            if "images_per_min_replica" in row:
                                row["cluster_images_per_min"] = round(row["images_per_min_replica"] * c_dp, 2)
                                row["cluster_disagg_images_per_min"] = round(
                                    row["disagg_stage2_only_images_per_min_replica"] * c_dp,
                                    2,
                                )
                            if "e2e_with_local_cpu_s" in row:
                                row["cluster_images_per_min_local_cpu"] = round(
                                    (60.0 / max(float(row["e2e_with_local_cpu_s"]), 1e-6)) * c_dp,
                                    2,
                                )
                            if "e2e_with_disagg_cpu_s" in row:
                                row["cluster_images_per_min_disagg"] = round(
                                    (60.0 / max(float(row["e2e_with_disagg_cpu_s"]), 1e-6)) * c_dp,
                                    2,
                                )
                            results.append(row)
                        out_js.unlink(missing_ok=True)
                        payload = {
                            "updated_at_utc": utc_iso(),
                            "models": ["tencent/HunyuanImage-3.0-Instruct-Distil", "tencent/HunyuanImage-3.0-Instruct"],
                            "slice_name": args.slice_name,
                            "chip_type": args.chip_type,
                            "num_hosts": args.num_hosts,
                            "cores_per_host": args.cores_per_host,
                            "cpu_offload_health": offload_health,
                            "stage_infos": stage_infos,
                            "manifests": manifests,
                            "results": results,
                        }
                        write_hy3_summary(
                            out_dir,
                            args.bucket,
                            args.slice_name,
                            args.chip_type,
                            args.num_hosts,
                            args.cores_per_host,
                            payload,
                        )
        finally:
            if args.use_shm and local_dir.exists():
                shutil.rmtree(local_dir, ignore_errors=True)

    log(rank, f"Completed HunyuanImage-3.0 benchmark suite on {args.slice_name}; total_rows={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
