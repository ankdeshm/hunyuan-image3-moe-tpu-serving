# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B-A13B MoE) Serving on Cloud TPU `v5p` & `v7x` — Slice `v7x-8`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **Stage 1 CoT & Stage 2 Diffusion TPU Compute (`VECTORIZED_GROUPED_MM` / `COMPILED_VECTORIZED_GROUPED_MM`):** **Local-Shard Layer-Scaled 64-Expert MoE + GQA Operator Microbenchmark** (`run_child_hy3_worker` in [`scripts/run_hunyuan_image3_tpu_bench.py`](../../scripts/run_hunyuan_image3_tpu_bench.py)) loading `165 GiB / tp_ep_size` real `tencent/HunyuanImage-3.0` `safetensors` shards onto HBM and timing `4` measured prefill/diffusion layers (`× 8 -> 32L`) plus `8` measured CoT decode layers (`× 4 -> 32L`) over `64 // tp_ep_size = 16` local experts (`torch.bmm`) **excluding inter-chip EP `AllToAll` / TP `AllReduce` collectives**.
> - **Disaggregated Image Encoding & TI2I Preprocessing (`POST /encode_image`, `POST /preprocess_vision`):** **100% Real End-to-End HTTP + AVX-512 Pillow `PNG`/`WebP` Encoding & Vision Preprocessing** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`), achieving **`1.09x–1.29x` synchronous PNG speedup** on `v5p` while isolating TPU host CPUs from Pillow GIL contention.

### Slice `v7x-8` (`v7x`, `TP/EP=4 × DP=2`, `5` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-8/SUMMARY.md`](./SUMMARY.md) · [`results/v7x-8/summary.json`](./summary.json) *(updated `2026-11-02T20:11:00.222155+00:00`)*

| Checkpoint Variant | MoE Kernel | Task & Resolution | Seq Tokens | Steps | Stage-1 CoT (s) | Stage-2 Step (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Local (s) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `8` | `0.046` | `68.09` | `119.9` | **`146.6`** | `0.710` | **`0.737`** | **`162.82`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `8` | `14.039` | `63.52` | `115.7` | **`144.3`** | `14.663` | **`14.692`** | **`8.17`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `8` | `9.243` | `63.79` | `106.9` | **`139.2`** | `9.860` | **`9.893`** | **`12.13`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `8` | `13.057` | `64.13` | `118.0` | **`193.9`** | `13.708` | **`13.790`** | **`8.70`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `8` | `13.309` | `194.85` | `465.6` | **`594.2`** | `15.334` | **`15.462`** | **`7.76`** |
