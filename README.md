# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B-A13B MoE) Serving on Cloud TPU `v5p` & `v7x`

tencent/HunyuanImage-3.0-Instruct (50-step) & HunyuanImage-3.0-Instruct-Distil (8-step) 83B-A13B 64-Expert MoE Unified CoT + Diffusion Serving on Cloud TPU v5p & v7x.

---

## 1. Model & System Architecture

| Component | Specification |
|---|---|
| **Checkpoints** | [`tencent/HunyuanImage-3.0-Instruct`](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct) (`50`-step) & [`HunyuanImage-3.0-Instruct-Distil`](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) (`8`-step) |
| **Architecture** | `83B` Total / `13B` Active `64`-Expert Unified Autoregressive Multimodal CoT + Transfusion MoE Diffusion + `32x` Spatial VAE |
| **Resolutions** | `1024x1024`, `1536x1024`, and `2048x2048` (`2K` high-resolution generation + TI2I instruction editing) |
| **TPU Execution** | `TorchTPU` (`eager` & `torch.compile(..., dynamic=False)`), `BF16` & `FP8_WEIGHT_DYNAMIC` (`out_dtype=bfloat16`), `TP=4` / `TP=8` |
| **Disaggregated CPU Offload** | `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512): `POST /encode_image` parallel PNG/WebP encoding & `POST /preprocess_vision` |

---

## 2. Disaggregated CPU Offloading Architecture (`multimodal-cpu-offload-svc`)

All non-TPU pre-processing and media encoding work is offloaded over intra-cluster `ClusterIP` HTTP (`port 8090`) to dedicated `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512, `176 GiB` DDR5) CPU pods (`scripts/multimodal_cpu_offload_server.py`), keeping TPU host CPUs and Python threads dedicated to XLA/PJRT dispatch:

- `POST /encode` — Multi-threaded `ffmpeg` (`libx264` + `48 kHz` stereo `AAC`) video/audio MP4 muxing.
- `POST /encode_image` — Thread-pooled `Pillow` `PNG` / `WebP` / `JPEG` batch image encoding.
- `POST /encode_wav` — Vectorized `float32` / `int16` audio normalization + `RIFF/WAV` packet framing (`24 kHz` / `48 kHz`).
- `POST /preprocess_vision` — Multi-camera RGB image decode, bicubic resize, and `bfloat16`/`float32` normalization.

---

## 3. Hardware Platforms Evaluated

| Accelerator | Slice Topologies | HBM Capacity & Bandwidth | Host Memory (`/dev/shm`) |
|---|---|---|---|
| **Cloud TPU `v5p`** | `v5p-16` (`2x2x4`, `4` hosts × `4` chips) & `v5p-32` (`2x4x4`, `8` hosts × `4` chips) | `95 GiB` HBM2e / chip (`2,765 GB/s` / chip, `459` BF16 TFLOP/s) | `462 GiB` RAM (`380 GiB` tmpfs) |
| **Cloud TPU `v7x` (Ironwood)** | `v7x-4` (`2x2x1`, `8` TensorCores), `v7x-8` (`2x2x2`, `16` TensorCores), & `v7x-16` (`2x2x4`, `32` TensorCores) | `192 GiB` HBM3e / chip (`96 GiB` / TensorCore, `7,370 GB/s` / chip, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip) | `944 GiB` RAM (`850 GiB` tmpfs) |

---

## 4. Benchmark Results (`v5p` & `v7x`)

Measured results linked 1:1 to raw JSON and Markdown summaries under [`results/`](./results/):

- **`v5p-32`** (`5` benchmark rows, updated `2026-11-02T19:12:22.017812+00:00`): [`results/v5p-32/SUMMARY.md`](./results/v5p-32/SUMMARY.md) · [`results/v5p-32/summary.json`](./results/v5p-32/summary.json)

# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B Total / 13B Active 64-Expert MoE) — `v5p-32` (`v5p`)

- **Updated (UTC):** `2026-11-02T19:12:22.017812+00:00`
- **Hardware:** `8` hosts x `4` TPU cores/host = `32` total TPU cores
- **Disaggregated CPU Offload Health:** `ok`

Compares `COMPILED_VECTORIZED_GROUPED_MM` (`torch.compile(..., dynamic=False)`), `VECTORIZED_GROUPED_MM`, and `EAGER_MOE` across BF16 and hardware FP8 MXU (`torch._scaled_mm`), alongside PNG / multi-reference vision CPU offload ablation (`multimodal-cpu-offload-svc`).

| Checkpoint | MoE Kernel | Precision | TP/EP | Cluster DP | Task | Res | Steps | Stage-1 CoT (s) | Diff Step (ms) | TPU E2E (s) | Local PNG (ms) | Disagg RTT (ms) | E2E Local (s) | E2E Disagg (s) | Cluster img/min (Disagg) | Status |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `4` | `8` | `T2I-Direct-1024x1024` | `1024x1024` | `8` | 0.066 | 148.65 | 1.255 | 195.76 | 151.2 | 1.451 | 1.406 | 341.39 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `4` | `8` | `T2I-ThinkRecaption-CoT-1024x1024` | `1024x1024` | `8` | 19.602 | 145.29 | 20.765 | 154.39 | 150.96 | 20.919 | 20.916 | 22.95 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `4` | `8` | `T2I-Wide-1280x768` | `1280x768` | `8` | 12.748 | 142.67 | 13.889 | 146.74 | 163.64 | 14.036 | 14.053 | 34.16 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `4` | `8` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `1024x1024` | `8` | 19.182 | 144.75 | 20.34 | 156.2 | 186.58 | 20.524 | 20.527 | 23.38 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `4` | `8` | `T2I-UltraHD-2048x2048-LongSeq20K` | `2048x2048` | `8` | 18.962 | 494.24 | 22.916 | 609.55 | 598.15 | 23.525 | 23.514 | 20.41 | `PASS` |

---

## 5. Repository Layout & Reproduction

- [`scripts/run_hunyuan_image3_tpu_bench.py`](./scripts/run_hunyuan_image3_tpu_bench.py) — Main TPU `v5p` / `v7x` benchmark driver (cooperative zero-disk HF-to-GCS staging, `/dev/shm` hydration, single-host PJRT isolation, `TorchTPU` / `vLLM` execution, and A/B CPU offload ablation).
- [`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) & [`scripts/cpu_offload_client.py`](./scripts/cpu_offload_client.py) — Disaggregated CPU offload HTTP server and client.
- [`scripts/gcs_stager.py`](./scripts/gcs_stager.py) & [`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py) — Multi-host GCS shard stager and single-host `libtpu` metadata hook.
- [`k8s/`](./k8s/) — Kubernetes Deployment & Indexed Job manifests for CPU offload pool, TPU `v5p`, and TPU `v7x`.

```bash
# 1. Deploy the disaggregated CPU offload service
kubectl apply -f k8s/multimodal-cpu-offload-service.yaml

# 2. Launch the TPU v5p / v7x Indexed Job
kubectl apply -f k8s/v5p32-cosmos3-and-hunyuan-image3.yaml
```
