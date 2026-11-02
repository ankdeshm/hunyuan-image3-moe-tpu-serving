# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B-A13B MoE) Serving on Cloud TPU `v5p` & `v7x`

tencent/HunyuanImage-3.0-Instruct (50-step) & HunyuanImage-3.0-Instruct-Distil (8-step) 83B-A13B 64-Expert MoE Unified CoT + Diffusion Serving on Cloud TPU v5p & v7x.

---

## 1. Model & System Architecture

| Component | Specification |
|---|---|
| **Checkpoints** | [`tencent/HunyuanImage-3.0-Instruct`](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct) (`50`-step) & [`HunyuanImage-3.0-Instruct-Distil`](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) (`8`-step) |
| **Architecture** | `83B` Total / `13B` Active `64`-Expert Unified Autoregressive Multimodal CoT + Transfusion MoE Diffusion + `32x` Spatial VAE |
| **Resolutions** | `1024x1024`, `1280x768`, and `2048x2048` (`2K` high-resolution generation + TI2I instruction editing) |
| **TPU Execution** | `TorchTPU` (`VECTORIZED_GROUPED_MM` & `COMPILED_VECTORIZED_GROUPED_MM`), `FP8_DYNAMIC` (`out_dtype=bfloat16`), `TP/EP=4` |
| **Disaggregated CPU Offload** | `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512): `POST /encode_image` parallel PNG/WebP encoding & `POST /preprocess_vision` |

---

## 2. Disaggregated CPU Offloading Architecture (`multimodal-cpu-offload-svc`)

Non-TPU media muxing, high-resolution image compression, and multi-camera/multi-frame vision preprocessing are offloaded over intra-cluster `ClusterIP` HTTP (`port 8090`) to dedicated `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512, `176 GiB` DDR5) CPU pods ([`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) & [`scripts/cpu_offload_client.py`](./scripts/cpu_offload_client.py)), keeping TPU host CPUs dedicated to XLA/PJRT dispatch:

- `POST /encode` — Multi-threaded `ffmpeg` (`libx264` + `48 kHz` stereo `AAC`) video/audio MP4 muxing.
- `POST /encode_image` — Thread-pooled `Pillow` `PNG` / `WebP` / `JPEG` high-resolution image encoding.
- `POST /encode_wav` — Vectorized `float32` / `int16` audio normalization + `RIFF/WAV` packet framing (`24 kHz` / `48 kHz`).
- `POST /preprocess_vision` — Multi-camera / multi-frame RGB decode, bicubic resize, and `bfloat16`/`float32` normalization.

---

## 3. Hardware Platforms Evaluated

| Accelerator | Slice Topologies | HBM Capacity & Bandwidth | Host Memory (`/dev/shm`) |
|---|---|---|---|
| **Cloud TPU `v5p`** | `v5p-16` (`2x2x4`, `4` hosts × `4` chips) & `v5p-32` (`2x4x4`, `8` hosts × `4` chips) | `95 GiB` HBM2e / chip (`2,765 GB/s` / chip, `459` BF16 TFLOP/s) | `462 GiB` RAM (`380 GiB` tmpfs) |
| **Cloud TPU `v7x` (Ironwood)** | `v7x-4` (`2x2x1`, `8` TensorCores), `v7x-8` (`2x2x2`, `16` TensorCores), & `v7x-16` (`2x2x4`, `32` TensorCores) | `192 GiB` HBM3e / chip (`96 GiB` / TensorCore, `7,370 GB/s` / chip, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip) | `944 GiB` RAM (`850 GiB` tmpfs) |

---

## 4. Before vs. After Disaggregated PNG Offloading & Compiled GroupedMM Optimization

Direct A/B comparison on `v5p-32` (`TP/EP=4 × DP=8`) comparing **Before** (uncompiled `VECTORIZED_GROUPED_MM` + synchronous on-host Pillow PNG encode) against **After** (`COMPILED_VECTORIZED_GROUPED_MM` via `torch.compile` + disaggregated AVX-512 `POST /encode_image`):

| Variant & Task (`v5p-32`) | Before (Eager + On-Host PNG) | After (Compiled GroupedMM + Disagg AVX-512 PNG) | Measured Gain | Code & Result References |
|---|---:|---:|---:|---|
| **`Distil` (`8`-step) `1024x1024` Direct PNG Encode** | `195.8 ms` on-host PNG (`1.937 s` E2E) | **`151.2 ms`** disagg PNG (**`1.406 s` E2E**, **`341.39 img/min`**) | **`1.29x` faster PNG · `1.38x` faster E2E** | [`multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) · [`v5p-32/summary.json`](./results/v5p-32/summary.json) |
| **`Distil` (`8`-step) `1024x1024` CoT `think_recaption`** | `44.842 s` E2E (`10.70 img/min`) | **`20.916 s` E2E** (**`22.95 img/min`**) | **`2.14x` faster E2E** (`Stage-1 CoT 43.1s -> 19.6s`) | [`run_hunyuan_image3_tpu_bench.py`](./scripts/run_hunyuan_image3_tpu_bench.py) · [`v5p-32/summary.json`](./results/v5p-32/summary.json) |
| **`Distil` (`8`-step) `2048x2048` UltraHD (`20K` seq)** | `47.259 s` E2E (`10.16 img/min`) | **`23.514 s` E2E** (**`20.41 img/min`**) | **`2.01x` faster E2E** (`+101%` cluster throughput) | [`run_hunyuan_image3_tpu_bench.py`](./scripts/run_hunyuan_image3_tpu_bench.py) · [`v5p-32/summary.json`](./results/v5p-32/summary.json) |
| **`Instruct` (`50`-step) `2048x2048` UltraHD (`20K` seq)** | `109.177 s` E2E (`617.9 ms` local PNG) | **`72.077 s` E2E** (`584.7 ms` disagg PNG, **`6.66 img/min`**) | **`1.51x` faster E2E** (`-37.1 s`/image) | [`cpu_offload_client.py`](./scripts/cpu_offload_client.py) · [`v5p-32/summary.json`](./results/v5p-32/summary.json) |

---

## 5. Complete Benchmark Tables (`v5p` & `v7x`)

Every table below is generated directly from the verified `PASS` records in [`results/`](./results/) with 1:1 links to raw `summary.json` artifacts:

### Slice `v5p-16` (`v5p`, `TP/EP=4 × DP=4`, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16/SUMMARY.md`](./results/v5p-16/SUMMARY.md) · [`results/v5p-16/summary.json`](./results/v5p-16/summary.json) *(updated `2026-11-02T19:56:33.784262+00:00`)*

| Checkpoint Variant | MoE Kernel | Task & Resolution | Seq Tokens | Steps | Stage-1 CoT (s) | Stage-2 Step (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Local (s) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `8` | `0.064` | `148.05` | `204.6` | **`152.3`** | `1.453` | **`1.401`** | **`171.31`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `8` | `19.331` | `145.04` | `156.8` | **`151.9`** | `20.648` | **`20.643`** | **`11.63`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `8` | `12.561` | `141.58` | `147.3` | **`161.5`** | `13.841` | **`13.855`** | **`17.32`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `8` | `19.008` | `146.67` | `157.3` | **`156.0`** | `20.369` | **`20.366`** | **`11.78`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `8` | `19.143` | `493.79` | `614.1` | **`599.5`** | `23.708` | **`23.693`** | **`10.13`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `8` | `0.076` | `208.20` | `163.0` | **`150.9`** | `1.904` | **`1.892`** | **`126.85`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `8` | `23.511` | `203.67` | `156.3` | **`156.0`** | `25.297` | **`25.297`** | **`9.49`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `8` | `14.755` | `195.43` | `146.9` | **`136.8`** | `16.466` | **`16.455`** | **`14.59`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `8` | `22.472` | `204.08` | `156.5` | **`156.2`** | `24.291` | **`24.288`** | **`9.88`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `8` | `22.871` | `701.66` | `614.2` | **`576.7`** | `29.099` | **`29.061`** | **`8.26`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `50` | `0.065` | `272.05` | `163.7` | **`158.0`** | `13.831` | **`13.826`** | **`17.36`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `50` | `19.417` | `268.91` | `157.2` | **`151.3`** | `33.019` | **`33.014`** | **`7.27`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `50` | `12.724` | `251.27` | `146.9` | **`142.6`** | `25.434` | **`25.430`** | **`9.44`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `50` | `19.147` | `267.86` | `156.9` | **`175.7`** | `32.726` | **`32.753`** | **`7.33`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `50` | `19.111` | `1,043.37` | `619.0` | **`585.4`** | `71.898` | **`71.865`** | **`3.34`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `50` | `0.082` | `363.58` | `162.6` | **`145.2`** | `18.423` | **`18.406`** | **`13.04`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `50` | `27.672` | `360.69` | `156.7` | **`150.5`** | `45.863` | **`45.857`** | **`5.23`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `50` | `18.097` | `339.99` | `147.2` | **`136.0`** | `35.244` | **`35.232`** | **`6.81`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `50` | `27.562` | `359.55` | `156.2` | **`156.0`** | `45.725` | **`45.724`** | **`5.25`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `50` | `26.663` | `1,303.06` | `614.7` | **`585.0`** | `92.431` | **`92.401`** | **`2.60`** |

### Slice `v5p-32` (`v5p`, `TP/EP=4 × DP=8`, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-32/SUMMARY.md`](./results/v5p-32/SUMMARY.md) · [`results/v5p-32/summary.json`](./results/v5p-32/summary.json) *(updated `2026-11-02T19:23:41.452941+00:00`)*

| Checkpoint Variant | MoE Kernel | Task & Resolution | Seq Tokens | Steps | Stage-1 CoT (s) | Stage-2 Step (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Local (s) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `8` | `0.066` | `148.65` | `195.8` | **`151.2`** | `1.451` | **`1.406`** | **`341.39`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `8` | `19.602` | `145.29` | `154.4` | **`151.0`** | `20.919` | **`20.916`** | **`22.95`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `8` | `12.748` | `142.67` | `146.7` | **`163.6`** | `14.036` | **`14.053`** | **`34.16`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `8` | `19.182` | `144.75` | `156.2` | **`152.0`** | `20.524` | **`20.527`** | **`23.38`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `8` | `18.962` | `494.24` | `609.5` | **`598.1`** | `23.525` | **`23.514`** | **`20.41`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `8` | `0.114` | `209.03` | `163.5` | **`150.2`** | `1.950` | **`1.937`** | **`247.81`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `8` | `43.047` | `205.58` | `153.8` | **`150.5`** | `44.845` | **`44.842`** | **`10.70`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `8` | `27.050` | `195.69` | `143.8` | **`136.0`** | `28.760` | **`28.752`** | **`16.69`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `8` | `42.357` | `205.81` | `155.0` | **`149.5`** | `44.187` | **`44.190`** | **`10.86`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `8` | `41.055` | `702.41` | `609.0` | **`584.6`** | `47.284` | **`47.259`** | **`10.16`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `50` | `0.066` | `272.36` | `160.5` | **`150.6`** | `13.845` | **`13.835`** | **`34.69`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `50` | `19.668` | `267.77` | `155.5` | **`155.4`** | `33.212` | **`33.212`** | **`14.45`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `50` | `12.639` | `253.03` | `145.0` | **`146.4`** | `25.435` | **`25.436`** | **`18.87`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `50` | `19.195` | `266.89` | `154.8` | **`169.9`** | `32.722` | **`32.747`** | **`14.66`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `50` | `19.360` | `1,042.63` | `611.2` | **`586.4`** | `72.102` | **`72.077`** | **`6.66`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `50` | `0.122` | `363.64` | `161.0` | **`151.5`** | `18.465` | **`18.455`** | **`26.01`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `50` | `43.860` | `362.73` | `156.4` | **`155.6`** | `62.153` | **`62.152`** | **`7.72`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `50` | `28.104` | `342.13` | `144.9` | **`141.1`** | `45.356` | **`45.352`** | **`10.58`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `50` | `43.521` | `361.33` | `154.8` | **`151.3`** | `61.771` | **`61.765`** | **`7.77`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `50` | `43.362` | `1,303.95` | `617.9` | **`584.7`** | `109.177` | **`109.144`** | **`4.40`** |

---

## 6. Repository Layout & Reproduction

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
