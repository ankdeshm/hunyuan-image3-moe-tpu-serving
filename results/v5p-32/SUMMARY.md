# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B-A13B MoE) Serving on Cloud TPU `v5p` & `v7x` — Slice `v5p-32`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **Stage 1 CoT & Stage 2 Diffusion TPU Compute (`VECTORIZED_GROUPED_MM` / `COMPILED_VECTORIZED_GROUPED_MM`):** **Local-Shard Layer-Scaled 64-Expert MoE + GQA Operator Microbenchmark** (`run_child_hy3_worker` in [`scripts/run_hunyuan_image3_tpu_bench.py`](../../scripts/run_hunyuan_image3_tpu_bench.py)) loading `165 GiB / tp_ep_size` real `tencent/HunyuanImage-3.0` `safetensors` shards onto HBM and timing `4` measured prefill/diffusion layers (`× 8 -> 32L`) plus `8` measured CoT decode layers (`× 4 -> 32L`) over `64 // tp_ep_size = 16` local experts (`torch.bmm`) **excluding inter-chip EP `AllToAll` / TP `AllReduce` collectives**.
> - **Disaggregated Image Encoding & TI2I Preprocessing (`POST /encode_image`, `POST /preprocess_vision`):** **100% Real End-to-End HTTP + AVX-512 Pillow `PNG`/`WebP` Encoding & Vision Preprocessing** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`), achieving **`1.09x–1.29x` synchronous PNG speedup** on `v5p` while isolating TPU host CPUs from Pillow GIL contention.

### Slice `v5p-32` (`v5p`, `TP/EP=4 × DP=8`, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-32/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-32/summary.json`](./summary.json) *(updated `2026-11-02T19:23:41.452941+00:00`)*

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
