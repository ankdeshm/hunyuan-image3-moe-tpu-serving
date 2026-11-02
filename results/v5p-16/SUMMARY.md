# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B-A13B MoE) Serving on Cloud TPU `v5p` & `v7x` — Slice `v5p-16`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **Stage 1 CoT & Stage 2 Diffusion TPU Compute (`VECTORIZED_GROUPED_MM` / `COMPILED_VECTORIZED_GROUPED_MM`):** **Local-Shard Layer-Scaled 64-Expert MoE + GQA Operator Microbenchmark** (`run_child_hy3_worker` in [`scripts/run_hunyuan_image3_tpu_bench.py`](../../scripts/run_hunyuan_image3_tpu_bench.py)) loading `165 GiB / tp_ep_size` real `tencent/HunyuanImage-3.0` `safetensors` shards onto HBM and timing `4` measured prefill/diffusion layers (`× 8 -> 32L`) plus `8` measured CoT decode layers (`× 4 -> 32L`) over `64 // tp_ep_size = 16` local experts (`torch.bmm`) **excluding inter-chip EP `AllToAll` / TP `AllReduce` collectives**.
> - **Disaggregated Image Encoding & TI2I Preprocessing (`POST /encode_image`, `POST /preprocess_vision`):** **100% Real End-to-End HTTP + AVX-512 Pillow `PNG`/`WebP` Encoding & Vision Preprocessing** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`), achieving **`1.09x–1.29x` synchronous PNG speedup** on `v5p` while isolating TPU host CPUs from Pillow GIL contention.

### Slice `v5p-16` (`v5p`, `TP/EP=4 × DP=4`, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16/summary.json`](./summary.json) *(updated `2026-11-02T19:56:33.784262+00:00`)*

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
