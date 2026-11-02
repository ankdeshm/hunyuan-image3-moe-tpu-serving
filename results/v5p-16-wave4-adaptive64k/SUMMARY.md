# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B-A13B MoE) Serving on Cloud TPU `v5p` & `v7x` — Slice `v5p-16-wave4-adaptive64k`

### Slice `v5p-16-wave4-adaptive64k` (`v5p`, `TP/EP=4 × DP=4`, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave4-adaptive64k/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16-wave4-adaptive64k/summary.json`](./summary.json) *(updated `2026-11-02T21:11:12.964757+00:00`)*

| Checkpoint Variant | MoE Kernel | Task & Resolution | Seq Tokens | Steps | Stage-1 CoT (s) | Stage-2 Step (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Local (s) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `8` | `0.065` | `148.53` | `199.6` | **`150.3`** | `1.452` | **`1.403`** | **`171.06`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `8` | `19.367` | `144.72` | `156.4` | **`143.8`** | `20.681` | **`20.669`** | **`11.61`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `8` | `12.804` | `141.28` | `147.2` | **`134.4`** | `14.082` | **`14.069`** | **`17.06`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `8` | `18.974` | `145.48` | `156.9` | **`144.5`** | `20.324` | **`20.308`** | **`11.82`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `8` | `19.147` | `494.02` | `618.5` | **`569.4`** | `23.718` | **`23.669`** | **`10.14`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `8` | `0.082` | `207.08` | `162.5` | **`151.6`** | `1.901` | **`1.890`** | **`126.98`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `8` | `28.993` | `204.84` | `156.8` | **`143.3`** | `30.789` | **`30.775`** | **`7.80`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `8` | `19.057` | `195.73` | `146.9` | **`134.6`** | `20.770` | **`20.757`** | **`11.56`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `8` | `25.609` | `204.00` | `156.6` | **`143.2`** | `27.427` | **`27.414`** | **`8.75`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `8` | `22.903` | `701.32` | `619.4` | **`573.4`** | `29.133` | **`29.087`** | **`8.25`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `50` | `0.064` | `271.30` | `163.3` | **`149.3`** | `13.792` | **`13.778`** | **`17.42`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `50` | `19.387` | `268.33` | `156.2` | **`143.2`** | `32.959` | **`32.946`** | **`7.28`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `50` | `12.677` | `252.11` | `146.8` | **`135.0`** | `25.430` | **`25.418`** | **`9.44`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `50` | `18.970` | `268.69` | `156.5` | **`143.8`** | `32.591` | **`32.573`** | **`7.37`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `50` | `19.153` | `1,045.91` | `617.3` | **`571.0`** | `72.065` | **`72.019`** | **`3.33`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `50` | `0.074` | `363.08` | `163.5` | **`149.8`** | `18.392` | **`18.378`** | **`13.06`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `50` | `24.201` | `359.60` | `156.1` | **`143.4`** | `42.337` | **`42.325`** | **`5.67`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `50` | `15.140` | `340.48` | `146.7` | **`134.8`** | `32.310` | **`32.298`** | **`7.43`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `50` | `22.515` | `359.41` | `156.8` | **`143.4`** | `40.671` | **`40.659`** | **`5.90`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `50` | `22.525` | `1,301.91` | `616.2` | **`571.5`** | `88.237` | **`88.192`** | **`2.72`** |
