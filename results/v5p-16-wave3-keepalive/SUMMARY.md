# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B-A13B MoE) Serving on Cloud TPU `v5p` & `v7x` — Slice `v5p-16-wave3-keepalive`

### Slice `v5p-16-wave3-keepalive` (`v5p`, `TP/EP=4 × DP=4`, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave3-keepalive/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16-wave3-keepalive/summary.json`](./summary.json) *(updated `2026-11-02T20:24:25.776721+00:00`)*

| Checkpoint Variant | MoE Kernel | Task & Resolution | Seq Tokens | Steps | Stage-1 CoT (s) | Stage-2 Step (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Local (s) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `8` | `0.064` | `148.87` | `199.7` | **`197.2`** | `1.455` | **`1.452`** | **`165.29`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `8` | `19.299` | `145.66` | `156.7` | **`143.5`** | `20.621` | **`20.608`** | **`11.65`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `8` | `12.528` | `141.88` | `147.4` | **`135.8`** | `13.810` | **`13.799`** | **`17.39`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `8` | `19.123` | `145.55` | `156.3` | **`145.2`** | `20.473` | **`20.458`** | **`11.73`** |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `8` | `19.132` | `493.70` | `617.6` | **`570.3`** | `23.699` | **`23.652`** | **`10.15`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `8` | `0.083` | `207.16` | `164.6` | **`145.7`** | `1.905` | **`1.886`** | **`127.25`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `8` | `27.273` | `203.83` | `156.1` | **`142.5`** | `29.060` | **`29.046`** | **`8.26`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `8` | `17.590` | `194.66` | `146.6` | **`134.1`** | `19.294` | **`19.282`** | **`12.45`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `8` | `27.279` | `204.08` | `156.6` | **`143.2`** | `29.097` | **`29.080`** | **`8.25`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `8` | `27.175` | `701.72` | `616.4` | **`570.8`** | `33.405` | **`33.360`** | **`7.19`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `50` | `0.064` | `270.35` | `163.3` | **`197.1`** | `13.745` | **`13.779`** | **`17.42`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `50` | `19.385` | `268.91` | `156.9` | **`151.9`** | `32.987` | **`32.982`** | **`7.28`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `50` | `12.635` | `250.17` | `147.2` | **`144.2`** | `25.291` | **`25.288`** | **`9.49`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `50` | `18.901` | `267.38` | `157.2` | **`146.0`** | `32.456` | **`32.447`** | **`7.40`** |
| `HunyuanImage-3-Instruct` | `COMPILED_VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `50` | `19.060` | `1,042.15` | `619.7` | **`572.9`** | `71.787` | **`71.740`** | **`3.35`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `50` | `0.077` | `363.25` | `163.4` | **`150.3`** | `18.403` | **`18.390`** | **`13.05`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `50` | `23.420` | `359.60` | `156.1` | **`143.4`** | `41.556` | **`41.543`** | **`5.78`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `50` | `14.682` | `340.09` | `146.6` | **`135.0`** | `31.833` | **`31.822`** | **`7.54`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `50` | `22.102` | `359.30` | `156.1` | **`143.4`** | `40.251` | **`40.241`** | **`5.96`** |
| `HunyuanImage-3-Instruct` | `VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `50` | `22.123` | `1,302.27` | `621.5` | **`572.6`** | `87.858` | **`87.809`** | **`2.73`** |
