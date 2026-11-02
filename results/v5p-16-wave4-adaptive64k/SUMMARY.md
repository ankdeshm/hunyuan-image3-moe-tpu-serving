# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B-A13B MoE) Serving on Cloud TPU `v5p` & `v7x` — Slice `v5p-16-wave4-adaptive64k`

### Slice `v5p-16-wave4-adaptive64k` (`v5p`, `TP/EP=4 × DP=4`, `10` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave4-adaptive64k/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16-wave4-adaptive64k/summary.json`](./summary.json) *(updated `2026-11-02T21:02:03.007267+00:00`)*

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
