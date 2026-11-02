# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B-A13B MoE) Serving on Cloud TPU `v5p` & `v7x` — Slice `v7x-8`

### Slice `v7x-8` (`v7x`, `TP/EP=4 × DP=2`, `5` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-8/SUMMARY.md`](./SUMMARY.md) · [`results/v7x-8/summary.json`](./summary.json) *(updated `2026-11-02T20:11:00.222155+00:00`)*

| Checkpoint Variant | MoE Kernel | Task & Resolution | Seq Tokens | Steps | Stage-1 CoT (s) | Stage-2 Step (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Local (s) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Direct-1024x1024` | `4,352` | `8` | `0.046` | `68.09` | `119.9` | **`146.6`** | `0.710` | **`0.737`** | **`162.82`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-ThinkRecaption-CoT-1024x1024` | `4,992` | `8` | `14.039` | `63.52` | `115.7` | **`144.3`** | `14.663` | **`14.692`** | **`8.17`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-Wide-1280x768` | `4,608` | `8` | `9.243` | `63.79` | `106.9` | **`139.2`** | `9.860` | **`9.893`** | **`12.13`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `8,832` | `8` | `13.057` | `64.13` | `118.0` | **`193.9`** | `13.708` | **`13.790`** | **`8.70`** |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `T2I-UltraHD-2048x2048-LongSeq20K` | `20,864` | `8` | `13.309` | `194.85` | `465.6` | **`594.2`** | `15.334` | **`15.462`** | **`7.76`** |
