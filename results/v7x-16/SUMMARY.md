# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B Total / 13B Active 64-Expert MoE) — `v7x-16` (`v7x`)

- **Updated (UTC):** `2026-11-03T09:42:48.906757+00:00`
- **Hardware:** `4` hosts x `8` TPU cores/host = `32` total TPU cores
- **Disaggregated CPU Offload Health:** `ok`

Compares `COMPILED_VECTORIZED_GROUPED_MM` (`torch.compile(..., dynamic=False)`), `VECTORIZED_GROUPED_MM`, and `EAGER_MOE` across BF16 and hardware FP8 MXU (`torch._scaled_mm`), alongside PNG / multi-reference vision CPU offload ablation (`multimodal-cpu-offload-svc`).

| Checkpoint | MoE Kernel | Precision | TP/EP | Cluster DP | Task | Res | Steps | Stage-1 CoT (s) | Diff Step (ms) | TPU E2E (s) | Local PNG (ms) | Disagg RTT (ms) | E2E Local (s) | E2E Disagg (s) | Cluster img/min (Disagg) | Status |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `4` | `T2I-Direct-1024x1024` | `1024x1024` | `8` | 0.045 | 70.02 | 0.605 | 118.82 | 146.0 | 0.724 | 0.751 | 319.57 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `4` | `T2I-ThinkRecaption-CoT-1024x1024` | `1024x1024` | `8` | 16.674 | 66.33 | 17.205 | 114.53 | 142.76 | 17.319 | 17.347 | 13.84 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `4` | `T2I-Wide-1280x768` | `1280x768` | `8` | 8.956 | 64.79 | 9.474 | 107.51 | 133.78 | 9.582 | 9.608 | 24.98 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `4` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `1024x1024` | `8` | 13.569 | 62.47 | 14.069 | 118.17 | 167.18 | 14.208 | 14.236 | 16.86 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `4` | `T2I-UltraHD-2048x2048-LongSeq20K` | `2048x2048` | `8` | 13.336 | 195.58 | 14.901 | 465.25 | 568.39 | 15.366 | 15.469 | 15.51 | `PASS` |
