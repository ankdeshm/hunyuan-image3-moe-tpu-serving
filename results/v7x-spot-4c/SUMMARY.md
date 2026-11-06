# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B Total / 13B Active 64-Expert MoE) — `v7x-spot-4c` (`v7x`)

- **Updated (UTC):** `2026-11-06T12:54:14.436490+00:00`
- **Hardware:** `1` hosts x `8` TPU cores/host = `8` total TPU cores
- **Disaggregated CPU Offload Health:** `ok`

Compares `COMPILED_VECTORIZED_GROUPED_MM` (`torch.compile(..., dynamic=False)`), `VECTORIZED_GROUPED_MM`, and `EAGER_MOE` across BF16 and hardware FP8 MXU (`torch._scaled_mm`), alongside PNG / multi-reference vision CPU offload ablation (`multimodal-cpu-offload-svc`).

| Checkpoint | MoE Kernel | Precision | TP/EP | Cluster DP | Task | Res | Steps | Stage-1 CoT (s) | Diff Step (ms) | TPU E2E (s) | Local PNG (ms) | Disagg RTT (ms) | E2E Local (s) | E2E Disagg (s) | Cluster img/min (Disagg) | Status |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `1` | `T2I-Direct-1024x1024` | `1024x1024` | `8` | 0.047 | 68.49 | 0.595 | 119.4 | 147.08 | 0.715 | 0.742 | 80.86 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `1` | `T2I-ThinkRecaption-CoT-1024x1024` | `1024x1024` | `8` | 15.129 | 62.05 | 15.626 | 114.16 | 143.02 | 15.74 | 15.769 | 3.8 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `1` | `T2I-Wide-1280x768` | `1280x768` | `8` | 9.505 | 64.72 | 10.023 | 108.16 | 134.19 | 10.131 | 10.157 | 5.91 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `1` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `1024x1024` | `8` | 14.326 | 65.58 | 14.851 | 113.53 | 167.71 | 14.987 | 15.019 | 3.99 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `1` | `T2I-UltraHD-2048x2048-LongSeq20K` | `2048x2048` | `8` | 14.449 | 194.96 | 16.008 | 456.26 | 577.58 | 16.465 | 16.586 | 3.62 | `PASS` |
