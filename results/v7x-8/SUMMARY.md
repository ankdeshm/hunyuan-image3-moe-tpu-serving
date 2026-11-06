# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B Total / 13B Active 64-Expert MoE) — `v7x-8` (`v7x`)

- **Updated (UTC):** `2026-11-06T13:00:45.205460+00:00`
- **Hardware:** `2` hosts x `8` TPU cores/host = `16` total TPU cores
- **Disaggregated CPU Offload Health:** `ok`

Compares `COMPILED_VECTORIZED_GROUPED_MM` (`torch.compile(..., dynamic=False)`), `VECTORIZED_GROUPED_MM`, and `EAGER_MOE` across BF16 and hardware FP8 MXU (`torch._scaled_mm`), alongside PNG / multi-reference vision CPU offload ablation (`multimodal-cpu-offload-svc`).

| Checkpoint | MoE Kernel | Precision | TP/EP | Cluster DP | Task | Res | Steps | Stage-1 CoT (s) | Diff Step (ms) | TPU E2E (s) | Local PNG (ms) | Disagg RTT (ms) | E2E Local (s) | E2E Disagg (s) | Cluster img/min (Disagg) | Status |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `2` | `T2I-Direct-1024x1024` | `1024x1024` | `8` | 0.048 | 69.13 | 0.601 | 118.93 | 146.23 | 0.72 | 0.747 | 160.64 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `2` | `T2I-ThinkRecaption-CoT-1024x1024` | `1024x1024` | `8` | 16.269 | 64.56 | 16.786 | 113.7 | 142.84 | 16.899 | 16.929 | 7.09 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `2` | `T2I-Wide-1280x768` | `1280x768` | `8` | 11.312 | 63.11 | 11.817 | 110.23 | 134.01 | 11.927 | 11.951 | 10.04 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `2` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `1024x1024` | `8` | 14.658 | 64.41 | 15.173 | 131.41 | 167.13 | 15.329 | 15.34 | 7.82 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `8` | `2` | `T2I-UltraHD-2048x2048-LongSeq20K` | `2048x2048` | `8` | 14.594 | 196.7 | 16.168 | 448.47 | 570.64 | 16.617 | 16.739 | 7.17 | `PASS` |
