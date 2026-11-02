# `tencent/HunyuanImage-3.0-Instruct` & `-Distil` (83B Total / 13B Active 64-Expert MoE) — `v5p-32` (`v5p`)

- **Updated (UTC):** `2026-11-02T19:12:22.017812+00:00`
- **Hardware:** `8` hosts x `4` TPU cores/host = `32` total TPU cores
- **Disaggregated CPU Offload Health:** `ok`

Compares `COMPILED_VECTORIZED_GROUPED_MM` (`torch.compile(..., dynamic=False)`), `VECTORIZED_GROUPED_MM`, and `EAGER_MOE` across BF16 and hardware FP8 MXU (`torch._scaled_mm`), alongside PNG / multi-reference vision CPU offload ablation (`multimodal-cpu-offload-svc`).

| Checkpoint | MoE Kernel | Precision | TP/EP | Cluster DP | Task | Res | Steps | Stage-1 CoT (s) | Diff Step (ms) | TPU E2E (s) | Local PNG (ms) | Disagg RTT (ms) | E2E Local (s) | E2E Disagg (s) | Cluster img/min (Disagg) | Status |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `4` | `8` | `T2I-Direct-1024x1024` | `1024x1024` | `8` | 0.066 | 148.65 | 1.255 | 195.76 | 151.2 | 1.451 | 1.406 | 341.39 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `4` | `8` | `T2I-ThinkRecaption-CoT-1024x1024` | `1024x1024` | `8` | 19.602 | 145.29 | 20.765 | 154.39 | 150.96 | 20.919 | 20.916 | 22.95 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `4` | `8` | `T2I-Wide-1280x768` | `1280x768` | `8` | 12.748 | 142.67 | 13.889 | 146.74 | 163.64 | 14.036 | 14.053 | 34.16 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `4` | `8` | `TI2I-MultiImageEdit-2Refs-1024x1024` | `1024x1024` | `8` | 19.182 | 144.75 | 20.34 | 156.2 | 186.58 | 20.524 | 20.527 | 23.38 | `PASS` |
| `HunyuanImage-3-Instruct-Distil` | `COMPILED_VECTORIZED_GROUPED_MM` | `FP8_DYNAMIC` | `4` | `8` | `T2I-UltraHD-2048x2048-LongSeq20K` | `2048x2048` | `8` | 18.962 | 494.24 | 22.916 | 609.55 | 598.15 | 23.525 | 23.514 | 20.41 | `PASS` |
