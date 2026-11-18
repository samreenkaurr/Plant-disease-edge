# Results (fill in after running scripts/run_experiments.sh)

Source of every number: `runs/<experiment>/eval_*/metrics.json`. Report mean +/- std over 3 seeds.

## 1. Lab-to-field gap
| Experiment | Train data | Aug | Acc (pv_val) | Acc (pd_test) | Macro-F1 (pd_test) |
|---|---|---|---|---|---|
| E1 | PV | basic | | | |
| E2 | PV | strong | | | |
| E3 | PV + PD | strong | | | |

## 2. Calibration (pd_test)
| Model | ECE raw | ECE after temperature scaling | Temperature |
|---|---|---|---|
| E3 | | | |

Selective prediction (E3, calibrated): coverage and accuracy at confidence threshold 0.5 / 0.6 / 0.7 / 0.8 / 0.9.

## 3. Compression
| Variant | Size (MB) | Acc (pd_test) | Latency low-end phone (ms) | Latency mid-range phone (ms) |
|---|---|---|---|---|
| fp32 | | | | |
| fp16 | | | | |
| dynamic | | | | |
| int8 | | | | |

Devices: (model, chip, RAM, OS version). Latency: median of 50 runs after 5 warm-up runs.

## 4. Explainability
Add 3-4 Grad-CAM overlays from `src/gradcam.py`, including at least one failure case.

## 5. User test
n = , tasks, success rate, comments.
