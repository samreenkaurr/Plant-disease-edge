#!/usr/bin/env bash
# Full experiment suite. Run from the repo root after scripts/../src/prepare_data.py.
set -euo pipefail
D=data/unified; R=runs
ev() { python src/evaluate.py --model "$1" --data "$2" --out "$3" ${4:+--temp-fit-dir "$4"}; }

# E1: lab-only baseline (exposes the lab->field gap)
python src/train.py --train $D/pv_train --val $D/pv_val --aug basic  --out $R/e1_baseline
# E2: lab-only + strong augmentation
python src/train.py --train $D/pv_train --val $D/pv_val --aug strong --out $R/e2_strong_aug
# E3: lab + field mix + strong augmentation (final model)
python src/train.py --train $D/pv_train $D/pd_train --val $D/pd_val --aug strong --out $R/e3_mixed

for e in e1_baseline e2_strong_aug e3_mixed; do
  ev $R/$e/model.keras $D/pv_val  $R/$e/eval_pv_val
  ev $R/$e/model.keras $D/pd_test $R/$e/eval_pd_test $D/pd_val
done

# Compression sweep on the final model, then evaluate every variant on the field test set
python src/export_tflite.py --model $R/e3_mixed/model.keras --calib-dir $D/pd_train --out $R/e3_mixed/tflite
for v in fp32 fp16 dynamic int8; do
  [ -f $R/e3_mixed/tflite/model_$v.tflite ] && ev $R/e3_mixed/tflite/model_$v.tflite $D/pd_test $R/e3_mixed/eval_tflite_$v
done

python src/make_app_assets.py --model $R/e3_mixed/tflite/model_fp16.tflite --calibration $R/e3_mixed/eval_pd_test/calibration.json
echo "Done. Metrics are in runs/*/eval_*/metrics.json"
