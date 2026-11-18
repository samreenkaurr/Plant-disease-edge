"""Copy a TFLite model, labels, and calibration into the Flutter app.
python src/make_app_assets.py --model runs/e3/tflite/model_int8.tflite --calibration runs/e3/eval_pd_test/calibration.json
"""
import argparse, shutil
from pathlib import Path
from common import LABELS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--calibration")
    ap.add_argument("--app", default="app/assets")
    a = ap.parse_args()
    d = Path(a.app); d.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.model, d / "model.tflite")
    (d / "labels.txt").write_text("\n".join(LABELS) + "\n")
    if a.calibration:
        shutil.copy(a.calibration, d / "calibration.json")
    print("app assets updated in", d)
