"""Export fp32 / fp16 / dynamic-range / full-int8 TFLite variants (float I/O) and report sizes.
python src/export_tflite.py --model runs/e3/model.keras --calib-dir data/unified/pd_train --out runs/e3/tflite
"""
import argparse, json
from pathlib import Path
import tensorflow as tf
from common import load_dir


def convert(model, variant, rep=None):
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    if variant == "fp16":
        c.optimizations = [tf.lite.Optimize.DEFAULT]; c.target_spec.supported_types = [tf.float16]
    elif variant == "dynamic":
        c.optimizations = [tf.lite.Optimize.DEFAULT]
    elif variant == "int8":
        c.optimizations = [tf.lite.Optimize.DEFAULT]; c.representative_dataset = rep
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return c.convert()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n-calib", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(a.model)
    ds = load_dir(a.calib_dir, batch_size=None, shuffle=True, seed=1).take(a.n_calib)

    def rep():
        for x, _ in ds:
            yield [tf.cast(x, tf.float32)[None]]

    sizes = {}
    for v in ["fp32", "fp16", "dynamic", "int8"]:
        try:
            blob = convert(model, v, rep)
            (out / f"model_{v}.tflite").write_bytes(blob)
            sizes[v] = round(len(blob) / 1e6, 3)
            print(f"{v:8s} {sizes[v]} MB")
        except Exception as e:  # keep going if one variant is unsupported by your TF version
            print(f"{v} failed: {e}")
    (out / "sizes_mb.json").write_text(json.dumps(sizes, indent=2))
