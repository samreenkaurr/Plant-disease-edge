"""Export fp32 / fp16 / dynamic / int8 TFLite variants (float I/O) and report sizes.
Each variant converts in its own subprocess, because the TFLite converter can hard-crash
(LLVM abort) and would otherwise kill the whole run.
python src/export_tflite.py --model runs/e3/model.keras --calib-dir data/unified/pd_train --out runs/e3/tflite
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path


def worker(saved_dir, variant, calib_dir, n_calib, out_file):
    import tensorflow as tf
    from common import load_dir
    c = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    if variant == "fp16":
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]
    elif variant == "dynamic":
        c.optimizations = [tf.lite.Optimize.DEFAULT]
    elif variant == "int8":
        ds = load_dir(calib_dir, batch_size=None, shuffle=True, seed=1).take(int(n_calib))

        def rep():
            for x, _ in ds:
                yield [tf.cast(x, tf.float32)[None]]

        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = rep
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    Path(out_file).write_bytes(c.convert())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:7])
        sys.exit(0)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calib", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    model = tf.keras.models.load_model(a.model)
    saved = tempfile.mkdtemp(prefix="saved_model_")
    model.export(saved)

    sizes = {}
    for v in ["fp32", "fp16", "dynamic", "int8"]:
        dst = out / f"model_{v}.tflite"
        r = subprocess.run([sys.executable, __file__, "--worker", saved, v, a.calib_dir, str(a.n_calib), str(dst)])
        if r.returncode == 0 and dst.exists():
            sizes[v] = round(dst.stat().st_size / 1e6, 3)
            print(f"{v:8s} {sizes[v]} MB")
        else:
            print(f"{v:8s} FAILED (converter exit code {r.returncode})")
    (out / "sizes_mb.json").write_text(json.dumps(sizes, indent=2))
