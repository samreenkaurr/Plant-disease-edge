"""Evaluate a .keras or .tflite model on a folder: accuracy, macro-F1, confusion matrix,
calibration (ECE/NLL), selective prediction, optional temperature scaling.
python src/evaluate.py --model runs/e3/model.keras --data data/unified/pd_test --out runs/e3/eval_pd_test \
       --temp-fit-dir data/unified/pd_val
"""
import argparse, json
from pathlib import Path
import numpy as np
from common import LABELS, load_dir


def predict_probs(model_path, data_dir):
    import tensorflow as tf
    ds = load_dir(data_dir, batch_size=32, shuffle=False)
    y = np.concatenate([lab.numpy() for _, lab in ds])
    if str(model_path).endswith(".tflite"):
        it = tf.lite.Interpreter(model_path=str(model_path)); it.allocate_tensors()
        i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
        probs = []
        for xb, _ in ds:
            for x in xb.numpy():
                it.set_tensor(i_in, x[None].astype(np.float32)); it.invoke()
                probs.append(it.get_tensor(i_out)[0])
        return np.array(probs), y
    model = tf.keras.models.load_model(model_path)
    return model.predict(ds, verbose=0), y


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def apply_temp(probs, T):
    return softmax(np.log(np.clip(probs, 1e-8, 1)) / T)


def fit_temperature(probs, y):
    from scipy.optimize import minimize_scalar
    nll = lambda T: -np.mean(np.log(np.clip(apply_temp(probs, T)[np.arange(len(y)), y], 1e-8, 1)))
    return float(minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x)


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def metrics(probs, y):
    from sklearn.metrics import f1_score, confusion_matrix
    pred = probs.argmax(1)
    present = sorted(set(y.tolist()))
    sel = {}
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = probs.max(1) >= t
        sel[str(t)] = {"coverage": float(m.mean()), "accuracy": float((pred[m] == y[m]).mean()) if m.any() else None}
    return {
        "n": int(len(y)), "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "per_class_f1": {LABELS[i]: float(v) for i, v in zip(present, f1_score(y, pred, labels=present, average=None, zero_division=0))},
        "ece": ece(probs, y),
        "nll": float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-8, 1)))),
        "selective": sel,
        "confusion": confusion_matrix(y, pred, labels=range(len(LABELS))).tolist(),
    }


def plot_cm(cm, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=60, ha="right"); ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout(); fig.savefig(path, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--temp-fit-dir")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    probs, y = predict_probs(a.model, a.data)
    res = {"raw": metrics(probs, y)}
    if a.temp_fit_dir:
        fp, fy = predict_probs(a.model, a.temp_fit_dir)
        T = fit_temperature(fp, fy)
        res["temperature"] = T
        res["calibrated"] = metrics(apply_temp(probs, T), y)
        (out / "calibration.json").write_text(json.dumps({"temperature": T, "threshold": 0.6}))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    plot_cm(res["raw"]["confusion"], out / "confusion.png")
    r = res["raw"]
    print(f"acc={r['accuracy']:.3f} macroF1={r['macro_f1']:.3f} ECE={r['ece']:.3f}" +
          (f" | calibrated ECE={res['calibrated']['ece']:.3f} (T={res['temperature']:.2f})" if "calibrated" in res else ""))
