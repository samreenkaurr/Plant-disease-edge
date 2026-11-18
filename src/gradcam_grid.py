"""Grad-CAM grid: compare what several models look at on the same test images.
Picks N images the LAST model gets wrong and N it gets right.
python src/gradcam_grid.py --models e1=runs/e1_baseline/model.keras e3=runs/e3_mixed/model.keras \
    --data data/unified/pd_test --n 4 --out figures/gradcam_e1_vs_e3.png
"""
import argparse, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from common import IMG_SIZE, LABELS, EXT


def make_cam_fn(model):
    last = next(l for l in reversed(model.layers) if len(l.output.shape) == 4)
    gm = tf.keras.Model(model.inputs, [last.output, model.output])

    def cam(x):
        with tf.GradientTape() as tape:
            conv, preds = gm(tf.constant(x[None]))
            c = int(tf.argmax(preds[0]))
            score = preds[0, c]
        g = tape.gradient(score, conv)
        w = tf.reduce_mean(g, axis=(1, 2))
        m = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0].numpy()
        m = m / (m.max() + 1e-8)
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0
        return m, c, float(preds[0, c])
    return cam


def short(label):
    return label.replace("_", " ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path/to/model.keras (last one picks the images)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/gradcam.png")
    a = ap.parse_args()

    items = []
    for i, lab in enumerate(LABELS):
        d = Path(a.data) / lab
        if d.exists():
            items += [(p, i) for p in sorted(d.iterdir()) if p.suffix.lower() in EXT]
    imgs = [np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) for p, _ in items]
    y = np.array([t for _, t in items])

    models = []
    for spec in a.models:
        name, path = spec.split("=", 1)
        models.append((name, make_cam_fn(tf.keras.models.load_model(path))))

    pred_last = np.array([models[-1][1](x)[1] for x in imgs])
    wrong = [i for i in range(len(y)) if pred_last[i] != y[i]]
    right = [i for i in range(len(y)) if pred_last[i] == y[i]]
    rng = random.Random(a.seed)
    rng.shuffle(wrong); rng.shuffle(right)
    sel = wrong[: a.n] + right[: a.n]

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, cols = len(sel), 1 + len(models)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    for r, i in enumerate(sel):
        axes[r][0].imshow(imgs[i].astype(np.uint8)); axes[r][0].axis("off")
        axes[r][0].set_title(f"true: {short(LABELS[y[i]])}", fontsize=8)
        for c, (name, fn) in enumerate(models, start=1):
            m, p, conf = fn(imgs[i])
            ax = axes[r][c]
            ax.imshow(imgs[i].astype(np.uint8)); ax.imshow(m, cmap="jet", alpha=0.4); ax.axis("off")
            ax.set_title(f"{name}: {short(LABELS[p])} ({conf:.2f})", fontsize=8, color="green" if p == y[i] else "red")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"saved {a.out}  ({len(wrong)} wrong / {len(right)} right for '{models[-1][0]}')")
