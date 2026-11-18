"""Grad-CAM overlay to check the model looks at lesions, not backgrounds.
python src/gradcam.py --model runs/e3/model.keras --image leaf.jpg --out cam.png
"""
import argparse
import numpy as np, tensorflow as tf
from PIL import Image
from common import IMG_SIZE, LABELS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--image", required=True); ap.add_argument("--out", default="cam.png")
    a = ap.parse_args()
    model = tf.keras.models.load_model(a.model)
    last = next(l for l in reversed(model.layers) if len(l.output.shape) == 4)
    gm = tf.keras.Model(model.inputs, [last.output, model.output])
    im = Image.open(a.image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = tf.constant(np.asarray(im, np.float32)[None])
    with tf.GradientTape() as tape:
        conv, preds = gm(x)
        cls = int(tf.argmax(preds[0]))
        score = preds[0, cls]
    g = tape.gradient(score, conv)
    w = tf.reduce_mean(g, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(im); ax[0].axis("off")
    ax[1].imshow(im); ax[1].imshow(cam, cmap="jet", alpha=0.4); ax[1].axis("off")
    ax[1].set_title(f"{LABELS[cls]} ({float(score):.2f})", fontsize=9)
    fig.tight_layout(); fig.savefig(a.out, dpi=150); print("saved", a.out)
