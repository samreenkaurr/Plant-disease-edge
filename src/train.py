"""Fine-tune MobileNetV3-Small. Example:
python src/train.py --train data/unified/pv_train --val data/unified/pv_val --aug basic --out runs/e1_baseline
python src/train.py --train data/unified/pv_train data/unified/pd_train --val data/unified/pd_val --aug strong --out runs/e3_mixed
"""
import argparse, json
from pathlib import Path
import tensorflow as tf
from common import LABELS, IMG_SIZE, AUGS, load_dir, count_images


def build_model(n_classes, pretrained=True, dropout=0.3):
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights="imagenet" if pretrained else None, include_preprocessing=True)  # expects 0-255 input
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(base.input, out), base


def make_train_ds(dirs, aug, bs, seed):
    ds = None
    for d in dirs:
        part = load_dir(d, batch_size=None, shuffle=True, seed=seed)
        ds = part if ds is None else ds.concatenate(part)
    ds = ds.shuffle(4096, seed=seed).map(AUGS[aug], num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)


def make_val_ds(dirs, bs):
    ds = None
    for d in dirs:
        part = load_dir(d, batch_size=None)
        ds = part if ds is None else ds.concatenate(part)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--aug", choices=list(AUGS), default="basic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-ft", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="smoke tests only")
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = make_train_ds(a.train, a.aug, a.batch, a.seed), make_val_ds(a.val, a.batch)
    counts = count_images(a.train)
    total = sum(counts)
    cw = {i: total / (len(LABELS) * max(c, 1)) for i, c in enumerate(counts)}

    model, base = build_model(len(LABELS), pretrained=not a.no_pretrained)
    metrics = ["accuracy"]
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)

    base.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=metrics)
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs_head, class_weight=cw)

    base.trainable = True
    for l in base.layers[: int(len(base.layers) * 0.6)]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-4), "sparse_categorical_crossentropy", metrics=metrics)
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs_ft, class_weight=cw, callbacks=[es])

    model.save(out / "model.keras")
    (out / "labels.json").write_text(json.dumps(LABELS))
    (out / "train_args.json").write_text(json.dumps(vars(a), indent=2))
    print("saved", out / "model.keras")
