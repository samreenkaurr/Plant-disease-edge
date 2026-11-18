"""Shared label set, folder->label mapping, and tf.data pipelines."""
import re
from pathlib import Path

IMG_SIZE = 224
# Classes present in BOTH PlantVillage and PlantDoc (tomato + potato).
LABELS = [
    "potato_early_blight", "potato_healthy", "potato_late_blight",
    "tomato_early_blight", "tomato_healthy", "tomato_late_blight",
    "tomato_septoria_leaf_spot",
]
EXT = {".jpg", ".jpeg", ".png"}


def map_folder_to_label(name):
    """Map a dataset folder name (PlantVillage or PlantDoc style) to a unified label, or None."""
    n = re.sub(r"[^a-z]+", " ", name.lower()).strip()
    crop = "tomato" if "tomato" in n else "potato" if "potato" in n else None
    if crop is None:
        return None
    if "early blight" in n:
        d = "early_blight"
    elif "late blight" in n:
        d = "late_blight"
    elif "septoria" in n:
        d = "septoria_leaf_spot"
    elif "healthy" in n or n in ("tomato leaf", "potato leaf"):
        d = "healthy"
    else:
        return None
    label = f"{crop}_{d}"
    return label if label in LABELS else None


def count_images(dirs):
    counts = [0] * len(LABELS)
    for d in dirs:
        for i, lab in enumerate(LABELS):
            p = Path(d) / lab
            if p.exists():
                counts[i] += sum(1 for f in p.iterdir() if f.suffix.lower() in EXT)
    return counts


def load_dir(path, batch_size=None, shuffle=False, seed=0):
    import tensorflow as tf
    return tf.keras.utils.image_dataset_from_directory(
        path, labels="inferred", label_mode="int", class_names=LABELS,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size, shuffle=shuffle, seed=seed)


def basic_aug(img, y):
    import tensorflow as tf
    return tf.image.random_flip_left_right(img), y


def strong_aug(img, y):
    """Field-style robustness: zoom/crop, colour + lighting jitter, cutout (fights clean-background shortcuts)."""
    import tensorflow as tf
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    s = tf.cast(tf.random.uniform([], 0.6, 1.0) * IMG_SIZE, tf.int32)
    img = tf.image.random_crop(img, tf.stack([s, s, 3]))
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.6, 1.4)
    img = tf.image.random_saturation(img, 0.6, 1.4)
    img = tf.image.random_hue(img, 0.04)
    cs = IMG_SIZE // 4
    cy = tf.random.uniform([], 0, IMG_SIZE, tf.int32)
    cx = tf.random.uniform([], 0, IMG_SIZE, tf.int32)
    ys, xs = tf.range(IMG_SIZE)[:, None], tf.range(IMG_SIZE)[None, :]
    mask = (tf.abs(ys - cy) < cs) & (tf.abs(xs - cx) < cs)
    mask = tf.cast(mask[..., None], tf.float32) * tf.cast(tf.random.uniform([]) < 0.5, tf.float32)
    img = img * (1 - mask) + tf.random.uniform([IMG_SIZE, IMG_SIZE, 3]) * mask
    return tf.clip_by_value(img, 0.0, 1.0) * 255.0, y


AUGS = {"basic": basic_aug, "strong": strong_aug}
