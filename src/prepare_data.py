"""Build unified, split folders from PlantVillage and PlantDoc.

Output (under --out):
  pv_train/ pv_val/        <- PlantVillage (lab images), 90/10 split
  pd_train/ pd_val/        <- PlantDoc TRAIN (field images), 90/10 split
  pd_test/                 <- PlantDoc TEST (held-out field images: the real benchmark)
Each contains one sub-folder per label in common.LABELS.
"""
import argparse, os, random, shutil
from pathlib import Path
from common import LABELS, EXT, map_folder_to_label


def collect(root):
    items = []
    root = Path(root)
    for d in [root, *root.rglob("*")]:
        if not d.is_dir():
            continue
        lab = map_folder_to_label(d.name)
        if lab:
            items += [(p, lab) for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in EXT]
    return items


def write(items, out):
    out = Path(out)
    for lab in LABELS:
        (out / lab).mkdir(parents=True, exist_ok=True)
    for i, (p, lab) in enumerate(items):
        dst = out / lab / f"{i:06d}_{p.name}"
        try:
            os.symlink(p.resolve(), dst)
        except OSError:
            shutil.copy2(p, dst)


def split(items, frac, seed):
    rng = random.Random(seed)
    by = {}
    for it in items:
        by.setdefault(it[1], []).append(it)
    a, b = [], []
    for lab, xs in by.items():
        rng.shuffle(xs)
        k = int(len(xs) * frac)
        a += xs[:k]; b += xs[k:]
    return a, b


def report(name, items):
    c = {}
    for _, l in items:
        c[l] = c.get(l, 0) + 1
    print(f"{name:9s} total={len(items):6d}  " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantvillage", required=True)
    ap.add_argument("--plantdoc-train", required=True)
    ap.add_argument("--plantdoc-test", required=True)
    ap.add_argument("--out", default="data/unified")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    pv_tr, pv_va = split(collect(a.plantvillage), 0.9, a.seed)
    pd_tr, pd_va = split(collect(a.plantdoc_train), 0.9, a.seed)
    pd_te = collect(a.plantdoc_test)
    for name, items in [("pv_train", pv_tr), ("pv_val", pv_va), ("pd_train", pd_tr), ("pd_val", pd_va), ("pd_test", pd_te)]:
        write(items, Path(a.out) / name)
        report(name, items)
    print("NOTE: PlantVillage has near-duplicate shots of the same leaf, so pv_val is optimistic. Use pd_test as the real benchmark.")
