"""
fer_lib.py - face-expression training library used by face_expression_kfold.ipynb

Everything that runs inside DataLoader worker processes lives in this module (not in the notebook),
because on Windows the workers are *spawned* and cannot import classes defined in a notebook.

Contents
--------
1. Dataset indexing   : originals vs "__aug" copies, md5 de-duplication, near-duplicate merging,
                        label fixes, leakage-free test split + stratified K folds by group
2. Face cropping      : YuNet crops of every image (same FaceCropper as the deployment module)
3. Models             : HSEmotion (AffectNet-pretrained, version-independent loader) and timm backbones
4. Training           : AdamW (backbone / head LRs), warm-up + cosine, label smoothing, mixup,
                        EMA, bf16 AMP, early stopping, head-only warm-up
5. Inference          : batched prediction with optional flip TTA
6. Export             : ensemble -> single ONNX (normalisation + softmax + TTA inside), verified
7. Analysis helpers   : bootstrap confidence interval, mislabel candidates, contact sheets
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import re
import time
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fer_infer_onnx import FaceCropper, center_square, read_bgr  # same crop logic as deployment

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUG_PATTERN = re.compile(r"__aug\d+$")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ======================================================================================
# 1. Dataset indexing
# ======================================================================================
def list_images(folder) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def original_stem(p) -> str:
    """'x__aug3' (or 'new_x__aug3' after a name collision) -> 'x'; originals keep their own stem."""
    stem = Path(p).stem
    if not AUG_PATTERN.search(stem):
        return stem
    stem = AUG_PATTERN.sub("", stem)
    while stem.startswith("new_"):
        stem = stem[4:]
    return stem


def file_md5(path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def detect_class_names(source_dir) -> list[str]:
    """Sub-folder names of source_dir (sorted) that contain at least one image."""
    source_dir = Path(source_dir)
    return sorted(d.name for d in source_dir.iterdir()
                  if d.is_dir() and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in d.rglob("*")))


def scan_source(source_dir, class_names=None, workers=8) -> tuple[pd.DataFrame, dict]:
    """Index SOURCE_DIR/<class>/*. Returns (df, report). class_names=None -> all sub-folders.

    df columns: path, file, label, augmented, orig_stem, group
    group = md5 of the original image (augmented copies inherit it). Exact duplicate files share a
    group; a group stored under two different labels is ambiguous and removed.
    """
    source_dir = Path(source_dir)
    class_names = class_names or detect_class_names(source_dir)
    missing = [c for c in class_names if not (source_dir / c).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing class folders in {source_dir}: {missing}")
    rows = []
    for c in class_names:
        for p in list_images(source_dir / c):
            rows.append({"path": str(p), "file": p.name, "label": c,
                         "augmented": bool(AUG_PATTERN.search(p.stem)), "orig_stem": original_stem(p)})
    df = pd.DataFrame(rows)
    orig = df[~df["augmented"]].copy()
    with ThreadPoolExecutor(workers) as pool:
        orig["md5"] = list(pool.map(file_md5, orig["path"]))
    stem_to_group = {(r.label, r.orig_stem): r.md5 for r in orig.itertuples()}
    df["group"] = [stem_to_group.get((l, s)) for l, s in zip(df["label"], df["orig_stem"])]
    orphans = df["group"].isna()
    df = df[~orphans].copy()
    n_lab = df.groupby("group")["label"].nunique()
    conflicts = set(n_lab[n_lab > 1].index)
    conflict_files = df.loc[df["group"].isin(conflicts) & ~df["augmented"], "path"].tolist()
    df = df[~df["group"].isin(conflicts)].reset_index(drop=True)
    report = {
        "files_found": len(rows),
        "originals": int((~df["augmented"]).sum()),
        "augmented_copies": int(df["augmented"].sum()),
        "orphan_augmented_dropped": int(orphans.sum()),
        "exact_duplicate_originals_merged": int((~df["augmented"]).sum() - df.loc[~df["augmented"], "group"].nunique()),
        "conflicting_label_groups_removed": len(conflicts),
        "conflicting_files": conflict_files[:50],
    }
    return df, report


def apply_label_fixes(df: pd.DataFrame, fixes_csv, class_names) -> tuple[pd.DataFrame, dict]:
    """Apply a hand-made label_fixes.csv (columns: file, action, new_label).

    file    : original file name (e.g. 'data1__Angry103.jpg') or full path
    action  : 'drop' (remove the original and all its copies) or 'relabel' (needs new_label)
    Rows with an empty action are ignored, so mislabel_candidates.csv can be edited and saved
    directly as label_fixes.csv.
    """
    fixes_csv = Path(fixes_csv)
    if not fixes_csv.exists():
        return df, {"label_fixes": "none"}
    fx = pd.read_csv(fixes_csv, dtype=str).fillna("")
    fx["action"] = fx["action"].str.strip().str.lower()
    fx = fx[fx["action"].isin(["drop", "relabel"])]
    df = df.copy()
    originals = df[~df["augmented"]]
    by_file = {}
    for r in originals.itertuples():
        by_file.setdefault(r.file, set()).add(r.group)
        by_file.setdefault(r.path, set()).add(r.group)
    dropped, relabelled, unknown = set(), {}, []
    for r in fx.itertuples():
        key = r.file.strip()
        groups = by_file.get(key) or by_file.get(Path(key).name)
        if not groups:
            unknown.append(key)
            continue
        if r.action == "drop":
            dropped |= groups
        else:
            new = r.new_label.strip().lower()
            if new not in class_names:
                raise ValueError(f"label_fixes.csv: unknown new_label '{r.new_label}' for {key}")
            for g in groups:
                relabelled[g] = new
    df = df[~df["group"].isin(dropped)]
    m = df["group"].isin(relabelled)
    df.loc[m, "label"] = df.loc[m, "group"].map(relabelled)
    return df.reset_index(drop=True), {"groups_dropped": len(dropped), "groups_relabelled": len(relabelled),
                                       "unknown_files_in_fixes": unknown[:20]}


def dhash(img_bgr) -> int:
    """64-bit difference hash of the grayscale image (robust to resizing, JPEG and colour)."""
    g = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def near_duplicate_pairs(hashes, max_hamming=3, chunk=2048):
    """Index pairs (i, j) with Hamming(hash_i, hash_j) <= max_hamming (max 3).
    The 64-bit hash is cut into 4 bands of 16 bits: two hashes that differ in <= 3 bits share at
    least one identical band, so only hashes inside the same band bucket are compared."""
    assert max_hamming <= 3, "banding with 4 bands is exact only up to distance 3"
    h = np.asarray(hashes, dtype=np.uint64)
    pairs = set()
    for b in range(4):
        band = ((h >> np.uint64(16 * b)) & np.uint64(0xFFFF)).astype(np.int64)
        order = np.argsort(band, kind="stable")
        bounds = np.flatnonzero(np.diff(band[order])) + 1
        for idx in np.split(order, bounds):
            if len(idx) < 2:
                continue
            hh = h[idx]
            for s0 in range(0, len(idx), chunk):
                d = _popcount64(hh[s0:s0 + chunk, None] ^ hh[None, :])
                ii, jj = np.nonzero(d <= max_hamming)
                keep = (s0 + ii) < jj
                pairs.update(zip(idx[s0 + ii[keep]].tolist(), idx[jj[keep]].tolist()))
    return pairs


def merge_near_duplicates(df: pd.DataFrame, max_hamming=3) -> tuple[pd.DataFrame, dict]:
    """Group originals whose face crops are near-identical (dHash distance <= max_hamming), e.g. the
    same photo stored twice in merged datasets. Near-duplicates must stay in the same split; if they
    carry different labels they are ambiguous and removed. Needs the 'dhash' column."""
    orig = df[~df["augmented"]].drop_duplicates("group").reset_index(drop=True)
    groups = orig["group"].to_numpy()
    parent = {g: g for g in groups}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    merged = 0
    for i, j in near_duplicate_pairs(orig["dhash"].to_numpy(dtype=np.uint64), max_hamming):
        a, b = find(groups[i]), find(groups[j])
        if a != b:
            parent[b] = a
            merged += 1
    df = df.copy()
    df["group"] = df["group"].map(lambda g: find(g) if g in parent else g)
    n_lab = df.groupby("group")["label"].nunique()
    conflicts = set(n_lab[n_lab > 1].index)
    conflict_rows = df.loc[df["group"].isin(conflicts) & ~df["augmented"], ["path", "label", "group"]]
    df = df[~df["group"].isin(conflicts)].reset_index(drop=True)
    return df, {"near_duplicate_merges": merged,
                "near_duplicate_label_conflict_groups_removed": len(conflicts),
                "near_duplicate_label_conflict_images_removed": int(len(conflict_rows)),
                "_conflicts": conflict_rows}


def make_splits(df: pd.DataFrame, class_names, test_fraction=0.15, n_folds=5, seed=42) -> pd.DataFrame:
    """Adds 'split' ('test' | 'trainval') and 'fold' (0..n_folds-1 for trainval, -1 for test).
    Splitting is done on groups (original + copies + near-duplicates), stratified by class."""
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(seed)
    g = df.drop_duplicates("group")[["group", "label"]].sort_values("group").reset_index(drop=True)
    test_groups = []
    for c in class_names:
        gs = g.loc[g["label"] == c, "group"].to_numpy().copy()
        rng.shuffle(gs)
        test_groups += list(gs[:int(round(len(gs) * test_fraction))])
    test_groups = set(test_groups)
    tv = g[~g["group"].isin(test_groups)].reset_index(drop=True)
    fold_of = {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(skf.split(tv["group"], tv["label"])):
        for gg in tv.loc[idx, "group"]:
            fold_of[gg] = k
    df = df.copy()
    df["split"] = np.where(df["group"].isin(test_groups), "test", "trainval")
    df["fold"] = df["group"].map(fold_of).fillna(-1).astype(int)
    assert df.groupby("group")["fold"].nunique().max() == 1, "group split across folds"
    return df


# ======================================================================================
# 2. Face cropping
# ======================================================================================
def is_gray(img_bgr) -> bool:
    if img_bgr.ndim == 2:
        return True
    small = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)
    return bool(np.abs(small[..., 0] - small[..., 1]).max() <= 8 and np.abs(small[..., 1] - small[..., 2]).max() <= 8)


def _crop_one(args):
    """-> (ok, crop_mode, dhash, gray); crop_mode = 'detected' | 'already_crop' | 'no_face'"""
    src, dst, crop_size, cropper_kw, use_detector, assume_crop_below = args
    img = read_bgr(src)
    if img is None:
        return False, "", 0, False
    if min(img.shape[:2]) < assume_crop_below or not use_detector:
        crop, mode = center_square(img), "already_crop"
    else:
        crop, found = _thread_cropper(cropper_kw).largest_face(img)
        mode = "detected" if found else "no_face"
    if crop.shape[0] > crop_size:            # downscale big crops; small ones (48x48) are kept as is
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    small = crop.shape[0] < 96              # tiny crops: lossless PNG; others: high-quality JPEG
    ok, buf = cv2.imencode(".png", crop) if small else cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return False, "", 0, False
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(dst))
    return True, mode, dhash(crop), is_gray(crop)


_TLS = __import__("threading").local()


def _thread_cropper(kw):
    # cv2.FaceDetectorYN is not thread-safe -> one detector per thread
    if getattr(_TLS, "cropper", None) is None or _TLS.kw != kw:
        _TLS.cropper, _TLS.kw = FaceCropper(**kw), kw
    return _TLS.cropper


def build_face_crops(df: pd.DataFrame, crop_dir, detector_path, margin=0.2, crop_size=320,
                     use_detector=True, assume_crop_below=100, workers=8, copies_per_original=0,
                     seed=42) -> tuple[pd.DataFrame, dict]:
    """Crop the largest face of every original (plus up to `copies_per_original` offline-augmented
    copies of each original, used for TRAINING only) into crop_dir. Images smaller than
    `assume_crop_below` px are already face crops and are only squared. Cached: existing crops built
    with the same settings are reused. Returns df restricted to the cropped images with columns
    crop_path, crop_mode ('detected' | 'already_crop' | 'no_face'), face_found, dhash, gray."""
    crop_dir = Path(crop_dir)
    keep = df[~df["augmented"]]
    if copies_per_original > 0:
        aug = df[df["augmented"]].sample(frac=1.0, random_state=seed)
        aug = aug.groupby("group", group_keys=False).head(copies_per_original)
        keep = pd.concat([keep, aug])
    keep = keep.copy()
    keep["crop_path"] = [str(crop_dir / r.label / f"{hashlib.md5(r.path.encode()).hexdigest()[:12]}_{Path(r.path).stem}")
                         for r in keep.itertuples()]      # extension (.png/.jpg) added below
    settings = {"margin": margin, "crop_size": crop_size, "use_detector": use_detector,
                "assume_crop_below": assume_crop_below, "detector": Path(detector_path).name, "v": 3}
    info_file = crop_dir / "crop_index.csv"
    settings_file = crop_dir / "crop_settings.json"
    cached = {}
    if info_file.exists() and settings_file.exists() and json.loads(settings_file.read_text()) == settings:
        old = pd.read_csv(info_file, dtype={"dhash": "uint64"})
        cached = {r.key: (r.crop_mode, int(r.dhash), bool(r.gray), r.crop_path) for r in old.itertuples()
                  if Path(r.crop_path).exists()}
    elif crop_dir.exists():
        import shutil
        shutil.rmtree(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    keep = keep.rename(columns={"crop_path": "key"})
    todo = [r for r in keep.itertuples() if r.key not in cached]
    kw = {"model_path": str(detector_path), "margin": margin}
    t0 = time.perf_counter()
    if todo:
        def job(r):
            # write to .jpg first; small crops are written as .png (lossless) by _crop_one's encoder choice
            img_ok = _crop_one((r.path, r.key + ".tmp", crop_size, kw, use_detector, assume_crop_below))
            return img_ok
        with ThreadPoolExecutor(workers) as pool:
            out = list(pool.map(job, todo))
        for r, (ok, mode, h, gray) in zip(todo, out):
            if ok:
                tmp = Path(r.key + ".tmp")
                with open(tmp, "rb") as f:
                    ext = ".png" if f.read(4) == b"\x89PNG" else ".jpg"
                final = Path(r.key + ext)
                tmp.replace(final)
                cached[r.key] = (mode, h, gray, str(final))
    keep = keep[keep["key"].isin(cached)].copy()
    keep["crop_path"] = keep["key"].map(lambda k: cached[k][3])
    keep["crop_mode"] = keep["key"].map(lambda k: cached[k][0])
    keep["face_found"] = keep["crop_mode"] == "detected"
    keep["dhash"] = keep["key"].map(lambda k: cached[k][1]).astype("uint64")
    keep["gray"] = keep["key"].map(lambda k: cached[k][2])
    keep = keep.drop(columns="key")
    pd.DataFrame({"key": list(cached), "crop_mode": [v[0] for v in cached.values()],
                  "dhash": [v[1] for v in cached.values()], "gray": [v[2] for v in cached.values()],
                  "crop_path": [v[3] for v in cached.values()]}).to_csv(info_file, index=False)
    settings_file.write_text(json.dumps(settings))
    o = keep[~keep["augmented"]]
    report = {"cropped_new": len(todo), "seconds": round(time.perf_counter() - t0, 1),
              "originals_cropped": int(len(o)), "unreadable": int(len(df[~df["augmented"]]) - len(o)),
              "crop_mode_originals": o["crop_mode"].value_counts().to_dict(),
              "grayscale_fraction_originals": float(o["gray"].mean()) if len(o) else 0.0,
              "augmented_copies_cropped": int(keep["augmented"].sum())}
    return keep.reset_index(drop=True), report


# ======================================================================================
# 3. Data loading + augmentation
# ======================================================================================
class FaceDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(map(str, paths))
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), int(self.labels[i])


def build_transforms(img_size, mean=IMAGENET_MEAN, std=IMAGENET_STD, train=True, erasing=0.1, grayscale=False):
    """grayscale=True converts every image to 3-channel gray (the exported ONNX model does the same
    internally), so colour cannot become a shortcut when most training images are gray."""
    from torchvision.transforms import v2 as T

    norm = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)]
    gray = [T.Grayscale(num_output_channels=3)] if grayscale else []
    if not train:
        # identical to fer_infer_onnx.preprocess_faces: short side -> img_size, centre crop
        return T.Compose([*gray, T.Resize(img_size, antialias=True), T.CenterCrop(img_size), *norm])
    t = [
        *gray,
        T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomAffine(degrees=12, translate=(0.05, 0.05))], p=0.5),
        T.RandomApply([T.ColorJitter(0.35, 0.35, 0.25, 0.03)], p=0.8),
        *([] if grayscale else [T.RandomGrayscale(p=0.1)]),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.15),
        *norm,
    ]
    if erasing > 0:   # small patches only, so eyes AND mouth are never both hidden
        t.append(T.RandomErasing(p=erasing, scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0))
    return T.Compose(t)


# ======================================================================================
# 4. Models
# ======================================================================================
HSEMOTION_CLASSES_8 = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
HSEMOTION_CLASSES_7 = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
# map your class names to HSEmotion's (AffectNet) class names
HSEMOTION_NAME_MAP = {"angry": "anger", "anger": "anger", "disgust": "disgust", "disgusted": "disgust",
                      "fear": "fear", "fearful": "fear", "happy": "happiness", "happiness": "happiness",
                      "neutral": "neutral", "sad": "sadness", "sadness": "sadness", "surprise": "surprise",
                      "surprised": "surprise", "suprise": "surprise", "contempt": "contempt"}
HSEMOTION_URLS = [
    "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/{name}.pt",
    "https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/{name}.pt?raw=true",
    "https://github.com/av-savchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/{name}.pt",
]

# Backbone registry: per-model defaults (all can be overridden in the notebook)
BACKBONES = {
    # AffectNet-pretrained EfficientNets (HSEmotion). Already know angry/fear/happy/neutral/sad/surprise.
    "hsemotion_b0": dict(kind="hsemotion", weights="enet_b0_8_best_afew", img_size=224,
                         lr_backbone=1e-4, lr_head=1e-3, batch=64, drop_rate=0.3, drop_path=0.1,
                         freeze_epochs=0, fallback="convnextv2_tiny"),
    "hsemotion_b2": dict(kind="hsemotion", weights="enet_b2_8", img_size=260,
                         lr_backbone=1e-4, lr_head=1e-3, batch=48, drop_rate=0.3, drop_path=0.1,
                         freeze_epochs=0, fallback="convnextv2_tiny"),
    # Strong ImageNet-22k backbones (timm, weights from Hugging Face)
    "convnextv2_tiny": dict(kind="timm", name="convnextv2_tiny.fcmae_ft_in22k_in1k", img_size=224,
                            lr_backbone=5e-5, lr_head=5e-4, batch=48, drop_rate=0.3, drop_path=0.2,
                            freeze_epochs=2),
    "effnetv2_s": dict(kind="timm", name="tf_efficientnetv2_s.in21k_ft_in1k", img_size=256,
                       lr_backbone=1e-4, lr_head=1e-3, batch=48, drop_rate=0.3, drop_path=0.2,
                       freeze_epochs=2),
    # Self-supervised ViT: robust features with little data
    "dinov2_vits14": dict(kind="timm", name="vit_small_patch14_dinov2.lvd142m", img_size=224,
                          lr_backbone=2e-5, lr_head=5e-4, batch=48, drop_rate=0.2, drop_path=0.1,
                          freeze_epochs=2, extra=dict(img_size=224)),
    "convnext_small": dict(kind="timm", name="convnext_small.fb_in22k_ft_in1k", img_size=224,
                           lr_backbone=5e-5, lr_head=5e-4, batch=32, drop_rate=0.3, drop_path=0.3,
                           freeze_epochs=2),
}


class _TimmStub(nn.Module):
    """Stand-in for any timm class while unpickling an old full-model checkpoint. nn.Module's
    __setstate__ restores _parameters/_buffers/_modules, so state_dict() works without the old
    timm code; we then load the weights into a freshly built model of the current timm."""
    _orig = ""

    def __init__(self, *a, **k):
        super().__init__()


_STUB_CACHE: dict[str, type] = {}


def _stub_class(module, name):
    key = f"{module}.{name}"
    if key not in _STUB_CACHE:
        _STUB_CACHE[key] = type(name, (_TimmStub,), {"_orig": key})
    return _STUB_CACHE[key]


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("timm"):
            return _stub_class(module, name)
        return super().find_class(module, name)


_stub_pickle = types.SimpleNamespace(Unpickler=_StubUnpickler, load=pickle.load, __name__="stub_pickle")


def download_hsemotion(name, cache_dir) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{name}.pt"
    home = Path.home() / ".hsemotion" / f"{name}.pt"     # reuse the hsemotion package cache if present
    if not dst.exists() and home.exists() and home.stat().st_size > 1e6:
        dst.write_bytes(home.read_bytes())
    if dst.exists() and dst.stat().st_size > 1e6:
        return dst
    errors = []
    for url in HSEMOTION_URLS:
        url = url.format(name=name)
        try:
            print("Downloading", url)
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 1e6:
                raise IOError(f"file too small ({tmp.stat().st_size} bytes) - not a model")
            tmp.replace(dst)
            return dst
        except Exception as e:  # try the next mirror
            errors.append(f"{url}: {e}")
    raise RuntimeError("Could not download HSEmotion weights. Download the file manually from "
                       "https://github.com/HSE-asavchenko/face-emotion-recognition/tree/main/models/affectnet_emotions "
                       f"and put it at {dst}\n" + "\n".join(errors))


def load_hsemotion(name, class_names, cache_dir, drop_rate=0.3, drop_path=0.1, verbose=True):
    """Build a current-timm EfficientNet with HSEmotion's AffectNet weights; the 6-class head is
    initialised from the matching rows of the 8-class AffectNet head (works zero-shot)."""
    import timm

    path = download_hsemotion(name, cache_dir)
    obj = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_stub_pickle)
    sd = obj.state_dict() if isinstance(obj, nn.Module) else obj
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    # classifier: Sequential(Linear) -> 'classifier.0.*'
    sd = {re.sub(r"^classifier\.0\.", "classifier.", k): v for k, v in sd.items()}
    w_old, b_old = sd.pop("classifier.weight"), sd.pop("classifier.bias")
    src_classes = HSEMOTION_CLASSES_7 if w_old.shape[0] == 7 else HSEMOTION_CLASSES_8

    arch = "b2" if "_b2_" in name + "_" else "b0"
    # 'tf_' EfficientNets (TF 'same' padding, BN eps 1e-3) were used by HSEmotion; check the BN eps
    eps = getattr(getattr(obj, "bn1", None), "eps", 1e-3) if isinstance(obj, nn.Module) else 1e-3
    timm_name = f"tf_efficientnet_{arch}" if abs(eps - 1e-3) < 1e-6 else f"efficientnet_{arch}"
    model = timm.create_model(timm_name, pretrained=False, num_classes=len(class_names),
                              drop_rate=drop_rate, drop_path_rate=drop_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("classifier.")]
    n_total = len(model.state_dict())
    if len(missing) > 0.02 * n_total or len(unexpected) > 0.02 * n_total:
        raise RuntimeError(f"HSEmotion weights do not match {timm_name}: {len(missing)} missing, "
                           f"{len(unexpected)} unexpected keys (e.g. {missing[:3]} {unexpected[:3]})")
    head = model.get_classifier()
    with torch.no_grad():
        mapped = 0
        for i, c in enumerate(class_names):
            src = HSEMOTION_NAME_MAP.get(c.lower())
            if src in src_classes:
                head.weight[i].copy_(w_old[src_classes.index(src)])
                head.bias[i].copy_(b_old[src_classes.index(src)])
                mapped += 1
    if verbose:
        print(f"HSEmotion '{name}' -> timm '{timm_name}': {n_total - len(missing)}/{n_total} tensors loaded, "
              f"{mapped}/{len(class_names)} head rows initialised from AffectNet classes")
    return model


def create_model(backbone: str, class_names, cache_dir, pretrained=True, overrides=None, verbose=True):
    """Returns (model, spec). spec = registry entry (+ overrides) with the resolved backbone name,
    image size, mean and std. HSEmotion falls back to spec['fallback'] if it cannot be loaded."""
    import timm

    spec = dict(BACKBONES[backbone])
    spec.update(overrides or {})
    spec["backbone"] = backbone
    if spec["kind"] == "hsemotion":
        try:
            model = load_hsemotion(spec["weights"], class_names, cache_dir, spec["drop_rate"],
                                   spec["drop_path"], verbose) if pretrained else \
                timm.create_model("tf_efficientnet_b2" if "b2" in spec["weights"] else "tf_efficientnet_b0",
                                  num_classes=len(class_names))
            spec.update(mean=IMAGENET_MEAN, std=IMAGENET_STD, resolved=f"hsemotion:{spec['weights']}")
            return model, spec
        except Exception as e:
            fb = spec.get("fallback")
            print(f"WARNING: HSEmotion backbone failed ({e}).\n         Falling back to '{fb}'.")
            model, fspec = create_model(fb, class_names, cache_dir, pretrained, None, verbose)
            fspec["backbone"] = backbone              # keep the output folder name stable
            fspec["fell_back_from"] = backbone
            return model, fspec
    model = timm.create_model(spec["name"], pretrained=pretrained, num_classes=len(class_names),
                              drop_rate=spec["drop_rate"], drop_path_rate=spec["drop_path"],
                              **spec.get("extra", {}))
    cfg = model.pretrained_cfg or {}
    spec.update(mean=tuple(cfg.get("mean", IMAGENET_MEAN)), std=tuple(cfg.get("std", IMAGENET_STD)),
                resolved=spec["name"])
    return model, spec


# ======================================================================================
# 5. Training
# ======================================================================================
@dataclass
class TrainConfig:
    epochs: int = 35
    patience: int = 10                 # early stopping on validation accuracy
    batch: int = 48
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 2.0
    freeze_epochs: int = 0             # head-only epochs at the start (random heads)
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2           # 0 disables mixup
    mixup_prob: float = 0.5
    class_weighting: str = "sqrt"      # 'none' | 'sqrt' | 'inverse'
    ema_decay: float = 0.998           # 0 disables EMA
    erasing: float = 0.1
    grad_clip: float = 1.0
    workers: int = 4
    amp: bool = True
    tta: bool = True                   # flip TTA for validation predictions
    seed: int = 42


def _param_groups(model, lr_backbone, lr_head, weight_decay):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups = {(h, d): [] for h in (0, 1) for d in (0, 1)}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or n.endswith(".bias") or "pos_embed" in n or "cls_token" in n
        groups[(int(id(p) in head_ids), int(no_decay))].append(p)
    out = []
    for (is_head, no_decay), ps in groups.items():
        if ps:
            out.append({"params": ps, "lr": lr_head if is_head else lr_backbone,
                        "weight_decay": 0.0 if no_decay else weight_decay})
    return out


def _set_backbone_trainable(model, trainable: bool):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad_(trainable)


def _make_ema(model, decay):
    if decay <= 0:
        return None
    from timm.utils import ModelEmaV3
    return ModelEmaV3(model, decay=decay, use_warmup=True)


def _amp_ctx(device, enabled):
    if enabled and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict_loader(model, loader, device, tta=True, amp=True):
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with _amp_ctx(device, amp):
            logits = model(x).float()
            p = logits.softmax(1)
            if tta:
                p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        probs.append(p.cpu())
    return torch.cat(probs).numpy()


def train_one_fold(model, spec, train_df, val_df, class_names, cfg: TrainConfig, out_dir, device,
                   log_every=0):
    """Train on train_df, select the best epoch on val_df (accuracy, then loss). Saves best.pt
    (state_dict + spec) and history.csv in out_dir. Returns dict with best metrics and val probs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cid = {c: i for i, c in enumerate(class_names)}
    ytr = train_df["label"].map(cid).to_numpy()
    yva = val_df["label"].map(cid).to_numpy()

    size, mean, std = spec["img_size"], spec["mean"], spec["std"]
    gray = bool(spec.get("grayscale", False))
    tr_ds = FaceDataset(train_df["crop_path"], ytr, build_transforms(size, mean, std, True, cfg.erasing, gray))
    va_ds = FaceDataset(val_df["crop_path"], yva, build_transforms(size, mean, std, False, grayscale=gray))
    pw = cfg.workers > 0
    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch, shuffle=True, drop_last=len(tr_ds) > cfg.batch,
                       num_workers=cfg.workers, pin_memory=device.type == "cuda", persistent_workers=pw)
    va_dl = DataLoader(va_ds, batch_size=cfg.batch * 2, shuffle=False, num_workers=cfg.workers,
                       pin_memory=device.type == "cuda", persistent_workers=pw)

    counts = np.bincount(ytr, minlength=len(class_names)).astype(np.float64)
    if cfg.class_weighting == "none":
        w = np.ones_like(counts)
    else:
        w = (counts.sum() / np.maximum(counts, 1))
        w = np.sqrt(w) if cfg.class_weighting == "sqrt" else w
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    val_crit = nn.CrossEntropyLoss()

    model = model.to(device).to(memory_format=torch.channels_last)
    if cfg.freeze_epochs > 0:
        _set_backbone_trainable(model, False)
    opt = torch.optim.AdamW(_param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))
    steps_per_epoch = len(tr_dl)
    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(cfg.warmup_epochs * steps_per_epoch))
    base_lrs = None

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))

    ema = _make_ema(model, cfg.ema_decay)
    use_scaler = cfg.amp and device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    history, best, bad, step = [], None, 0, 0
    yva_t = torch.tensor(yva)
    for epoch in range(cfg.epochs):
        if cfg.freeze_epochs > 0 and epoch == cfg.freeze_epochs:
            _set_backbone_trainable(model, True)
            opt = torch.optim.AdamW(_param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))
            base_lrs = None
        if base_lrs is None:
            base_lrs = [g["lr"] for g in opt.param_groups]
        model.train()
        t0, run_loss, n_seen = time.perf_counter(), 0.0, 0
        for x, y in tr_dl:
            f = lr_factor(step)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * f
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            use_mix = cfg.mixup_alpha > 0 and np.random.rand() < cfg.mixup_prob
            if use_mix:
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[perm]
            with _amp_ctx(device, cfg.amp):
                out = model(x)
                loss = lam * crit(out, y) + (1 - lam) * crit(out, y[perm]) if use_mix else crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model, step=step)
            run_loss += float(loss.detach()) * x.size(0)
            n_seen += x.size(0)
            step += 1

        # validate raw and EMA weights, keep whichever is better
        cands = [("raw", model)] + ([("ema", ema.module)] if ema is not None else [])
        ep_best = None
        for tag, m in cands:
            p = predict_loader(m, va_dl, device, tta=cfg.tta, amp=cfg.amp)
            acc = float((p.argmax(1) == yva).mean())
            vloss = float(val_crit(torch.log(torch.tensor(p).clamp_min(1e-7)), yva_t))
            if ep_best is None or (acc, -vloss) > (ep_best[1], -ep_best[2]):
                ep_best = (tag, acc, vloss, p, m)
        tag, acc, vloss, p, m = ep_best
        rec = {"epoch": epoch + 1, "train_loss": run_loss / max(n_seen, 1), "val_loss": vloss, "val_acc": acc,
               "weights": tag, "lr_head": opt.param_groups[-1]["lr"], "sec": round(time.perf_counter() - t0, 1)}
        history.append(rec)
        improved = best is None or (acc, -vloss) > (best["val_acc"], -best["val_loss"])
        if improved:
            best = {**rec, "val_probs": p}
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in m.state_dict().items()},
                        "spec": spec, "class_names": list(class_names), "epoch": epoch + 1,
                        "val_acc": acc, "config": asdict(cfg)}, out_dir / "best.pt")
            bad = 0
        else:
            bad += 1
        if log_every and ((epoch + 1) % log_every == 0 or improved):
            print(f"  ep {epoch + 1:3d} loss {rec['train_loss']:.3f} | val loss {vloss:.3f} acc {acc:.4f} "
                  f"[{tag}]{' *' if improved else ''} ({rec['sec']}s)")
        if bad >= cfg.patience and epoch + 1 > cfg.freeze_epochs + cfg.warmup_epochs:
            break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    return {"best_epoch": best["epoch"], "val_acc": best["val_acc"], "val_loss": best["val_loss"],
            "val_probs": best["val_probs"], "history": pd.DataFrame(history)}


def load_fold_model(ckpt_path, cache_dir=None, device="cpu"):
    """Rebuild a model from best.pt (no weight download needed)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    spec = ck["spec"]
    import timm
    if spec["resolved"].startswith("hsemotion:"):
        sd = ck["state_dict"]
        width = sd["conv_head.weight"].shape[0]
        arch = "b2" if width == 1408 else "b0"
        # tf_ vs plain variant: tf_ was used unless the loader detected otherwise; try both
        last_err = None
        for nm in (f"tf_efficientnet_{arch}", f"efficientnet_{arch}"):
            try:
                model = timm.create_model(nm, pretrained=False, num_classes=len(ck["class_names"]))
                model.load_state_dict(sd)
                break
            except Exception as e:
                last_err = e
        else:
            raise last_err
    else:
        model = timm.create_model(spec["name"], pretrained=False, num_classes=len(ck["class_names"]),
                                  **spec.get("extra", {}))
        model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), spec, ck


@torch.no_grad()
def predict_paths(model, spec, paths, device, tta=True, batch=128, workers=0, amp=True):
    ds = FaceDataset(paths, np.zeros(len(paths)), build_transforms(spec["img_size"], spec["mean"], spec["std"], False,
                                                                   grayscale=bool(spec.get("grayscale", False))))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    model = model.to(device).to(memory_format=torch.channels_last)
    return predict_loader(model, dl, device, tta=tta, amp=amp)


# ======================================================================================
# 6. Ensemble export
# ======================================================================================
class EnsembleWrapper(nn.Module):
    """Input : float32 N x 3 x S x S, RGB in [0, 1] (S = input_size)
    Output: N x C class probabilities = mean over members (and over flips if tta)."""

    def __init__(self, members, input_size, tta=True):
        super().__init__()
        self.models = nn.ModuleList([m for m, _ in members])
        self.sizes = [int(s["img_size"]) for _, s in members]
        self.grays = [bool(s.get("grayscale", False)) for _, s in members]
        for i, (_, s) in enumerate(members):
            self.register_buffer(f"mean{i}", torch.tensor(s["mean"]).view(1, 3, 1, 1), persistent=False)
            self.register_buffer(f"std{i}", torch.tensor(s["std"]).view(1, 3, 1, 1), persistent=False)
        self.input_size = int(input_size)
        self.tta = bool(tta)

    def forward(self, x):
        total = None
        n = 0
        views = [x, torch.flip(x, dims=[3])] if self.tta else [x]
        for i, m in enumerate(self.models):
            xi = x
            if self.grays[i]:   # ITU-R 601 luma, same as PIL 'L' used in training
                xi = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand(-1, 3, -1, -1)
            if self.sizes[i] != self.input_size:
                xi = F.interpolate(xi, size=(self.sizes[i], self.sizes[i]), mode="bilinear", align_corners=False)
            mean, std = getattr(self, f"mean{i}"), getattr(self, f"std{i}")
            vs = [xi, torch.flip(xi, dims=[3])] if self.tta else [xi]
            for v in vs:
                p = m((v - mean) / std).softmax(1)
                total = p if total is None else total + p
                n += 1
        return total / n


def export_onnx(wrapper: EnsembleWrapper, onnx_path, opset=17):
    wrapper = wrapper.eval().cpu().float()
    s = wrapper.input_size
    dummy = torch.rand(2, 3, s, s)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=["images"], output_names=["probs"], opset_version=opset)
    try:   # classic TorchScript exporter (stable for CNNs/ViTs)
        torch.onnx.export(wrapper, (dummy,), str(onnx_path), dynamo=False,
                          dynamic_axes={"images": {0: "batch"}, "probs": {0: "batch"}}, **kw)
    except TypeError:   # very old torch without the dynamo flag
        torch.onnx.export(wrapper, (dummy,), str(onnx_path),
                          dynamic_axes={"images": {0: "batch"}, "probs": {0: "batch"}}, **kw)
    except Exception as e:
        print("Classic ONNX exporter failed, trying the dynamo exporter:", e)
        batch = torch.export.Dim("batch", min=1, max=1024)
        torch.onnx.export(wrapper, (dummy,), str(onnx_path), dynamo=True,
                          dynamic_shapes={"x": {0: batch}}, **kw)
    # make it self-contained (dynamo may write external .data files)
    import onnx
    m = onnx.load(str(onnx_path))
    onnx.save(m, str(onnx_path), save_as_external_data=False)
    for extra in onnx_path.parent.glob(onnx_path.name + ".data"):
        extra.unlink()
    with torch.no_grad():
        ref = wrapper(dummy).numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]
    return float(np.abs(ref - got).max())


# ======================================================================================
# 7. Analysis helpers
# ======================================================================================
def bootstrap_ci(y_true, y_pred, n=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    accs = (y_true[idx] == y_pred[idx]).mean(1)
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def mislabel_candidates(df, probs, class_names, top=150):
    """Rank originals by how strongly the out-of-fold model disagrees with their label."""
    cid = {c: i for i, c in enumerate(class_names)}
    y = df["label"].map(cid).to_numpy()
    pred = probs.argmax(1)
    out = pd.DataFrame({
        "file": df["file"].to_numpy(), "path": df["path"].to_numpy(), "crop_path": df["crop_path"].to_numpy(),
        "label": df["label"].to_numpy(), "predicted": [class_names[i] for i in pred],
        "p_label": probs[np.arange(len(y)), y], "p_predicted": probs.max(1),
    })
    out = out[out["label"] != out["predicted"]].copy()
    out["suspicion"] = out["p_predicted"] - out["p_label"]
    out = out.sort_values("suspicion", ascending=False).head(top)
    out["action"] = ""          # fill with 'drop' or 'relabel'
    out["new_label"] = ""       # for 'relabel'
    return out.reset_index(drop=True)


def contact_sheet(rows: pd.DataFrame, out_path, cols=8, cell=160, title_fn=None):
    """Grid of crops with a caption each (OpenCV only, fast for hundreds of images)."""
    n = len(rows)
    if n == 0:
        return None
    rows_n = math.ceil(n / cols)
    cap = 36
    sheet = np.full((rows_n * (cell + cap), cols * cell, 3), 255, np.uint8)
    for k, r in enumerate(rows.itertuples()):
        img = read_bgr(r.crop_path)
        if img is None:
            continue
        img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        y0, x0 = (k // cols) * (cell + cap), (k % cols) * cell
        sheet[y0:y0 + cell, x0:x0 + cell] = img
        lines = title_fn(r) if title_fn else [str(k)]
        for li, t in enumerate(lines[:2]):
            cv2.putText(sheet, t[:26], (x0 + 3, y0 + cell + 14 + 15 * li), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(out_path))
    return out_path
