"""Dataset ingestion, quality checks, duplicate suppression, augmented object materialization."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from sqlalchemy import select, func
from tqdm import tqdm
import cv2
import numpy as np

from .augmentation import augment, variant_seed, RECIPE_VERSION
from .database import ImageAsset, Augmentation
from .images import encode_png, decode_image, assess, hamming

DATASETS = {"cifar100": {"channels": 3, "classes": 100, "train": 50000, "test": 10000},
            "fashionmnist": {"channels": 1, "classes": 10, "train": 60000, "test": 10000}}


def load_official_dataset(name: str, root: str, train: bool):
    """Downloads official dataset archives via torchvision, then reads the originals."""
    from torchvision.datasets import CIFAR100, FashionMNIST
    if name == "cifar100":
        ds = CIFAR100(root=root, train=train, download=True)
        return np.asarray(ds.data), list(ds.targets)
    if name == "fashionmnist":
        ds = FashionMNIST(root=root, train=train, download=True)
        return ds.data.numpy(), list(ds.targets)
    raise ValueError(f"Unsupported dataset {name}")


def ingest(name, *, storage, session, config, dataset_dir="datasets", limit=None, batch_size=500):
    if name not in DATASETS:
        raise ValueError(name)
    counts = {"uploaded": 0, "skipped_existing": 0, "quarantined": 0}
    for split in ("train", "test"):
        images, targets = load_official_dataset(name, dataset_dir, split == "train")
        known = set(session.scalars(select(ImageAsset.source_index).where(
            ImageAsset.dataset == name, ImageAsset.split == split)).all())
        total = len(images) if limit is None else min(limit, len(images))
        pending = []
        for idx in tqdm(range(total), desc=f"ingest {name}/{split}"):
            if idx in known:
                counts["skipped_existing"] += 1
                continue
            image = images[idx]
            if name == "cifar100":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            data = encode_png(image)
            qa = assess(data)
            key = config.key("raw", name, split, f"{idx:06d}.png")
            storage.put(key, data)
            pending.append(ImageAsset(dataset=name, split=split, source_index=idx,
                                      label=int(targets[idx]), key=key, **qa))
            counts["uploaded"] += 1
            counts["quarantined"] += qa["status"] != "ready"
            if len(pending) >= batch_size:
                session.add_all(pending)
                session.commit()
                pending.clear()
        if pending:
            session.add_all(pending)
            session.commit()
    return counts


def clean(*, storage, session, dataset: str, limit=None):
    q = select(ImageAsset).where(ImageAsset.dataset == dataset, ImageAsset.status.in_(["ready", "quarantined"])).order_by(ImageAsset.id)
    if limit is not None:
        q = q.limit(limit)
    counts = {"checked": 0, "ready": 0, "quarantined": 0}
    for asset in session.scalars(q).yield_per(500):
        try:
            qa = assess(storage.get(asset.key))
            for field, value in qa.items():
                setattr(asset, field, value)
        except (ValueError, OSError, KeyError, FileNotFoundError) as e:
            asset.status = "quarantined"
            asset.reason = "unreadable_or_missing"
        counts["checked"] += 1
        counts[asset.status] += 1
        if counts["checked"] % 500 == 0:
            session.commit()
    session.commit()
    return counts


def deduplicate(*, session, dataset: str, near_threshold=0):
    """Keep test intact. Quarantine training copies also present in official test.

    Exact matching uses SHA-256 of canonical PNG bytes. Near matching uses a
    conservative pHash cutoff and class agreement, on *training records only*.
    Blocked pHash indices avoid full O(N^2) scans at CIFAR scale. Disabling
    near-threshold (0) avoids dropping legitimate similar 32px images.
    """
    if not 0 <= near_threshold <= 3:
        raise ValueError("near_threshold must be 0..3 for 4-way 16-bit index")
    test_hashes = {a.sha256: a.id for a in session.scalars(
        select(ImageAsset).where(ImageAsset.dataset == dataset, ImageAsset.split == "test"))}
    train = list(session.scalars(select(ImageAsset).where(
        ImageAsset.dataset == dataset, ImageAsset.split == "train").order_by(ImageAsset.id)))
    by_sha = {}
    phash_groups = defaultdict(list)
    counts = {"train_checked": 0, "exact_duplicates": 0, "near_duplicates": 0,
              "train_test_collisions": 0, "label_conflicts": 0, "ready": 0}
    for a in train:
        # Always reset to quality policy first; reruns are idempotent.
        if a.reason in ("exact_duplicate", "near_duplicate", "train_test_collision", "label_conflict"):
            a.status, a.reason, a.duplicate_of_id = "ready", None, None
        counts["train_checked"] += 1
        if a.status != "ready":
            continue
        if a.sha256 in test_hashes:
            a.status, a.reason, a.duplicate_of_id = "duplicate", "train_test_collision", test_hashes[a.sha256]
            counts["train_test_collisions"] += 1
            continue
        prior = by_sha.get(a.sha256)
        if prior:
            a.status = "duplicate" if a.label == prior.label else "quarantined"
            a.reason = "exact_duplicate" if a.label == prior.label else "label_conflict"
            a.duplicate_of_id = prior.id
            counts["exact_duplicates" if a.label == prior.label else "label_conflicts"] += 1
            continue
        if near_threshold > 0:
            h = int(a.phash, 16)
            possible = {}
            for i in range(4):
                for match in phash_groups[(a.label, i, (h >> (i * 16)) & 0xFFFF)]:
                    possible[match.id] = match
            duplicate = next((old for old in possible.values()
                              if hamming(a.phash, old.phash) <= near_threshold), None)
            if duplicate:
                a.status, a.reason, a.duplicate_of_id = "duplicate", "near_duplicate", duplicate.id
                counts["near_duplicates"] += 1
                continue
            for i in range(4):
                phash_groups[(a.label, i, (h >> (i * 16)) & 0xFFFF)].append(a)
        by_sha[a.sha256] = a
        counts["ready"] += 1
    session.commit()
    return counts


def materialize(*, storage, session, config, dataset: str, variants=9, limit=None, batch_size=250):
    """Generate actual augmented PNG objects in S3, train split only.

    Resumable: persistent metadata unique on image/version/variant_no. Safe
    reruns skip completed variants. No validation/test images are transformed.
    """
    if not 1 <= variants <= 32:
        raise ValueError("variants must be between 1 and 32")
    q = select(ImageAsset).where(ImageAsset.dataset == dataset,
                                 ImageAsset.split == "train", ImageAsset.status == "ready").order_by(ImageAsset.id)
    if limit is not None:
        q = q.limit(limit)
    assets = list(session.scalars(q))
    existing = set(session.execute(select(Augmentation.asset_id, Augmentation.variant_no).join(ImageAsset).where(
        ImageAsset.dataset == dataset, Augmentation.recipe_version == RECIPE_VERSION)).all())
    counts = {"source_images": len(assets), "uploaded": 0, "skipped_existing": 0}
    pending = []
    for a in tqdm(assets, desc=f"augment {dataset}"):
        missing = [v for v in range(1, variants + 1) if (a.id, v) not in existing]
        if not missing:
            counts["skipped_existing"] += variants
            continue
        image = decode_image(storage.get(a.key))
        for v in range(1, variants + 1):
            if v not in missing:
                counts["skipped_existing"] += 1
                continue
            seed = variant_seed(dataset, a.split, a.source_index, v)
            encoded = encode_png(augment(image, seed, dataset=dataset))
            key = config.key("augmented", RECIPE_VERSION, dataset, "train", f"{a.source_index:06d}", f"v{v:02d}.png")
            storage.put(key, encoded)
            pending.append(Augmentation(asset_id=a.id, recipe_version=RECIPE_VERSION,
                                        variant_no=v, seed=seed, key=key,
                                        sha256=sha256(encoded).hexdigest()))
            counts["uploaded"] += 1
            if len(pending) >= batch_size:
                session.add_all(pending)
                session.commit()
                pending.clear()
    if pending:
        session.add_all(pending)
        session.commit()
    return counts


def inspect(session, dataset=None):
    q = select(ImageAsset.dataset, ImageAsset.split, ImageAsset.status, func.count()).group_by(
        ImageAsset.dataset, ImageAsset.split, ImageAsset.status)
    if dataset:
        q = q.where(ImageAsset.dataset == dataset)
    rows = session.execute(q).all()
    vq = select(ImageAsset.dataset, func.count(Augmentation.id)).join(Augmentation, ImageAsset.id == Augmentation.asset_id).group_by(ImageAsset.dataset)
    if dataset:
        vq = vq.where(ImageAsset.dataset == dataset)
    return {"assets": [{"dataset": d, "split": s, "status": st, "count": n}
                        for d, s, st, n in rows],
            "materialized_variants": dict(session.execute(vq).all())}


def import_s3_manifest(*, storage, session, manifest_path: str, limit=None, batch_size=500):
    """Index images already in S3: CSV with dataset,split,source_index,label,key.

    Does not re-upload images. Downloads each listed key once to check actual
    content, calculate canonical metadata and reject unlisted/missing labels.
    Existing (dataset,split,source_index) records are ignored for safe resumes.
    """
    import csv
    path = Path(manifest_path)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"dataset", "split", "source_index", "label", "key"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires columns: dataset,split,source_index,label,key")
        # Loading 130K rows into memory is modest; streamed S3 keys are still processed sequentially.
        known = set(session.execute(select(ImageAsset.dataset, ImageAsset.split,
                                           ImageAsset.source_index)).all())
        seen = set()
        pending = []
        counts = {"indexed": 0, "skipped_existing": 0, "quarantined": 0}
        for line, row in enumerate(reader, start=2):
            if limit is not None and counts["indexed"] >= limit:
                break
            dataset, split = row["dataset"], row["split"]
            if dataset not in DATASETS or split not in ("train", "test"):
                raise ValueError(f"Invalid dataset or split in line {line}")
            source_index, label = int(row["source_index"]), int(row["label"])
            if source_index < 0 or not 0 <= label < DATASETS[dataset]["classes"]:
                raise ValueError(f"Bad index/label on line {line}")
            key = row["key"]
            if not key or key.startswith("/") or ".." in key.split("/"):
                raise ValueError(f"Unsafe/missing S3 key on line {line}")
            identity = (dataset, split, source_index)
            if identity in known or identity in seen:
                counts["skipped_existing"] += 1
                continue
            qa = assess(storage.get(key))
            seen.add(identity)
            counts["indexed"] += 1
            counts["quarantined"] += qa["status"] == "quarantined"
            pending.append(ImageAsset(dataset=dataset, split=split, source_index=source_index,
                                      label=label, key=key, **qa))
            if len(pending) >= batch_size:
                session.add_all(pending)
                session.commit()
                pending.clear()
        if pending:
            session.add_all(pending)
            session.commit()
    return counts
