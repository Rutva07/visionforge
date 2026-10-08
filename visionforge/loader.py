"""SQL indexed datasets; cached image bytes loaded lazily from S3 during training."""
import hashlib
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sqlalchemy import select

from .database import ImageAsset, Augmentation
from .images import decode_image
from .storage import CachedStorage


def val_partition(dataset: str, source_index: int) -> bool:
    """Stable 10% train-to-validation partition, identical across experiment arms."""
    digest = hashlib.sha256(f"{dataset}:{source_index}:val-v1".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10 == 0


def image_channels(dataset: str) -> int:
    return 3 if dataset == "cifar100" else 1


class IndexedImages(Dataset):
    def __init__(self, *, session, storage, cache_dir, dataset, partition, mode="baseline", limit=None):
        if partition not in ("train", "val", "test"):
            raise ValueError(partition)
        if mode not in ("baseline", "augmented"):
            raise ValueError(mode)
        self.dataset, self.partition, self.mode = dataset, partition, mode
        self.store = CachedStorage(storage, cache_dir)
        split = "test" if partition == "test" else "train"
        q = select(ImageAsset).where(ImageAsset.dataset == dataset, ImageAsset.split == split,
                                     ImageAsset.status == "ready").order_by(ImageAsset.source_index)
        assets = list(session.scalars(q))
        if partition != "test":
            assets = [a for a in assets if val_partition(dataset, a.source_index) == (partition == "val")]
        if limit is not None:
            assets = assets[:limit]
        self.samples = [(a.key, a.label) for a in assets]
        # Only add physically materialized variants to train; nothing leaks to val/test.
        if partition == "train" and mode == "augmented" and assets:
            ids = [a.id for a in assets]
            # SQLAlchemy expands large IN lists; shard for engines with limited parameter counts.
            for offset in range(0, len(ids), 500):
                qv = select(Augmentation.key, ImageAsset.label).join(ImageAsset).where(
                    Augmentation.asset_id.in_(ids[offset:offset + 500]),
                    Augmentation.status == "ready")
                self.samples.extend(session.execute(qv).all())
        self.channels = image_channels(dataset)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        key, label = self.samples[i]
        arr = decode_image(self.store.get(key))
        if arr.ndim == 2:
            arr = arr[:, :, None]
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if arr.shape[-1] != self.channels:
            raise ValueError(f"Channel count mismatch for {key}")
        # Diffusion/classification share [-1, 1] image range.
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float() / 127.5 - 1.0
        return x, int(label)
