from unittest.mock import patch
import cv2
import numpy as np
from sqlalchemy import select, func
from visionforge.database import ImageAsset, Augmentation
from visionforge.images import assess, encode_png, decode_image, hamming, perceptual_hash
from visionforge.pipeline import ingest, clean, deduplicate, materialize, inspect
from visionforge.loader import IndexedImages, val_partition
from visionforge.augmentation import augment, variant_seed


def toy_dataset(name, root, train):
    rng = np.random.default_rng(14 if train else 141)
    size = 40 if train else 12
    if name == "fashionmnist":
        images = rng.integers(5, 255, (size, 28, 28), dtype=np.uint8)
        labels = [i % 10 for i in range(size)]
    else:
        images = rng.integers(5, 255, (size, 32, 32, 3), dtype=np.uint8)
        labels = [i % 100 for i in range(size)]
    if train and name == "fashionmnist":
        images[1] = images[0]
        labels[1] = labels[0]
    return images, labels


def test_hash_and_augmentation():
    img = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    data = encode_png(img)
    assert decode_image(data).shape == (32, 32)
    assert assess(data)["width"] == 32
    assert hamming(perceptual_hash(img), perceptual_hash(img)) == 0
    s = variant_seed("fashionmnist", "train", 1, 3)
    assert np.array_equal(augment(img, s, dataset="fashionmnist"), augment(img, s, dataset="fashionmnist"))


def test_offline_end_to_end_ingest_quality_dedup_materialize(rig, tmp_path):
    cfg, store, session = rig
    with patch("visionforge.pipeline.load_official_dataset", side_effect=toy_dataset):
        outcome = ingest("fashionmnist", storage=store, session=session, config=cfg, dataset_dir=str(tmp_path))
        assert outcome["uploaded"] == 52
        repeated = ingest("fashionmnist", storage=store, session=session, config=cfg, dataset_dir=str(tmp_path))
        assert repeated["skipped_existing"] == 52
    quality = clean(storage=store, session=session, dataset="fashionmnist")
    assert quality["checked"] == 52
    d = deduplicate(session=session, dataset="fashionmnist")
    assert d["exact_duplicates"] == 1
    assert session.scalar(select(func.count()).select_from(ImageAsset).where(
        ImageAsset.status == "duplicate")) == 1
    a = materialize(storage=store, session=session, config=cfg, dataset="fashionmnist", variants=2)
    assert a["uploaded"] == 39 * 2
    again = materialize(storage=store, session=session, config=cfg, dataset="fashionmnist", variants=2)
    assert again["uploaded"] == 0
    assert session.scalar(select(func.count()).select_from(Augmentation)) == 78
    train = IndexedImages(session=session, storage=store, cache_dir=cfg.cache_dir,
                          dataset="fashionmnist", partition="train", mode="baseline")
    augmented = IndexedImages(session=session, storage=store, cache_dir=cfg.cache_dir,
                              dataset="fashionmnist", partition="train", mode="augmented")
    val = IndexedImages(session=session, storage=store, cache_dir=cfg.cache_dir,
                        dataset="fashionmnist", partition="val", mode="augmented")
    test = IndexedImages(session=session, storage=store, cache_dir=cfg.cache_dir,
                         dataset="fashionmnist", partition="test", mode="augmented")
    assert len(augmented) == len(train) * 3
    assert len(val) > 0 and len(test) == 12
    assert len(augmented) + len(val) * 3 == 39 * 3
    assert train[0][0].shape == (1, 28, 28)
    assert inspect(session, "fashionmnist")["materialized_variants"]["fashionmnist"] == 78


def test_index_existing_s3_manifest_without_reupload(rig, tmp_path):
    from visionforge.pipeline import import_s3_manifest
    cfg, store, session = rig
    rng = np.random.default_rng(101)
    array = rng.integers(4, 251, (32, 32, 3), dtype=np.uint8)
    key = "preloaded/cifar100/train/012345.png"
    store.put(key, encode_png(array))
    csv_file = tmp_path / "manifest.csv"
    csv_file.write_text("dataset,split,source_index,label,key\ncifar100,train,12345,24," + key + "\n")
    result = import_s3_manifest(storage=store, session=session, manifest_path=str(csv_file))
    assert result["indexed"] == 1
    row = session.scalar(select(ImageAsset))
    assert (row.dataset, row.split, row.source_index, row.label, row.key) == (
        "cifar100", "train", 12345, 24, key)
    assert import_s3_manifest(storage=store, session=session, manifest_path=str(csv_file))["skipped_existing"] == 1


def test_cifar100_rgb_ingestion_and_three_channel_training_input(rig):
    cfg, store, session = rig
    with patch("visionforge.pipeline.load_official_dataset", side_effect=toy_dataset):
        outcome = ingest("cifar100", storage=store, session=session, config=cfg, limit=30)
    assert outcome["uploaded"] == 42  # 30 training + 12 official-like test fixture images
    deduplicate(session=session, dataset="cifar100")
    out = materialize(storage=store, session=session, config=cfg, dataset="cifar100", variants=1, limit=8)
    assert out["uploaded"] == 8
    train = IndexedImages(session=session, storage=store, cache_dir=cfg.cache_dir,
                          dataset="cifar100", partition="train", mode="augmented")
    assert train[0][0].shape == (3, 32, 32)
    assert train[0][0].min() >= -1 and train[0][0].max() <= 1
