"""Versioned, class-preserving augmentations; test and validation images never augmented."""
import hashlib
import cv2
import numpy as np

RECIPE_VERSION = "cv2-v1"


def variant_seed(dataset: str, split: str, source_index: int, variant_no: int, recipe: str = RECIPE_VERSION) -> int:
    raw = f"{dataset}:{split}:{source_index}:{variant_no}:{recipe}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def augment(image: np.ndarray, seed: int, *, dataset: str) -> np.ndarray:
    """Mild rotation, crop/translate, horizontal flip and illumination jitter.

    Cautious policy: fashion images aren't flipped to avoid shoe orientation bias.
    Intensity is clipped; output dtype/shape is identical to input.
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    rotation = float(rng.uniform(-10.0, 10.0))
    scale = float(rng.uniform(0.93, 1.03))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, scale)
    matrix[:, 2] += rng.uniform(-1.5, 1.5, size=2)
    out = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    if dataset == "cifar100" and rng.random() < 0.5:
        out = cv2.flip(out, 1)
    alpha = float(rng.uniform(0.90, 1.10))
    beta = float(rng.uniform(-7, 7))
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return out
