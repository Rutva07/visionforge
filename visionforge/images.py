"""Image canonicalization, quality assessment and conservative perceptual hash."""
import hashlib
import cv2
import numpy as np


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("OpenCV failed to encode image")
    return encoded.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None or arr.size == 0:
        raise ValueError("Invalid/corrupted image data")
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (1, 3)):
        raise ValueError("Unsupported image shape")
    return arr


def perceptual_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    square = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(square)[:8, :8]
    med = np.median(dct.flatten()[1:])
    bits = (dct > med).flatten()
    number = sum(int(bit) << i for i, bit in enumerate(bits))
    return f"{number:016x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def assess(data: bytes) -> dict:
    img = decode_image(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    pixel_std = float(gray.std())
    reason = None
    if min(w, h) < 16:
        reason = "too_small"
    elif pixel_std < 2.0:
        reason = "near_constant"
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "phash": perceptual_hash(img), "width": w, "height": h,
        "channels": 1 if img.ndim == 2 else int(img.shape[2]),
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "pixel_std": pixel_std, "status": "quarantined" if reason else "ready", "reason": reason,
    }
