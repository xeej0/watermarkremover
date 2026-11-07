"""Synthetic test data: textured 'photos' plus a semi-transparent text mark."""
import cv2
import numpy as np


def photo(seed: int, h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        a, b, d = rng.uniform(-1, 1, 3)
        img[..., c] = 120 + 60 * np.sin(a * xx / 40 + b * yy / 50 + d * 6)
    noise = cv2.GaussianBlur(rng.normal(0, 40, (h, w, 3)).astype(np.float32), (0, 0), 6)
    img += noise
    for _ in range(6):
        col = rng.uniform(0, 255, 3).tolist()
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(10, 50)), col, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def watermark_layer(h: int = 240, w: int = 320, text: str = "SAMPLE"):
    """Return (alpha HxW float in [0,1], colour BGR float)."""
    a = np.zeros((h, w), np.uint8)
    cv2.putText(a, text, (int(w * 0.18), int(h * 0.58)), cv2.FONT_HERSHEY_SIMPLEX, w / 220, 255, max(2, w // 110), cv2.LINE_AA)
    return a.astype(np.float32) / 255, np.array([255, 255, 255], np.float32)


def apply(img: np.ndarray, alpha: np.ndarray, colour, opacity: float = 0.45) -> np.ndarray:
    a = (alpha * opacity)[..., None]
    out = img.astype(np.float32) * (1 - a) + colour * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def psnr(a, b, mask=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if mask is not None:
        d = d[mask > 0]
    mse = d.mean()
    return 99.0 if mse == 0 else 10 * np.log10(255**2 / mse)
