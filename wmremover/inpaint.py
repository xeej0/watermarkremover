"""Inpainting backends.

``LamaInpainter`` runs LaMa (Suvorov et al., "Resolution-robust Large Mask
Inpainting with Fourier Convolutions", WACV 2022) through ONNX Runtime. LaMa's
fast Fourier convolutions give it an image-wide receptive field, so it
reconstructs repeating texture and structure behind a watermark far better
than diffusion-free classical methods, and it runs on CPU in well under a
second per region.

Full images are never squashed into the model's input size. Instead, each
cluster of masked pixels is cropped with surrounding context, inpainted at
model resolution, scaled back and composited into the original with a
feathered edge, so untouched pixels stay bit-identical and output resolution
is unlimited.

``ClassicalInpainter`` (OpenCV Telea) is the zero-download fallback.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_MODEL_URL = os.environ.get(
    "WMREMOVER_MODEL_URL",
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
)
DEFAULT_MODEL_SHA256 = os.environ.get("WMREMOVER_MODEL_SHA256")  # optional pin
CACHE_DIR = Path(os.environ.get("WMREMOVER_CACHE", Path.home() / ".cache" / "wmremover"))


# --------------------------------------------------------------------------
# region helpers
# --------------------------------------------------------------------------


def mask_regions(mask: np.ndarray, margin_frac: float = 0.6, min_margin: int = 48) -> List[Tuple[int, int, int, int]]:
    """Group masked pixels into crop boxes (x0, y0, x1, y1) with context.

    Nearby components are merged so each crop carries enough surroundings for
    the model to infer what belongs behind the mark.
    """
    H, W = mask.shape
    if not mask.any():
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        m = max(min_margin, int(margin_frac * max(w, h)))
        boxes.append([max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)])
    # merge overlapping boxes until stable
    merged = True
    while merged:
        merged = False
        out: List[List[int]] = []
        for b in boxes:
            for o in out:
                if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]:
                    o[:] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
                    merged = True
                    break
            else:
                out.append(list(b))
        boxes = out
    return [tuple(b) for b in boxes]  # type: ignore[misc]


def feather_composite(original: np.ndarray, filled: np.ndarray, mask: np.ndarray, feather: int = 3) -> np.ndarray:
    """Paste ``filled`` into ``original`` inside ``mask`` with a soft edge."""
    m = (mask > 0).astype(np.float32)
    if feather > 0:
        k = 2 * feather + 1
        grown = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m = np.maximum(m, cv2.GaussianBlur(grown, (k, k), 0) * grown)
    m = m[..., None]
    out = filled.astype(np.float32) * m + original.astype(np.float32) * (1 - m)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class _CropInpainter:
    """Shared crop / paste logic. Subclasses implement ``_inpaint_crop``."""

    name = "base"

    def _inpaint_crop(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, img: np.ndarray, mask: np.ndarray, feather: int = 3) -> np.ndarray:
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError("image and mask sizes differ")
        mask = (mask > 127).astype(np.uint8) * 255
        out = img.copy()
        for x0, y0, x1, y1 in mask_regions(mask):
            crop, cmask = out[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            filled = self._inpaint_crop(crop, cmask)
            out[y0:y1, x0:x1] = feather_composite(crop, filled, cmask, feather)
        return out


class ClassicalInpainter(_CropInpainter):
    """OpenCV fast-marching inpainting. Always available, no model needed."""

    name = "classical"

    def __init__(self, radius: int = 5):
        self.radius = radius

    def _inpaint_crop(self, img, mask):
        return cv2.inpaint(img, mask, self.radius, cv2.INPAINT_TELEA)


# --------------------------------------------------------------------------
# LaMa
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(url: str = DEFAULT_MODEL_URL, sha256: Optional[str] = DEFAULT_MODEL_SHA256) -> Path:
    """Download the LaMa ONNX weights once into the cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / Path(url.split("?")[0]).name
    if dest.exists() and (not sha256 or _sha256(dest) == sha256):
        return dest
    tmp = dest.with_suffix(".part")
    print(f"[wmremover] downloading LaMa model from {url}", file=sys.stderr)

    def hook(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
            print(f"\r[wmremover] {pct:3d}%", end="", file=sys.stderr)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print(file=sys.stderr)
    if sha256 and _sha256(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded model failed SHA-256 verification")
    tmp.replace(dest)
    return dest


class LamaInpainter(_CropInpainter):
    """LaMa via ONNX Runtime (CPU, CUDA, CoreML, DirectML: whatever is installed)."""

    name = "lama"

    def __init__(self, model_path: Optional[str] = None, providers: Optional[List[str]] = None, max_side: int = 1024):
        import onnxruntime as ort

        path = str(model_path or os.environ.get("WMREMOVER_MODEL") or ensure_model())
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, opts, providers=providers or ort.get_available_providers())
        inputs = self.session.get_inputs()
        if len(inputs) != 2:
            raise ValueError(f"expected a 2-input (image, mask) LaMa model, got {len(inputs)} inputs")
        img_in = next((i for i in inputs if "mask" not in i.name.lower()), inputs[0])
        mask_in = next(i for i in inputs if i is not img_in)
        self.img_name, self.mask_name = img_in.name, mask_in.name
        self.out_name = self.session.get_outputs()[0].name
        hw = img_in.shape[2:4]
        self.fixed = tuple(hw) if all(isinstance(d, int) and d > 0 for d in hw) else None
        self.max_side = max_side

    def _model_size(self, h: int, w: int) -> Tuple[int, int]:
        if self.fixed:
            return self.fixed  # type: ignore[return-value]
        s = min(1.0, self.max_side / max(h, w))
        return (max(8, int(round(h * s / 8)) * 8), max(8, int(round(w * s / 8)) * 8))

    def _inpaint_crop(self, img, mask):
        h, w = img.shape[:2]
        mh, mw = self._model_size(h, w)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (mw, mh), interpolation=cv2.INTER_AREA if mh * mw < h * w else cv2.INTER_CUBIC)
        m = cv2.resize(mask, (mw, mh), interpolation=cv2.INTER_NEAREST)
        m = (m > 127).astype(np.float32)
        x = x.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        x = x * (1 - m[None, None])  # LaMa expects the hole zeroed out
        y = self.session.run([self.out_name], {self.img_name: x, self.mask_name: m[None, None]})[0][0]
        y = y.transpose(1, 2, 0)
        if float(y.max()) <= 1.5:  # some exports emit [0, 1], others [0, 255]
            y = y * 255.0
        y = np.clip(y, 0, 255).astype(np.uint8)
        y = cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


_CACHE: dict = {}


def get_inpainter(backend: str = "auto", model_path: Optional[str] = None):
    """Return an inpainter. ``auto`` tries LaMa and falls back to classical."""
    key = (backend, model_path)
    if key in _CACHE:
        return _CACHE[key]
    if backend == "classical":
        inst = ClassicalInpainter()
    elif backend == "lama":
        inst = LamaInpainter(model_path)
    elif backend == "auto":
        try:
            inst = LamaInpainter(model_path)
        except Exception as exc:  # no network, no model, bad runtime...
            print(f"[wmremover] LaMa unavailable ({exc}); using classical inpainting", file=sys.stderr)
            inst = ClassicalInpainter()
    else:
        raise ValueError(f"unknown backend {backend!r}")
    _CACHE[key] = inst
    return inst
