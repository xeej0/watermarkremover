"""Watermark localisation.

Two strategies are provided:

* ``detect_watermark``: single-image detection. Visible watermarks are thin,
  low-saturation, semi-transparent strokes (text or logo outlines) laid on top
  of the photo. We look for stroke-like structures with morphological top-hat
  and black-hat filters, group them into candidate regions and keep regions
  whose strokes look like an overlay (consistent polarity and colour, lower
  saturation than the background, high stroke density). Works well for
  typical text/logo marks; always allow the user to refine the result.

* ``estimate_batch_watermark``: multi-image estimation following Dekel et al.,
  "On the Effectiveness of Visible Watermarks" (CVPR 2017). When the same
  watermark is stamped at the same place on many images, the median of the
  image gradients cancels the varying content and leaves the watermark's
  gradient. This yields a very precise mask and enables matting-based
  removal (inverting the alpha blend) instead of hallucinating content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h

WORK_SIDE = 720  # detection runs at most at this short-side resolution


def _odd(n: float) -> int:
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def _clip_box(box: Box, shape) -> Box:
    h, w = shape[:2]
    x, y, bw, bh = (int(v) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"box {box} lies outside the image ({w}x{h})")
    return x0, y0, x1 - x0, y1 - y0


def stroke_response(img: np.ndarray, return_extras: bool = False):
    """Per-pixel score in [0, 1] for 'thin overlay stroke' structures.

    Multi-scale morphological top-hat / black-hat: a stroke of width s lights
    up once the structuring element is wider than s, so a small bank of
    kernels covers hairline text up to bold logos.

    With ``return_extras`` also returns (polarity, saturation, kmax) where
    polarity is +1 for lighter-than-surroundings strokes and -1 for darker.
    """
    H, W = img.shape[:2]
    scale = min(1.0, WORK_SIDE / min(H, W))
    work = cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    kmax = max(9, min(h, w) / 18)
    ks = sorted({_odd(k) for k in np.geomspace(7, kmax, 5)})
    resp = np.zeros((h, w), np.float32)
    polarity = np.zeros((h, w), np.int8)
    for k in ks:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        v = np.maximum(th, bh)
        upd = v > resp
        resp[upd] = v[upd]
        polarity[upd] = np.where(th[upd] >= bh[upd], 1, -1)

    # Overlays are usually white/grey/black: down-weight saturated pixels.
    sat = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32) / 255.0
    resp *= 1.0 - 0.7 * sat

    # Suppress dense natural texture (foliage, gravel): there the response is
    # high everywhere, whereas watermark strokes stand out from a calmer
    # neighbourhood.
    big = _odd(kmax * 3)
    local = cv2.blur(resp, (big, big))
    resp = np.clip(resp - 0.6 * local, 0, None)
    # Normalise against a bounded contrast scale (grey levels) so one very
    # high-contrast edge (a skyline) cannot drown out the watermark.
    m = float(np.clip(np.percentile(resp, 99.7), 20, 45))
    resp = np.clip(resp / m, 0, 1)
    if scale < 1:
        resp = cv2.resize(resp, (W, H), interpolation=cv2.INTER_LINEAR)
        polarity = cv2.resize(polarity, (W, H), interpolation=cv2.INTER_NEAREST)
        sat = cv2.resize(sat, (W, H), interpolation=cv2.INTER_LINEAR)
    if return_extras:
        return resp, polarity, sat, kmax / scale
    return resp


def detect_watermark(
    img: np.ndarray,
    box: Optional[Box] = None,
    sensitivity: float = 0.5,
    dilate: int = 3,
) -> np.ndarray:
    """Return a uint8 mask (255 = watermark) for a BGR uint8 image.

    ``box`` restricts detection to a region (x, y, w, h). If nothing is found
    inside the box the whole box is returned, so a box always yields a usable
    mask. ``sensitivity`` in [0, 1]: higher keeps fainter strokes.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected a BGR image of shape (H, W, 3)")
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    H, W = img.shape[:2]
    full = np.zeros((H, W), np.uint8)

    if box is not None:
        x, y, bw, bh = _clip_box(box, img.shape)
        pad = max(8, int(0.15 * max(bw, bh)))
        cx0, cy0 = max(0, x - pad), max(0, y - pad)
        cx1, cy1 = min(W, x + bw + pad), min(H, y + bh + pad)
        sub = img[cy0:cy1, cx0:cx1]
        resp = stroke_response(sub)
        thr = max(0.08, np.percentile(resp, 100 - (8 + 30 * sensitivity)))
        strokes = (resp > thr).astype(np.uint8) * 255
        inner = np.zeros_like(strokes)
        inner[y - cy0 : y - cy0 + bh, x - cx0 : x - cx0 + bw] = 255
        strokes &= inner
        if cv2.countNonZero(strokes) < 0.01 * bw * bh:
            full[y : y + bh, x : x + bw] = 255
            return full
        full[cy0:cy1, cx0:cx1] = strokes
        return _finish_mask(full, dilate)

    resp, polarity, sat, kmax = stroke_response(img, return_extras=True)
    thr = 0.55 - 0.35 * sensitivity
    strokes = (resp > thr).astype(np.uint8) * 255
    if not strokes.any():
        return full

    # Group nearby strokes (letters of a word, parts of a logo) into regions.
    gk = _odd(max(15, min(H, W) / 18))
    regions = cv2.morphologyEx(
        strokes, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gk, gk))
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(regions, connectivity=8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Overlay cues: strokes of one mark share a polarity (all lighter or all
    # darker), and a white/grey/black overlay lowers saturation vs. surroundings.
    not_stroke = (strokes == 0).astype(np.float32)
    bk = _odd(kmax * 2)
    sat_bg = cv2.blur(sat * not_stroke, (bk, bk)) / np.maximum(cv2.blur(not_stroke, (bk, bk)), 1e-3)
    region_pol = np.zeros(n)
    min_area = 0.0008 * H * W
    max_area = 0.6 * H * W
    keep = np.zeros(n, bool)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue
        region = labels == i
        s = strokes.astype(bool) & region
        cnt = int(s.sum())
        if cnt < 30:
            continue
        density = cnt / area
        colour_std = float(lab[s].std(axis=0)[1:].mean())  # chroma spread
        strength = float(resp[s].mean())
        pol_sum = float(polarity[s].sum())
        consistency = abs(pol_sum) / cnt
        dsat = float(np.clip((sat_bg[s] - sat[s]).mean(), -0.2, 0.2))
        region_pol[i] = np.sign(pol_sum)
        score = strength * 1.5 + min(density, 0.5) - colour_std / 40.0 + 0.8 * (consistency - 0.6) + 2 * dsat
        if score > 0.75 - 0.3 * sensitivity:
            keep[i] = True
    if not keep[1:].any():
        return full
    # Hysteresis: inside kept regions also take fainter pixels of the region's
    # polarity, recovering letters that cross bright or busy backgrounds.
    in_kept = keep[labels]
    faint = (resp > thr * 0.45) & (polarity == region_pol[labels])
    full = (in_kept & (strokes.astype(bool) | faint)).astype(np.uint8) * 255
    full = fill_holes(full, max_hole_frac=0.003)
    return _finish_mask(full, dilate)


def fill_holes(mask: np.ndarray, max_hole_frac: float = 0.002) -> np.ndarray:
    """Fill small enclosed holes (letter counters) in a binary mask."""
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = np.zeros((h + 4, w + 4), np.uint8)
    outside = padded.copy()
    cv2.floodFill(outside, ff, (0, 0), 255)
    holes = cv2.bitwise_not(outside)[1:-1, 1:-1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    big = np.where(stats[:, cv2.CC_STAT_AREA] > max_hole_frac * h * w)[0]
    holes[np.isin(labels, big[big > 0])] = 0
    return mask | holes


def _finish_mask(mask: np.ndarray, dilate: int) -> np.ndarray:
    # Remove specks, then grow to cover anti-aliased halos and drop shadows.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < 4)[0]
    if len(small) > 1:
        mask[np.isin(labels, small[small > 0])] = 0
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask, k)
    return mask


@dataclass
class BatchWatermark:
    """Result of multi-image watermark estimation."""

    mask: np.ndarray  # uint8, 255 where the watermark lives
    grad_x: np.ndarray  # median gradient of the watermark, float32 HxWx3
    grad_y: np.ndarray


def estimate_batch_watermark(
    images: Sequence[np.ndarray], dilate: int = 2, k_sigma: float = 3.0
) -> BatchWatermark:
    """Locate a watermark shared by many same-sized images (Dekel et al.)."""
    if len(images) < 3:
        raise ValueError("batch estimation needs at least 3 images")
    shape = images[0].shape
    if any(im.shape != shape for im in images):
        raise ValueError("batch estimation needs images of identical size")

    gxs, gys = [], []
    for im in images:
        f = im.astype(np.float32) / 255.0
        gxs.append(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3))
        gys.append(cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))
    gx = np.median(np.stack(gxs), axis=0)
    gy = np.median(np.stack(gys), axis=0)
    del gxs, gys
    mag = np.sqrt((gx**2 + gy**2).sum(axis=2))

    # Robust threshold: background median-gradient is ~noise.
    med = float(np.median(mag))
    mad = float(np.median(np.abs(mag - med))) * 1.4826 + 1e-6
    edges = (mag > med + k_sigma * 3 * mad).astype(np.uint8) * 255
    h, w = edges.shape
    # Fill stroke interiors: edges outline the glyphs, close them.
    ck = _odd(max(5, min(h, w) / 100))
    mask = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(6, int(0.00005 * h * w))
    drop = np.where(stats[:, cv2.CC_STAT_AREA] < min_area)[0]
    mask[np.isin(labels, drop[drop > 0])] = 0
    # Fill holes inside letters (e.g. "o", "a").
    mask = fill_holes(mask)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask, k)
    return BatchWatermark(mask=mask, grad_x=gx, grad_y=gy)
