"""End-to-end removal pipelines."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detect import Box, detect_watermark, estimate_batch_watermark
from .inpaint import get_inpainter


def remove_watermark(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    box: Optional[Box] = None,
    backend: str = "auto",
    sensitivity: float = 0.5,
    dilate: int = 3,
    inpainter=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove a visible watermark from one BGR uint8 image.

    Mask priority: explicit ``mask`` > detection inside ``box`` > automatic
    full-image detection. Returns ``(result, mask_used)``.
    """
    if mask is None:
        mask = detect_watermark(img, box=box, sensitivity=sensitivity, dilate=dilate)
    elif mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 127).astype(np.uint8) * 255
    if not mask.any():
        return img.copy(), mask
    inpainter = inpainter or get_inpainter(backend)
    return inpainter(img, mask), mask


def _solve_alpha_matte(images: Sequence[np.ndarray], priors: Sequence[np.ndarray], mask: np.ndarray):
    """Per-pixel least squares for the blend I = a*W + (1-a)*J.

    Rewritten as I - J = u - a*J with u = a*W, which is linear in (u, a).
    Each image contributes 3 equations per pixel (one per channel), so with
    N >= 2 diverse backgrounds the 4 unknowns are over-determined. ``priors``
    are content estimates J (from inpainting); they are only used to pin
    down the matte, the final pixels come from the real observations.
    """
    ys, xs = np.nonzero(mask)
    I = np.stack([im[ys, xs].astype(np.float64) / 255 for im in images])  # N,P,3
    J = np.stack([p[ys, xs].astype(np.float64) / 255 for p in priors])
    N, P, _ = I.shape
    D = I - J
    AtA = np.zeros((P, 4, 4))
    AtA[:, [0, 1, 2], [0, 1, 2]] = N
    cross = -J.sum(axis=0)  # P,3
    AtA[:, :3, 3] = cross
    AtA[:, 3, :3] = cross
    AtA[:, 3, 3] = (J**2).sum(axis=(0, 2))
    AtA += np.eye(4) * 1e-4
    Atb = np.zeros((P, 4))
    Atb[:, :3] = D.sum(axis=0)
    Atb[:, 3] = -(J * D).sum(axis=(0, 2))
    sol = np.linalg.solve(AtA, Atb[..., None])[..., 0]
    u, a = sol[:, :3], sol[:, 3]

    # Most watermarks are a single flat colour (white, black or grey). If the
    # free per-pixel solution agrees on one colour, re-solve with W fixed:
    # one unknown per pixel instead of four is far more robust to errors in
    # the content priors.
    strong = a > 0.15
    if strong.sum() >= 20:
        w_px = u[strong] / a[strong, None]
        W = np.median(w_px, axis=0)
        spread = np.median(np.abs(w_px - W), axis=0).mean()
        if spread < 0.12:
            W = np.clip(W, 0, 1)
            G = W[None, None, :] - J  # N,P,3
            a = (D * G).sum(axis=(0, 2)) / ((G * G).sum(axis=(0, 2)) + 1e-6)
            a = np.clip(a, 0, 0.99)
            u = a[:, None] * W[None, :]
    return ys, xs, u, a


def remove_watermark_batch(
    images: Sequence[np.ndarray],
    backend: str = "auto",
    max_alpha: float = 0.85,
    inpainter=None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Remove a watermark shared by several same-sized images.

    1. Localise the mark from the median gradient (Dekel et al. 2017).
    2. Inpaint each image to get a content prior J_n.
    3. Solve a per-pixel alpha matte (a, u = a*W) jointly across all images.
    4. Invert the blend: J_n = (I_n - u) / (1 - a). This recovers the real
       pixels under the mark instead of hallucinating them.
    5. Where the mark is nearly opaque (a > max_alpha) inversion is ill-posed;
       use the inpainted prior there, with a smooth transition.

    Returns ``(results, mask, alpha_map)``.
    """
    est = estimate_batch_watermark(images)
    mask = est.mask
    H, W = mask.shape
    inpainter = inpainter or get_inpainter(backend)
    priors = [inpainter(im, mask) for im in images]
    alpha_map = np.zeros((H, W), np.float32)
    if not mask.any():
        return [im.copy() for im in images], mask, alpha_map

    ys, xs, u, a = _solve_alpha_matte(images, priors, mask)
    # No spatial smoothing: anti-aliased stroke edges change alpha within a
    # pixel or two, and blurring the matte there costs >10 dB PSNR.
    a_img = np.zeros((H, W), np.float32)
    a_img[ys, xs] = np.clip(a, 0, 0.99)
    u_img = np.zeros((H, W, 3), np.float32)
    u_img[ys, xs] = np.clip(u, 0, 1)
    m = (mask > 0).astype(np.float32)
    alpha_map = a_img

    # Blend weight between inversion (reliable at low alpha) and the prior.
    lo, hi = max_alpha - 0.15, max_alpha
    t = np.clip((a_img - lo) / (hi - lo), 0, 1)
    w_prior = (t * t * (3 - 2 * t))[..., None]
    denom = np.maximum(1 - a_img, 1 - max_alpha)[..., None]

    results = []
    for im, prior in zip(images, priors):
        f = im.astype(np.float32) / 255
        inv = np.clip((f - u_img) / denom, 0, 1)
        out = inv * (1 - w_prior) + prior.astype(np.float32) / 255 * w_prior
        out = np.where(m[..., None] > 0, out, f)
        results.append(np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8))
    return results, mask, alpha_map
