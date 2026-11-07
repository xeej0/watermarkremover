"""Command line interface.

Examples::

    wmremove photo.jpg                          # auto-detect, writes photo_clean.png
    wmremove photo.jpg --box 820,600,300,90     # look only inside a region
    wmremove photo.jpg --mask mask.png          # use your own mask
    wmremove shots/*.jpg --batch -o out/        # same mark on many images
    wmremove serve                              # web UI on http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .inpaint import get_inpainter
from .pipeline import remove_watermark, remove_watermark_batch


def read_image(path: str) -> np.ndarray:
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"error: cannot read image {path}")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, int(img.max())))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise SystemExit(f"error: cannot encode {path}")
    buf.tofile(str(path))


def _out_path(src: str, out: str | None, many: bool) -> Path:
    p = Path(src)
    if out is None:
        return p.with_name(f"{p.stem}_clean.png")
    o = Path(out)
    if many or o.is_dir() or out.endswith(("/", "\\")):
        return o / f"{p.stem}_clean.png"
    return o


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        return _serve(argv[1:])

    ap = argparse.ArgumentParser(prog="wmremove", description="Remove visible watermarks from images you own or are licensed to edit.")
    ap.add_argument("images", nargs="+", help="input image(s)")
    ap.add_argument("-o", "--output", help="output file, or directory for several inputs")
    ap.add_argument("--mask", help="mask image (white = remove); single input only")
    ap.add_argument("--box", help="x,y,w,h region that contains the watermark")
    ap.add_argument("--batch", action="store_true", help="images share one watermark: estimate it jointly and invert the blend")
    ap.add_argument("--backend", choices=["auto", "lama", "classical"], default="auto")
    ap.add_argument("--model", help="path to a LaMa ONNX model (default: download on first use)")
    ap.add_argument("--sensitivity", type=float, default=0.5, help="detection sensitivity 0..1 (default 0.5)")
    ap.add_argument("--dilate", type=int, default=3, help="grow the detected mask by N pixels (default 3)")
    ap.add_argument("--save-mask", action="store_true", help="also write the mask used")
    args = ap.parse_args(argv)

    many = len(args.images) > 1
    inpainter = get_inpainter(args.backend, args.model)
    print(f"[wmremover] inpainting backend: {inpainter.name}", file=sys.stderr)

    if args.batch:
        imgs = [read_image(p) for p in args.images]
        results, mask, _ = remove_watermark_batch(imgs, inpainter=inpainter)
        for src, res in zip(args.images, results):
            dst = _out_path(src, args.output, True)
            write_image(dst, res)
            print(dst)
        if args.save_mask:
            write_image(_out_path(args.images[0], args.output, True).with_name("batch_mask.png"), mask)
        return 0

    if args.mask and many:
        ap.error("--mask works with a single input image")
    box = tuple(int(v) for v in args.box.split(",")) if args.box else None
    if box is not None and len(box) != 4:
        ap.error("--box expects x,y,w,h")
    for src in args.images:
        img = read_image(src)
        mask = None
        if args.mask:
            mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise SystemExit(f"error: cannot read mask {args.mask}")
        res, used = remove_watermark(
            img, mask=mask, box=box, sensitivity=args.sensitivity, dilate=args.dilate, inpainter=inpainter
        )
        dst = _out_path(src, args.output, many)
        if not used.any():
            print(f"[wmremover] no watermark detected in {src}; try --box or --sensitivity", file=sys.stderr)
        write_image(dst, res)
        if args.save_mask:
            write_image(dst.with_name(dst.stem + "_mask.png"), used)
        print(dst)
    return 0


def _serve(argv) -> int:
    ap = argparse.ArgumentParser(prog="wmremove serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--backend", choices=["auto", "lama", "classical"], default="auto")
    args = ap.parse_args(argv)
    try:
        import uvicorn
    except ImportError:
        raise SystemExit('the web UI needs extras: pip install "wmremover[server]"')
    import os

    os.environ["WMREMOVER_BACKEND"] = args.backend
    print(f"[wmremover] open http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run("wmremover.server:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
