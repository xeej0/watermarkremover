"""FastAPI backend for the browser UI (``wmremove serve``)."""

from __future__ import annotations

import base64
import io
import os
import zipfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from .detect import detect_watermark
from .inpaint import get_inpainter
from .pipeline import remove_watermark, remove_watermark_batch

MAX_BYTES = int(os.environ.get("WMREMOVER_MAX_BYTES", 40 * 1024 * 1024))
MAX_PIXELS = int(os.environ.get("WMREMOVER_MAX_PIXELS", 60_000_000))
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="wmremover", docs_url="/api/docs")


def _inpainter():
    return get_inpainter(os.environ.get("WMREMOVER_BACKEND", "auto"))


async def _decode(upload: UploadFile, gray: bool = False) -> np.ndarray:
    data = await upload.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"cannot decode {upload.filename!r} as an image")
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise HTTPException(413, "image has too many pixels")
    return img


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "encoding failed")
    return buf.tobytes()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {"backend": _inpainter().name}


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    box: Optional[str] = Form(None),
    sensitivity: float = Form(0.5),
):
    img = await _decode(image)
    b = None
    if box:
        try:
            b = tuple(int(float(v)) for v in box.split(","))
            assert len(b) == 4
        except Exception:
            raise HTTPException(400, "box must be x,y,w,h")
    try:
        mask = detect_watermark(img, box=b, sensitivity=sensitivity)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return JSONResponse(
        {"mask": "data:image/png;base64," + base64.b64encode(_png(mask)).decode(), "found": bool(mask.any())}
    )


@app.post("/api/remove")
async def remove(image: UploadFile = File(...), mask: UploadFile = File(...)):
    img = await _decode(image)
    m = await _decode(mask, gray=True)
    res, _ = remove_watermark(img, mask=m, inpainter=_inpainter())
    return Response(_png(res), media_type="image/png")


@app.post("/api/batch")
async def batch(images: List[UploadFile] = File(...)):
    imgs = [await _decode(u) for u in images]
    try:
        results, mask, _ = remove_watermark_batch(imgs, inpainter=_inpainter())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for u, r in zip(images, results):
            zf.writestr(f"{Path(u.filename or 'image').stem}_clean.png", _png(r))
        zf.writestr("watermark_mask.png", _png(mask))
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="wmremover_batch.zip"'},
    )
