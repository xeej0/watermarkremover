# Unmark: AI watermark remover

Detects and removes **visible** watermarks (text, logos, stamps) from images,
using a learned inpainting model plus a multi-image matting solver that
recovers the real pixels under a mark instead of guessing them.

Intended for images you own or are licensed to edit: cleaning up your own
photos and renders, removing preview marks from stock you have paid for,
restoring scans, and so on. It does not attempt to strip invisible
provenance signals (C2PA metadata, AI-content watermarks such as SynthID).

Two ways to use it:

- **Unmark Studio** (`web/`): a static website. Everything, including the AI
  model, runs in the visitor's browser, so images are never uploaded and
  hosting is free. Deploys to Vercel as-is.
- **wmremover** (`wmremover/`): a Python package with a CLI, a Python API and
  a small FastAPI server, for scripting and bulk jobs.

## Unmark Studio (web app)

Features: auto-detect, box-detect (drag around a mark to select only its
strokes), brush and eraser with undo/redo, zoom and pan (wheel, pinch, space
drag), AI inpainting with LaMa (WebGPU, or multi-threaded WASM on CPU),
instant PatchMatch fill with no download, before/after split view, PNG/JPEG
export, and a batch mode that cleans a whole set sharing one watermark and
exports a ZIP.

The LaMa model (208 MB) downloads from Hugging Face on first use and is kept
in the browser's Cache Storage, so later visits start instantly. If it cannot
be downloaded, PatchMatch is used automatically, or a local `.onnx` file can
be picked.

### Deploy to Vercel

`vercel.json` at the repository root serves `web/` with the right headers
(cross-origin isolation for multi-threaded WASM). There is no build step.

- **Dashboard:** vercel.com > Add New > Project > import this repository >
  Deploy. Leave the framework preset at "Other".
- **CLI:** `cd web && npx vercel deploy --prod`

### Run locally

```bash
cd web
python3 -m http.server 8080      # then open http://localhost:8080
node --test tests/engine.test.cjs
```

| File | Role |
|---|---|
| `web/index.html` | Layout and styles |
| `web/app.js` | Canvas editing, history, AI model loading and inference, batch UI, ZIP export |
| `web/engine-worker.js` | Web Worker: stroke detector, PatchMatch, push-pull fill, batch matting |

## How it works

| Stage | Method |
|---|---|
| Inpainting | **LaMa** (Suvorov et al., WACV 2022) via ONNX Runtime. Fourier convolutions give an image-wide receptive field, so it rebuilds textures and structure behind large marks. |
| Resolution | Crop-and-paste: each cluster of masked pixels is cropped with context, inpainted at model resolution and feather-composited back. Untouched pixels stay bit-identical and there is no output size limit. |
| Single-image detection | Multi-scale morphological top-hat / black-hat stroke filter, then region scoring: stroke density, colour consistency, a consistent light-or-dark polarity, and desaturation relative to the background. Hysteresis and hole filling recover faint parts of letters. Optional box hint. |
| Batch detection | Median-gradient watermark estimation (Dekel et al., *On the Effectiveness of Visible Watermarks*, CVPR 2017): with 3+ images carrying the same mark, varying content cancels and the mark's outline remains. |
| Batch removal | Per-pixel least-squares alpha matte across all images for the blend `I = a*W + (1-a)*J`, with a flat-colour refinement, then exact inversion `J = (I - a*W) / (1 - a)`. Nearly opaque pixels fall back to inpainting. |
| Fallback | PatchMatch (browser) or OpenCV Telea (Python) when the model cannot be loaded. |

Synthetic benchmark (`tests/`), PSNR inside the watermark, 8 images with a 45% white text mark:

| | PSNR |
|---|---|
| Watermarked | ~13 dB |
| Inpainting only (classical) | ~24 dB |
| Batch matting | **~38 dB** |

## Python package

```bash
pip install -e ".[server]"
```

The LaMa weights download on first use into `~/.cache/wmremover`. Override
with `WMREMOVER_MODEL=/path/to/lama.onnx` or `WMREMOVER_MODEL_URL=...` (and
optionally pin `WMREMOVER_MODEL_SHA256`). Any 2-input (image, mask) LaMa ONNX
export works, fixed or dynamic size.

```bash
wmremove photo.jpg                         # auto-detect -> photo_clean.png
wmremove photo.jpg --box 820,600,300,90    # only look inside a region
wmremove photo.jpg --mask mask.png         # your own mask (white = remove)
wmremove a.jpg b.jpg c.jpg --batch -o out/ # shared watermark, matting mode
wmremove photo.jpg --backend classical     # no model download
wmremove serve                             # simple web UI on :7860
```

Other flags: `--sensitivity 0..1`, `--dilate N`, `--save-mask`, `--model PATH`.

```python
import cv2
from wmremover import remove_watermark, remove_watermark_batch

img = cv2.imread("photo.jpg")
clean, mask = remove_watermark(img)                  # or mask=..., box=(x, y, w, h)
cleans, mask, alpha = remove_watermark_batch([cv2.imread(p) for p in paths])
```

## Tips

- Auto-detect is a starting point. For busy photos, drag a box around the
  mark or paint it; the brush is always the most reliable mask.
- Grow the mask slightly if a faint halo or drop shadow remains.
- For sets with the same mark (stock previews, a photographer's portfolio),
  batch mode is far more faithful than any inpainting: it restores the
  original pixels.

## Tests

```bash
pip install -e ".[server,dev]"
pytest -q
node --test web/tests/engine.test.cjs
```

Tests run offline: the LaMa wrapper is exercised with a tiny generated ONNX
graph that mimics the model's inputs and outputs.
