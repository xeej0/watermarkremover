import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth import apply, photo, psnr, watermark_layer  # noqa: E402

from wmremover import (  # noqa: E402
    ClassicalInpainter,
    LamaInpainter,
    detect_watermark,
    estimate_batch_watermark,
    remove_watermark,
    remove_watermark_batch,
)
from wmremover.inpaint import mask_regions  # noqa: E402

ALPHA, COLOUR = watermark_layer()
GT = ALPHA > 0.05


@pytest.fixture(scope="module")
def classical():
    return ClassicalInpainter()


def test_detect_finds_text_watermark():
    recalls, fps = [], []
    for s in range(5):
        wm = apply(photo(s), ALPHA, COLOUR)
        m = detect_watermark(wm) > 0
        recalls.append(m[GT].mean())
        fps.append(m[~GT].mean())
    assert np.mean(recalls) > 0.85
    assert np.mean(fps) < 0.06


def test_detect_clean_image_is_mostly_empty():
    m = detect_watermark(photo(42))
    assert (m > 0).mean() < 0.03


def test_detect_scales_to_large_images():
    h, w = 960, 1280
    alpha, colour = watermark_layer(h, w)
    gt = alpha > 0.05
    m = detect_watermark(apply(photo(1, h, w), alpha, colour)) > 0
    assert m[gt].mean() > 0.85


def test_box_restricts_detection_and_falls_back_to_box():
    wm = apply(photo(1), ALPHA, COLOUR)
    ys, xs = np.nonzero(GT)
    box = (xs.min() - 5, ys.min() - 5, xs.max() - xs.min() + 10, ys.max() - ys.min() + 10)
    m = detect_watermark(wm, box=box, dilate=0) > 0
    assert m[GT].mean() > 0.7
    outside = np.ones_like(m)
    outside[box[1] : box[1] + box[3], box[0] : box[0] + box[2]] = False
    assert not m[outside].any()
    # flat region: nothing to find, so the whole box is returned
    flat = np.full((100, 100, 3), 128, np.uint8)
    assert (detect_watermark(flat, box=(10, 10, 20, 20)) > 0).sum() == 400


def test_single_removal_improves_psnr(classical):
    clean = photo(3)
    wm = apply(clean, ALPHA, COLOUR)
    res, mask = remove_watermark(wm, inpainter=classical)
    assert psnr(res, clean, GT) > psnr(wm, clean, GT) + 5
    # pixels outside the (feathered) mask are untouched
    far = cv2.dilate(mask, np.ones((15, 15), np.uint8)) == 0
    assert np.array_equal(res[far], wm[far])


def test_batch_matting_recovers_true_pixels(classical):
    cleans = [photo(100 + i) for i in range(8)]
    wms = [apply(c, ALPHA, COLOUR) for c in cleans]
    est = estimate_batch_watermark(wms)
    assert (est.mask[GT] > 0).mean() > 0.98
    results, mask, alpha = remove_watermark_batch(wms, inpainter=classical)
    assert abs(float(alpha[ALPHA > 0.95].mean()) - 0.45) < 0.05
    for c, w, r in zip(cleans, wms, results):
        inpaint_only = classical(w, mask)
        assert psnr(r, c, GT) > 32
        assert psnr(r, c, GT) > psnr(inpaint_only, c, GT) - 0.5


def test_batch_rejects_bad_input():
    with pytest.raises(ValueError):
        estimate_batch_watermark([photo(0), photo(1)])
    with pytest.raises(ValueError):
        estimate_batch_watermark([photo(0), photo(1), photo(2, h=100, w=100)])


def test_mask_regions_merges_nearby_components():
    m = np.zeros((400, 400), np.uint8)
    m[50:60, 50:60] = 255
    m[50:60, 75:85] = 255  # close: merged
    m[350:360, 350:360] = 255  # far: separate
    assert len(mask_regions(m)) == 2


def _dummy_lama(path: Path, fixed: bool):
    """ONNX graph mimicking LaMa's I/O: out = (image + mask*0.5) * 255."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    hw = [64, 64] if fixed else ["H", "W"]
    img = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, *hw])
    msk = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, *hw])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, *hw])
    half = helper.make_tensor("half", TensorProto.FLOAT, [], [0.5])
    k = helper.make_tensor("k", TensorProto.FLOAT, [], [255.0])
    nodes = [
        helper.make_node("Mul", ["mask", "half"], ["m2"]),
        helper.make_node("Add", ["image", "m2"], ["s"]),
        helper.make_node("Mul", ["s", "k"], ["output"]),
    ]
    g = helper.make_graph(nodes, "dummy", [img, msk], [out], initializer=[half, k])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


@pytest.mark.parametrize("fixed", [True, False])
def test_lama_wrapper_plumbing(tmp_path, fixed):
    p = tmp_path / "dummy.onnx"
    _dummy_lama(p, fixed)
    lama = LamaInpainter(str(p), providers=["CPUExecutionProvider"])
    assert (lama.fixed == (64, 64)) == fixed
    img = np.full((200, 300, 3), 40, np.uint8)
    mask = np.zeros((200, 300), np.uint8)
    mask[90:110, 140:160] = 255
    out = lama(img, mask, feather=0)
    # hole is zeroed then +0.5 => ~128 inside, untouched outside
    assert abs(int(out[100, 150, 0]) - 128) <= 3
    assert np.array_equal(out[mask == 0], img[mask == 0])


def test_cli_end_to_end(tmp_path):
    from wmremover.cli import main

    clean = photo(7)
    src = tmp_path / "in.jpg"
    cv2.imwrite(str(src), apply(clean, ALPHA, COLOUR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    out = tmp_path / "out.png"
    assert main([str(src), "-o", str(out), "--backend", "classical", "--save-mask"]) == 0
    res = cv2.imread(str(out))
    assert res.shape == clean.shape
    assert (tmp_path / "out_mask.png").exists()


def test_server_endpoints():
    pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    import os

    os.environ["WMREMOVER_BACKEND"] = "classical"
    from fastapi.testclient import TestClient

    from wmremover.server import app

    c = TestClient(app)
    assert c.get("/api/info").json()["backend"] == "classical"
    assert "Watermark Remover" in c.get("/").text
    wm = apply(photo(2), ALPHA, COLOUR)
    png = cv2.imencode(".png", wm)[1].tobytes()
    r = c.post("/api/detect", files={"image": ("a.png", png, "image/png")}, data={"sensitivity": "0.5"})
    assert r.status_code == 200 and r.json()["found"]
    mask = cv2.imencode(".png", (GT * 255).astype(np.uint8))[1].tobytes()
    r = c.post("/api/remove", files={"image": ("a.png", png, "image/png"), "mask": ("m.png", mask, "image/png")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    files = [("images", (f"{i}.png", cv2.imencode(".png", apply(photo(50 + i), ALPHA, COLOUR))[1].tobytes(), "image/png")) for i in range(4)]
    r = c.post("/api/batch", files=files)
    assert r.status_code == 200 and r.content[:2] == b"PK"
    assert c.post("/api/detect", files={"image": ("x.png", b"nope", "image/png")}).status_code == 400
