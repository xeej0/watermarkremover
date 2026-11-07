// Run with: node --test web/tests/engine.test.cjs
const test = require("node:test");
const assert = require("node:assert");
const E = require("../engine-worker.js");

function rng(seed) { return () => ((seed = (seed * 16807) % 2147483647) / 2147483647); }

// Smooth colourful "photo" with noise
function photo(seed, w = 320, h = 240) {
  const r = rng(seed + 1), d = new Uint8ClampedArray(w * h * 4);
  const p = Array.from({ length: 9 }, () => r() * 2 - 1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4;
    for (let c = 0; c < 3; c++) d[i + c] = 120 + 60 * Math.sin(p[c * 3] * x / 40 + p[c * 3 + 1] * y / 50 + p[c * 3 + 2] * 6) + (r() - 0.5) * 8;
    d[i + 3] = 255;
  }
  return d;
}
// Blocky "logo": bars, like a text mark
function markAlpha(w = 320, h = 240) {
  const a = new Float32Array(w * h);
  const bar = (x0, y0, bw, bh) => { for (let y = y0; y < y0 + bh; y++) for (let x = x0; x < x0 + bw; x++) a[y * w + x] = 1; };
  for (let k = 0; k < 6; k++) { bar(70 + k * 32, 100, 6, 40); bar(70 + k * 32, 100, 20, 6); bar(70 + k * 32, 134, 20, 6); }
  return a;
}
function apply(img, alpha, op = 0.45) {
  const out = Uint8ClampedArray.from(img);
  for (let i = 0; i < alpha.length; i++) if (alpha[i]) for (let c = 0; c < 3; c++) out[i * 4 + c] = img[i * 4 + c] * (1 - op * alpha[i]) + 255 * op * alpha[i];
  return out;
}
function psnr(a, b, m) {
  let s = 0, n = 0;
  for (let i = 0; i < m.length; i++) if (m[i]) for (let c = 0; c < 3; c++) { const d = a[i * 4 + c] - b[i * 4 + c]; s += d * d; n++; }
  return 10 * Math.log10(255 * 255 / (s / n));
}
const W = 320, H = 240, ALPHA = markAlpha(), GT = ALPHA.map((v) => (v > 0 ? 1 : 0));

test("detect finds the mark", () => {
  const { mask, found } = E.detect({ data: apply(photo(1), ALPHA), w: W, h: H });
  assert.ok(found);
  let hit = 0, tot = 0, fp = 0;
  for (let i = 0; i < GT.length; i++) { if (GT[i]) { tot++; if (mask[i]) hit++; } else if (mask[i]) fp++; }
  assert.ok(hit / tot > 0.85, `recall ${hit / tot}`);
  assert.ok(fp / (GT.length - tot) < 0.08, `fp ${fp / (GT.length - tot)}`);
});

test("box detection stays inside the box", () => {
  const { mask } = E.detect({ data: apply(photo(2), ALPHA), w: W, h: H, box: [60, 90, 120, 60], grow: 0 });
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (mask[y * W + x]) assert.ok(x >= 60 && x < 180 && y >= 90 && y < 150);
});

test("PatchMatch inpainting improves the masked area", () => {
  const clean = photo(3), wm = apply(clean, ALPHA), mask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let on = 0; for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const yy = y + dy, xx = x + dx; if (yy >= 0 && xx >= 0 && yy < H && xx < W && GT[yy * W + xx]) on = 1; }
    mask[y * W + x] = on ? 255 : 0;
  }
  const { data } = E.inpaint({ data: wm, w: W, h: H, mask });
  assert.ok(psnr(data, clean, GT) > psnr(wm, clean, GT) + 6);
  for (let i = 0; i < mask.length; i++) if (!mask[i]) {
    // untouched far from the mask (feather radius is small)
    const x = i % W, y = (i / W) | 0; if (x < 50 || y < 80 || y > 160) assert.strictEqual(data[i * 4], wm[i * 4]);
  }
});

test("batch matting recovers the real pixels", () => {
  const cleans = [0, 1, 2, 3, 4, 5].map((s) => photo(100 + s)), wms = cleans.map((c) => apply(c, ALPHA));
  const r = E.batch({ images: wms, w: W, h: H });
  assert.ok(r.found);
  for (let n = 0; n < cleans.length; n++) assert.ok(psnr(r.results[n], cleans[n], GT) > 32, `image ${n}: ${psnr(r.results[n], cleans[n], GT)}`);
});

test("batch rejects too few images", () => {
  assert.throws(() => E.batch({ images: [photo(1), photo(2)], w: W, h: H }));
});

test("regions merge nearby components and keep far ones apart", () => {
  const m = new Uint8Array(400 * 400);
  const sq = (x0, y0) => { for (let y = y0; y < y0 + 10; y++) for (let x = x0; x < x0 + 10; x++) m[y * 400 + x] = 255; };
  sq(50, 50); sq(75, 50); sq(350, 350);
  assert.strictEqual(E.regions({ mask: m, w: 400, h: 400 }).boxes.length, 2);
});
