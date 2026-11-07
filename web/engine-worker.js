/* Unmark engine worker.
 *
 * Runs off the main thread so the page stays responsive:
 *   detect       multi-scale stroke detector (top-hat / black-hat), JS port of wmremover/detect.py
 *   inpaint      multi-scale PatchMatch completion (Barnes et al. 2009, Wexler et al. 2007)
 *   regions      group mask pixels into crop boxes with context
 *   composite    feathered paste of a filled crop into the original
 *   batch        shared-watermark estimation (Dekel et al. CVPR 2017) + alpha-matte inversion
 *
 * Images travel as RGBA Uint8ClampedArray, masks as Uint8Array (0 / 255).
 */
"use strict";

let currentId = 0;
const IN_WORKER = typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope;
function progress(p, label) { if (IN_WORKER) self.postMessage({ id: currentId, progress: p, label }); }

if (IN_WORKER) self.onmessage = (e) => {
  const { id, op, args } = e.data;
  currentId = id;
  try {
    const fn = OPS[op];
    if (!fn) throw new Error("unknown op " + op);
    const result = fn(args);
    const transfer = [];
    collectBuffers(result, transfer);
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: (err && err.message) || String(err) });
  }
};

function collectBuffers(obj, out) {
  if (!obj || typeof obj !== "object") return;
  if (ArrayBuffer.isView(obj)) { if (!out.includes(obj.buffer)) out.push(obj.buffer); return; }
  for (const k in obj) collectBuffers(obj[k], out);
}

const odd = (n) => { n = Math.round(n); return n % 2 ? n : n + 1; };

// ---------------------------------------------------------------- morphology
// Van Herk / Gil-Werman running min/max: O(1) per pixel for any window size.
function vhgwLine(src, dst, n, k, isMax, buf, g, hh, get, set) {
  const r = k >> 1, L = n + k - 1, neutral = isMax ? -Infinity : Infinity;
  for (let i = 0; i < L; i++) { const j = i - r; buf[i] = j >= 0 && j < n ? get(j) : neutral; }
  for (let i = 0; i < L; i++) {
    const v = buf[i];
    g[i] = i % k === 0 ? v : (isMax ? (v > g[i - 1] ? v : g[i - 1]) : (v < g[i - 1] ? v : g[i - 1]));
  }
  for (let i = L - 1; i >= 0; i--) {
    const v = buf[i];
    hh[i] = (i % k === k - 1 || i === L - 1) ? v : (isMax ? (v > hh[i + 1] ? v : hh[i + 1]) : (v < hh[i + 1] ? v : hh[i + 1]));
  }
  for (let j = 0; j < n; j++) {
    const a = hh[j], b = g[j + k - 1];
    set(j, isMax ? (a > b ? a : b) : (a < b ? a : b));
  }
}
function morph(src, w, h, k, isMax) {
  if (k <= 1) return Float32Array.from(src);
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  const L = Math.max(w, h) + k;
  const buf = new Float32Array(L), g = new Float32Array(L), hh = new Float32Array(L);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    vhgwLine(src, tmp, w, k, isMax, buf, g, hh, (j) => src[o + j], (j, v) => { tmp[o + j] = v; });
  }
  for (let x = 0; x < w; x++) {
    vhgwLine(tmp, out, h, k, isMax, buf, g, hh, (j) => tmp[j * w + x], (j, v) => { out[j * w + x] = v; });
  }
  return out;
}
const erode = (a, w, h, k) => morph(a, w, h, k, false);
const dilate = (a, w, h, k) => morph(a, w, h, k, true);

function boxBlur(src, w, h, k) {
  // integral image, window k x k clipped at borders (normalised by actual count)
  const W1 = w + 1, I = new Float64Array(W1 * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) { row += src[y * w + x]; I[(y + 1) * W1 + x + 1] = I[y * W1 + x + 1] + row; }
  }
  const r = k >> 1, out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
      const s = I[y1 * W1 + x1] - I[y0 * W1 + x1] - I[y1 * W1 + x0] + I[y0 * W1 + x0];
      out[y * w + x] = s / ((y1 - y0) * (x1 - x0));
    }
  }
  return out;
}

function percentile(arr, p) {
  const n = arr.length, step = Math.max(1, Math.floor(n / 200000));
  const s = new Float32Array(Math.ceil(n / step));
  for (let i = 0, j = 0; i < n; i += step) s[j++] = arr[i];
  s.sort();
  return s[Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))))];
}

// ------------------------------------------------------- connected components
function label(mask, w, h) {
  // two-pass union-find, 8-connectivity. mask: truthy = foreground
  const labels = new Int32Array(w * h), parent = [0];
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const unite = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[a < b ? b : a] = a < b ? a : b; };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x; if (!mask[i]) continue;
    let l = 0;
    const nb = [x > 0 ? i - 1 : -1, y > 0 && x > 0 ? i - w - 1 : -1, y > 0 ? i - w : -1, y > 0 && x < w - 1 ? i - w + 1 : -1];
    for (const j of nb) if (j >= 0 && labels[j]) { if (!l) l = labels[j]; else unite(l, labels[j]); }
    if (!l) { l = parent.length; parent.push(l); }
    labels[i] = l;
  }
  const remap = new Int32Array(parent.length); let n = 0;
  for (let i = 1; i < parent.length; i++) { const r = find(i); if (!remap[r]) remap[r] = ++n; remap[i] = remap[r]; }
  const area = new Int32Array(n + 1), x0 = new Int32Array(n + 1).fill(1 << 30), y0 = new Int32Array(n + 1).fill(1 << 30), x1 = new Int32Array(n + 1), y1 = new Int32Array(n + 1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x; if (!labels[i]) continue;
    const l = (labels[i] = remap[labels[i]]);
    area[l]++; if (x < x0[l]) x0[l] = x; if (y < y0[l]) y0[l] = y; if (x > x1[l]) x1[l] = x; if (y > y1[l]) y1[l] = y;
  }
  return { labels, n, area, x0, y0, x1, y1 };
}

function removeSmall(mask, w, h, minArea) {
  const { labels, area } = label(mask, w, h);
  for (let i = 0; i < mask.length; i++) if (labels[i] && area[labels[i]] < minArea) mask[i] = 0;
  return mask;
}

function dilateMask(mask, w, h, r) {
  if (r <= 0) return mask;
  const f = new Float32Array(mask.length);
  for (let i = 0; i < f.length; i++) f[i] = mask[i] ? 1 : 0;
  // disk-ish: square of side 2r+1 then trim corners via a second, diamond pass
  const sq = dilate(f, w, h, 2 * r + 1);
  const out = new Uint8Array(mask.length);
  const r2 = (r + 0.5) * (r + 0.5);
  // exact disk only for small radii, square approximation for big ones (cheap)
  if (r <= 6) {
    const offs = [];
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) if (dx * dx + dy * dy <= r2) offs.push([dx, dy]);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (!sq[i]) continue;
      if (mask[i]) { out[i] = 255; continue; }
      for (const [dx, dy] of offs) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < w && yy < h && mask[yy * w + xx]) { out[i] = 255; break; }
      }
    }
  } else for (let i = 0; i < out.length; i++) out[i] = sq[i] ? 255 : 0;
  return out;
}

function closeMask(mask, w, h, k) {
  const f = new Float32Array(mask.length);
  for (let i = 0; i < f.length; i++) f[i] = mask[i] ? 1 : 0;
  const c = erode(dilate(f, w, h, k), w, h, k);
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < out.length; i++) out[i] = c[i] ? 255 : 0;
  return out;
}

function fillHoles(mask, w, h, maxFrac) {
  const inv = new Uint8Array(mask.length);
  for (let i = 0; i < inv.length; i++) inv[i] = mask[i] ? 0 : 1;
  const { labels, n, area, x0, y0, x1, y1 } = label(inv, w, h);
  const fill = new Uint8Array(n + 1);
  for (let l = 1; l <= n; l++) {
    const touches = x0[l] === 0 || y0[l] === 0 || x1[l] === w - 1 || y1[l] === h - 1;
    fill[l] = !touches && area[l] <= maxFrac * w * h ? 1 : 0;
  }
  const out = Uint8Array.from(mask);
  for (let i = 0; i < out.length; i++) if (labels[i] && fill[labels[i]]) out[i] = 255;
  return out;
}

// ------------------------------------------------------------------ detection
function rgbToLab(r, g, b) {
  const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  const R = lin(r), G = lin(g), B = lin(b);
  let X = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047, Y = 0.2126 * R + 0.7152 * G + 0.0722 * B, Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883;
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  X = f(X); Y = f(Y); Z = f(Z);
  return [116 * Y - 16, 500 * (X - Y), 200 * (Y - Z)];
}

function strokeResponse(data, w, h) {
  const n = w * h, gray = new Float32Array(n), sat = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = data[p], g = data[p + 1], b = data[p + 2];
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    sat[i] = mx ? (mx - mn) / mx : 0;
  }
  const kmax = Math.max(9, Math.min(w, h) / 18);
  const ks = new Set();
  for (let i = 0; i < 5; i++) ks.add(odd(7 * Math.pow(kmax / 7, i / 4)));
  const resp = new Float32Array(n), polarity = new Int8Array(n);
  for (const k of ks) {
    const ero = erode(gray, w, h, k), dil = dilate(gray, w, h, k);
    const open = dilate(ero, w, h, k), close = erode(dil, w, h, k);
    for (let i = 0; i < n; i++) {
      const bright = gray[i] - open[i], dark = close[i] - gray[i], v = Math.max(bright, dark);
      if (v > resp[i]) { resp[i] = v; polarity[i] = bright >= dark ? 1 : -1; }
    }
  }
  for (let i = 0; i < n; i++) resp[i] *= 1 - 0.7 * sat[i];
  const local = boxBlur(resp, w, h, odd(kmax * 3));
  for (let i = 0; i < n; i++) resp[i] = Math.max(0, resp[i] - 0.6 * local[i]);
  // Normalise against a bounded contrast scale (grey levels) so one very
  // high-contrast edge (a skyline, a horizon) cannot drown out the watermark.
  const m = Math.min(45, Math.max(20, percentile(resp, 99.7)));
  for (let i = 0; i < n; i++) resp[i] = Math.min(1, resp[i] / m);
  resp.polarity = polarity; resp.sat = sat; resp.kmax = kmax;
  return resp;
}

function detect({ data, w, h, box, sensitivity = 0.5, grow = 3 }) {
  sensitivity = Math.min(1, Math.max(0, sensitivity));
  const n = w * h;
  let out = new Uint8Array(n);
  if (box) {
    let [x, y, bw, bh] = box.map(Math.round);
    x = Math.max(0, x); y = Math.max(0, y); bw = Math.min(w - x, bw); bh = Math.min(h - y, bh);
    if (bw < 2 || bh < 2) return { mask: out, found: false };
    const pad = Math.max(8, Math.round(0.15 * Math.max(bw, bh)));
    const cx0 = Math.max(0, x - pad), cy0 = Math.max(0, y - pad), cx1 = Math.min(w, x + bw + pad), cy1 = Math.min(h, y + bh + pad);
    const cw = cx1 - cx0, ch = cy1 - cy0, sub = new Uint8ClampedArray(cw * ch * 4);
    for (let yy = 0; yy < ch; yy++) sub.set(data.subarray(((cy0 + yy) * w + cx0) * 4, ((cy0 + yy) * w + cx1) * 4), yy * cw * 4);
    const resp = strokeResponse(sub, cw, ch);
    const thr = Math.max(0.08, percentile(resp, 100 - (8 + 30 * sensitivity)));
    let cnt = 0;
    for (let yy = y; yy < y + bh; yy++) for (let xx = x; xx < x + bw; xx++) {
      if (resp[(yy - cy0) * cw + (xx - cx0)] > thr) { out[yy * w + xx] = 255; cnt++; }
    }
    if (cnt < 0.01 * bw * bh) {
      out.fill(0);
      for (let yy = y; yy < y + bh; yy++) out.fill(255, yy * w + x, yy * w + x + bw);
      return { mask: out, found: true };
    }
    return { mask: finishMask(out, w, h, grow), found: true };
  }

  progress(0.1, "Scanning strokes");
  const resp = strokeResponse(data, w, h);
  progress(0.7, "Scoring regions");
  const thr = 0.55 - 0.35 * sensitivity;
  const strokes = new Uint8Array(n);
  let any = false;
  for (let i = 0; i < n; i++) if (resp[i] > thr) { strokes[i] = 255; any = true; }
  if (!any) return { mask: out, found: false };
  const gk = odd(Math.max(15, Math.min(h, w) / 18));
  const regions = closeMask(strokes, w, h, gk);
  const { labels, n: nl, area } = label(regions, w, h);
  const minA = 0.0008 * n, maxA = 0.6 * n;
  const cnt = new Float64Array(nl + 1), rs = new Float64Array(nl + 1), sa = new Float64Array(nl + 1), sb = new Float64Array(nl + 1), sa2 = new Float64Array(nl + 1), sb2 = new Float64Array(nl + 1);
  for (let i = 0; i < n; i++) {
    const l = labels[i]; if (!l || !strokes[i] || area[l] < minA || area[l] > maxA) continue;
    const [, A, B] = rgbToLab(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]);
    cnt[l]++; rs[l] += resp[i]; sa[l] += A; sb[l] += B; sa2[l] += A * A; sb2[l] += B * B;
  }
  // Overlay cues: (1) a watermark's strokes are all lighter or all darker than
  // what is behind them, texture mixes both; (2) a white/grey/black overlay
  // lowers saturation relative to the surrounding background.
  const notStroke = new Float32Array(n), satBgNum = new Float32Array(n);
  for (let i = 0; i < n; i++) { notStroke[i] = strokes[i] ? 0 : 1; satBgNum[i] = resp.sat[i] * notStroke[i]; }
  const bk = odd(resp.kmax * 2), bgN = boxBlur(satBgNum, w, h, bk), bgD = boxBlur(notStroke, w, h, bk);
  const pol = new Float64Array(nl + 1), dsat = new Float64Array(nl + 1);
  for (let i = 0; i < n; i++) {
    const l = labels[i]; if (!l || !strokes[i]) continue;
    pol[l] += resp.polarity[i];
    if (bgD[i] > 0.05) dsat[l] += bgN[i] / bgD[i] - resp.sat[i];
  }
  const keep = new Uint8Array(nl + 1); let kept = 0;
  for (let l = 1; l <= nl; l++) {
    if (cnt[l] < 30) continue;
    const c = cnt[l], density = c / area[l];
    const stdA = Math.sqrt(Math.max(0, sa2[l] / c - (sa[l] / c) ** 2)), stdB = Math.sqrt(Math.max(0, sb2[l] / c - (sb[l] / c) ** 2));
    const consistency = Math.abs(pol[l]) / c, ds = Math.max(-0.2, Math.min(0.2, dsat[l] / c));
    const score = (rs[l] / c) * 1.5 + Math.min(density, 0.5) - (stdA + stdB) / 2 / 40 + 0.8 * (consistency - 0.6) + 2 * ds;
    if (score > 0.75 - 0.3 * sensitivity) { keep[l] = 1; kept++; }
  }
  if (!kept) return { mask: out, found: false };
  // Hysteresis: inside kept regions also take fainter pixels of the region's
  // polarity, which recovers letters crossing bright or busy backgrounds.
  const lowThr = thr * 0.45;
  for (let i = 0; i < n; i++) {
    const l = labels[i]; if (!l || !keep[l]) continue;
    if (strokes[i] || (resp[i] > lowThr && resp.polarity[i] === Math.sign(pol[l]))) out[i] = 255;
  }
  out = fillHoles(out, w, h, 0.003); // letter interiors enclosed by detected edges
  return { mask: finishMask(out, w, h, grow), found: true };
}

function finishMask(mask, w, h, grow) {
  removeSmall(mask, w, h, 4);
  return dilateMask(mask, w, h, grow);
}

// ------------------------------------------------------------------- regions
function regions({ mask, w, h, marginFrac = 0.6, minMargin = 48 }) {
  const { n, x0, y0, x1, y1 } = label(mask, w, h);
  let boxes = [];
  for (let l = 1; l <= n; l++) {
    const bw = x1[l] - x0[l] + 1, bh = y1[l] - y0[l] + 1, m = Math.max(minMargin, Math.round(marginFrac * Math.max(bw, bh)));
    boxes.push([Math.max(0, x0[l] - m), Math.max(0, y0[l] - m), Math.min(w, x1[l] + 1 + m), Math.min(h, y1[l] + 1 + m)]);
  }
  let merged = true;
  while (merged) {
    merged = false; const out = [];
    for (const b of boxes) {
      const o = out.find((o) => b[0] < o[2] && o[0] < b[2] && b[1] < o[3] && o[1] < b[3]);
      if (o) { o[0] = Math.min(o[0], b[0]); o[1] = Math.min(o[1], b[1]); o[2] = Math.max(o[2], b[2]); o[3] = Math.max(o[3], b[3]); merged = true; }
      else out.push(b.slice());
    }
    boxes = out;
  }
  return { boxes };
}

function cropRGBA(data, w, box) {
  const [x0, y0, x1, y1] = box, cw = x1 - x0, ch = y1 - y0, out = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) out.set(data.subarray(((y0 + y) * w + x0) * 4, ((y0 + y) * w + x1) * 4), y * cw * 4);
  return out;
}
function cropMask(mask, w, box) {
  const [x0, y0, x1, y1] = box, cw = x1 - x0, ch = y1 - y0, out = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++) out.set(mask.subarray((y0 + y) * w + x0, (y0 + y) * w + x1), y * cw);
  return out;
}

// Feathered paste of `filled` (crop-sized RGBA) into `data` at `box`, inside `mask` (crop-sized).
function compositeInto(data, w, box, filled, cmask, feather = 3) {
  const [x0, y0, x1, y1] = box, cw = x1 - x0, ch = y1 - y0, n = cw * ch;
  const hard = new Float32Array(n);
  for (let i = 0; i < n; i++) hard[i] = cmask[i] ? 1 : 0;
  let m = hard;
  if (feather > 0) {
    const k = 2 * feather + 1, grown = dilate(hard, cw, ch, k), soft = boxBlur(grown, cw, ch, k);
    m = new Float32Array(n);
    for (let i = 0; i < n; i++) m[i] = Math.max(hard[i], soft[i] * grown[i]);
  }
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const i = y * cw + x, a = m[i]; if (!a) continue;
    const p = ((y0 + y) * w + x0 + x) * 4, q = i * 4;
    for (let c = 0; c < 3; c++) data[p + c] = filled[q + c] * a + data[p + c] * (1 - a) + 0.5;
  }
}

function composite({ data, w, box, filled, cmask, feather }) {
  compositeInto(data, w, box, filled, cmask, feather);
  return { data };
}

// ------------------------------------------------------------ push-pull fill
// Smooth membrane-like interpolation of holes from a weighted image pyramid.
function pushPull(img, w, h, known) {
  // img Float32 3ch, known Float32 weights in [0,1]; returns filled Float32 3ch
  if (w <= 1 && h <= 1) return Float32Array.from(img);
  const W2 = Math.ceil(w / 2), H2 = Math.ceil(h / 2);
  const cimg = new Float32Array(W2 * H2 * 3), ck = new Float32Array(W2 * H2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, j = (y >> 1) * W2 + (x >> 1), k = known[i];
    ck[j] += k; for (let c = 0; c < 3; c++) cimg[j * 3 + c] += img[i * 3 + c] * k;
  }
  for (let j = 0; j < W2 * H2; j++) {
    if (ck[j] > 0) for (let c = 0; c < 3; c++) cimg[j * 3 + c] /= ck[j];
    ck[j] = Math.min(1, ck[j]);
  }
  const coarse = W2 === w && H2 === h ? cimg : pushPull(cimg, W2, H2, ck);
  const out = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, k = known[i];
    // bilinear sample of coarse
    const fx = Math.min(W2 - 1, Math.max(0, (x - 0.5) / 2)), fy = Math.min(H2 - 1, Math.max(0, (y - 0.5) / 2));
    const xa = Math.floor(fx), ya = Math.floor(fy), xb = Math.min(W2 - 1, xa + 1), yb = Math.min(H2 - 1, ya + 1), tx = fx - xa, ty = fy - ya;
    for (let c = 0; c < 3; c++) {
      const v = (coarse[(ya * W2 + xa) * 3 + c] * (1 - tx) + coarse[(ya * W2 + xb) * 3 + c] * tx) * (1 - ty) +
                (coarse[(yb * W2 + xa) * 3 + c] * (1 - tx) + coarse[(yb * W2 + xb) * 3 + c] * tx) * ty;
      out[i * 3 + c] = img[i * 3 + c] * k + v * (1 - k);
    }
  }
  return out;
}

// ----------------------------------------------------------------- PatchMatch
const PR = 3; // patch radius (7x7 patches)

function patchMatchFill(img0, w, h, hole0, onLevel) {
  // img0 Float32 3ch (0..255), hole0 Uint8 (1 = fill). Returns Float32 3ch.
  // Build pyramid. Coarse pixel is known if any child is known (average of known children).
  const pyr = [{ img: img0, hole: hole0, w, h }];
  // depth from hole thickness (chamfer distance to nearest known pixel)
  const dist = new Float32Array(w * h);
  for (let i = 0; i < dist.length; i++) dist[i] = hole0[i] ? 1e9 : 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = y * w + x; if (x) dist[i] = Math.min(dist[i], dist[i - 1] + 1); if (y) dist[i] = Math.min(dist[i], dist[i - w] + 1); }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) { const i = y * w + x; if (x < w - 1) dist[i] = Math.min(dist[i], dist[i + 1] + 1); if (y < h - 1) dist[i] = Math.min(dist[i], dist[i + w] + 1); }
  let maxD = 0; for (let i = 0; i < dist.length; i++) if (dist[i] < 1e9 && dist[i] > maxD) maxD = dist[i];
  const levels = Math.max(0, Math.min(Math.ceil(Math.log2(Math.max(1, maxD / 2))), 6));
  for (let l = 0; l < levels; l++) {
    const p = pyr[pyr.length - 1];
    if (Math.min(p.w, p.h) < 4 * (2 * PR + 1)) break;
    const W2 = p.w >> 1, H2 = p.h >> 1, img = new Float32Array(W2 * H2 * 3), hole = new Uint8Array(W2 * H2);
    for (let y = 0; y < H2; y++) for (let x = 0; x < W2; x++) {
      let s0 = 0, s1 = 0, s2 = 0, c = 0;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const i = (2 * y + dy) * p.w + 2 * x + dx;
        if (!p.hole[i]) { s0 += p.img[i * 3]; s1 += p.img[i * 3 + 1]; s2 += p.img[i * 3 + 2]; c++; }
      }
      const j = y * W2 + x;
      if (c) { img[j * 3] = s0 / c; img[j * 3 + 1] = s1 / c; img[j * 3 + 2] = s2 / c; } else hole[j] = 1;
    }
    pyr.push({ img, hole, w: W2, h: H2 });
  }

  let est = null, nnf = null, prev = null;
  for (let li = pyr.length - 1; li >= 0; li--) {
    const { img, hole, w: lw, h: lh } = pyr[li];
    const N = lw * lh;
    // valid source centers: whole patch known and inside image
    const holeD = dilateMaskBool(hole, lw, lh, PR);
    const valid = new Uint8Array(N), srcList = [];
    for (let y = PR; y < lh - PR; y++) for (let x = PR; x < lw - PR; x++) { const i = y * lw + x; if (!holeD[i]) { valid[i] = 1; srcList.push(i); } }
    const targets = [];
    for (let i = 0; i < N; i++) if (holeD[i]) targets.push(i);
    if (!srcList.length || !targets.length) {
      // not enough context: smooth fill
      const known = new Float32Array(N); for (let i = 0; i < N; i++) known[i] = hole[i] ? 0 : 1;
      est = pushPull(img, lw, lh, known); prev = { w: lw, h: lh }; nnf = null; continue;
    }
    // initial estimate
    if (!est) {
      const known = new Float32Array(N); for (let i = 0; i < N; i++) known[i] = hole[i] ? 0 : 1;
      est = pushPull(img, lw, lh, known);
    } else {
      const up = new Float32Array(N * 3);
      for (let y = 0; y < lh; y++) for (let x = 0; x < lw; x++) {
        const i = y * lw + x;
        if (!hole[i]) { up[i * 3] = img[i * 3]; up[i * 3 + 1] = img[i * 3 + 1]; up[i * 3 + 2] = img[i * 3 + 2]; continue; }
        const px = Math.min(prev.w - 1, x >> 1), py = Math.min(prev.h - 1, y >> 1), j = (py * prev.w + px) * 3;
        up[i * 3] = est[j]; up[i * 3 + 1] = est[j + 1]; up[i * 3 + 2] = est[j + 2];
      }
      est = up;
    }
    // NNF init (upsample previous or random)
    const nf = new Int32Array(N).fill(-1), nd = new Float32Array(N).fill(Infinity);
    for (const t of targets) {
      let s = -1;
      if (nnf) {
        const x = t % lw, y = (t / lw) | 0, px = Math.min(prev.w - 1, x >> 1), py = Math.min(prev.h - 1, y >> 1);
        let ps = nnf[py * prev.w + px];
        if (ps < 0) ps = py * prev.w + px;
        const sx = 2 * (ps % prev.w) + (x & 1), sy = 2 * ((ps / prev.w) | 0) + (y & 1);
        if (sx < lw && sy < lh && valid[sy * lw + sx]) s = sy * lw + sx;
      }
      if (s < 0) s = srcList[(Math.random() * srcList.length) | 0];
      nf[t] = s;
    }
    const isCoarsest = li === pyr.length - 1;
    const emIters = isCoarsest ? 8 : li === 0 ? 3 : 4;
    for (let em = 0; em < emIters; em++) {
      for (const t of targets) nd[t] = patchDist(est, lw, lh, t, nf[t], Infinity);
      for (let it = 0; it < 2; it++) pmPass(est, lw, lh, targets, nf, nd, valid, srcList, (it + em) % 2 === 1);
      vote(est, img, hole, lw, lh, targets, nf, nd);
    }
    nnf = nf; prev = { w: lw, h: lh };
    if (onLevel) onLevel(pyr.length - li, pyr.length);
  }
  return est;
}

function dilateMaskBool(hole, w, h, r) {
  const f = new Float32Array(hole.length);
  for (let i = 0; i < f.length; i++) f[i] = hole[i] ? 1 : 0;
  const d = dilate(f, w, h, 2 * r + 1), out = new Uint8Array(hole.length);
  for (let i = 0; i < out.length; i++) out[i] = d[i] ? 1 : 0;
  return out;
}

function patchDist(est, w, h, t, s, best) {
  const tx = t % w, ty = (t / w) | 0, sx = s % w, sy = (s / w) | 0;
  let d = 0, cnt = 0;
  for (let dy = -PR; dy <= PR; dy++) {
    const yy = ty + dy; if (yy < 0 || yy >= h) continue;
    const trow = yy * w, srow = (sy + dy) * w;
    for (let dx = -PR; dx <= PR; dx++) {
      const xx = tx + dx; if (xx < 0 || xx >= w) continue;
      const a = (trow + xx) * 3, b = (srow + sx + dx) * 3;
      const e0 = est[a] - est[b], e1 = est[a + 1] - est[b + 1], e2 = est[a + 2] - est[b + 2];
      d += e0 * e0 + e1 * e1 + e2 * e2; cnt++;
    }
    if (d > best) return Infinity; // normalised value can only be larger
  }
  return (d / cnt) * 49;
}

function pmPass(est, w, h, targets, nf, nd, valid, srcList, reverse) {
  const n = targets.length, step = reverse ? -1 : 1, start = reverse ? n - 1 : 0;
  const dir = reverse ? 1 : -1; // neighbour offset to propagate from
  const maxR = Math.max(w, h);
  for (let k = 0, idx = start; k < n; k++, idx += step) {
    const t = targets[idx], tx = t % w, ty = (t / w) | 0;
    let best = nd[t], bs = nf[t];
    // propagation from horizontal and vertical neighbours
    for (const [ox, oy] of [[dir, 0], [0, dir]]) {
      const nx = tx + ox, ny = ty + oy; if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const ns = nf[ny * w + nx]; // -1: neighbour is outside the target set
      const cx = (ns >= 0 ? ns % w : nx) - ox, cy = (ns >= 0 ? (ns / w) | 0 : ny) - oy;
      if (cx < 0 || cy < 0 || cx >= w || cy >= h) continue;
      const cand = cy * w + cx;
      if (!valid[cand] || cand === bs) continue;
      const d = patchDist(est, w, h, t, cand, best);
      if (d < best) { best = d; bs = cand; }
    }
    // random search around current best
    const bx = bs % w, by = (bs / w) | 0;
    for (let r = maxR; r >= 1; r >>= 1) {
      const cx = bx + (((Math.random() * 2 - 1) * r) | 0), cy = by + (((Math.random() * 2 - 1) * r) | 0);
      if (cx < 0 || cy < 0 || cx >= w || cy >= h) continue;
      const cand = cy * w + cx; if (!valid[cand]) continue;
      const d = patchDist(est, w, h, t, cand, best);
      if (d < best) { best = d; bs = cand; }
    }
    // occasional global jump helps escape poor local optima
    const g = srcList[(Math.random() * srcList.length) | 0], dg = patchDist(est, w, h, t, g, best);
    if (dg < best) { best = dg; bs = g; }
    nf[t] = bs; nd[t] = best;
  }
}

function vote(est, img, hole, w, h, targets, nf, nd) {
  const N = w * h, acc = new Float32Array(N * 3), wsum = new Float32Array(N);
  // similarity weights relative to the median patch distance
  const ds = new Float32Array(targets.length);
  for (let i = 0; i < targets.length; i++) ds[i] = nd[targets[i]];
  const sorted = Float32Array.from(ds).sort();
  const sigma2 = Math.max(1, sorted[Math.floor(sorted.length * 0.75)] || 1);
  for (let i = 0; i < targets.length; i++) {
    const t = targets[i], s = nf[t], tx = t % w, ty = (t / w) | 0, sx = s % w, sy = (s / w) | 0;
    const wt = Math.exp(-ds[i] / (2 * sigma2)) + 1e-6;
    for (let dy = -PR; dy <= PR; dy++) {
      const yy = ty + dy; if (yy < 0 || yy >= h) continue;
      for (let dx = -PR; dx <= PR; dx++) {
        const xx = tx + dx; if (xx < 0 || xx >= w) continue;
        const p = yy * w + xx; if (!hole[p]) continue;
        const q = ((sy + dy) * w + sx + dx) * 3;
        acc[p * 3] += est[q] * wt; acc[p * 3 + 1] += est[q + 1] * wt; acc[p * 3 + 2] += est[q + 2] * wt; wsum[p] += wt;
      }
    }
  }
  for (let p = 0; p < N; p++) if (hole[p] && wsum[p] > 0) {
    est[p * 3] = acc[p * 3] / wsum[p]; est[p * 3 + 1] = acc[p * 3 + 1] / wsum[p]; est[p * 3 + 2] = acc[p * 3 + 2] / wsum[p];
  }
}

function inpaintPatchMatch(data, w, h, mask, label = "Filling") {
  const { boxes } = regions({ mask, w, h, marginFrac: 0.8, minMargin: 40 });
  const out = Uint8ClampedArray.from(data);
  boxes.forEach((box, bi) => {
    const [x0, y0, x1, y1] = box, cw = x1 - x0, ch = y1 - y0;
    const crop = cropRGBA(out, w, box), cmask = cropMask(mask, w, box);
    const f = new Float32Array(cw * ch * 3), hole = new Uint8Array(cw * ch);
    for (let i = 0; i < cw * ch; i++) { f[i * 3] = crop[i * 4]; f[i * 3 + 1] = crop[i * 4 + 1]; f[i * 3 + 2] = crop[i * 4 + 2]; hole[i] = cmask[i] ? 1 : 0; }
    const filled = patchMatchFill(f, cw, ch, hole, (l, L) => progress((bi + l / L) / boxes.length, `${label} region ${bi + 1}/${boxes.length}`));
    const rgba = new Uint8ClampedArray(cw * ch * 4);
    for (let i = 0; i < cw * ch; i++) { rgba[i * 4] = filled[i * 3]; rgba[i * 4 + 1] = filled[i * 3 + 1]; rgba[i * 4 + 2] = filled[i * 3 + 2]; rgba[i * 4 + 3] = 255; }
    compositeInto(out, w, box, rgba, cmask, 2);
  });
  return out;
}

function inpaint({ data, w, h, mask }) {
  return { data: inpaintPatchMatch(data, w, h, mask) };
}

// ------------------------------------------------------- batch: shared mark
function median(arr) { const s = Float32Array.from(arr).sort(); const n = s.length; return n % 2 ? s[n >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]); }

function batch({ images, w, h, maxAlpha = 0.85 }) {
  const N = images.length, P = w * h;
  if (N < 3) throw new Error("Batch mode needs at least 3 images.");
  progress(0.02, "Estimating shared watermark");
  // 1. median gradient across images (Sobel, per channel)
  const mag = new Float32Array(P), gx = new Float32Array(N), gy = new Float32Array(N);
  for (let y = 0; y < h; y++) {
    if (y % 64 === 0) progress(0.02 + 0.25 * (y / h), "Estimating shared watermark");
    const ym = Math.max(0, y - 1), yp = Math.min(h - 1, y + 1);
    for (let x = 0; x < w; x++) {
      const xm = Math.max(0, x - 1), xp = Math.min(w - 1, x + 1);
      let m2 = 0;
      for (let c = 0; c < 3; c++) {
        for (let n = 0; n < N; n++) {
          const d = images[n];
          const a = d[(ym * w + xm) * 4 + c], b = d[(ym * w + x) * 4 + c], cc = d[(ym * w + xp) * 4 + c];
          const e = d[(y * w + xm) * 4 + c], f = d[(y * w + xp) * 4 + c];
          const g = d[(yp * w + xm) * 4 + c], hh = d[(yp * w + x) * 4 + c], ii = d[(yp * w + xp) * 4 + c];
          gx[n] = (cc + 2 * f + ii - a - 2 * e - g) / 255; gy[n] = (g + 2 * hh + ii - a - 2 * b - cc) / 255;
        }
        const mx = median(gx), my = median(gy);
        m2 += mx * mx + my * my;
      }
      mag[y * w + x] = Math.sqrt(m2);
    }
  }
  const med = percentile(mag, 50);
  const dev = new Float32Array(P); for (let i = 0; i < P; i++) dev[i] = Math.abs(mag[i] - med);
  const mad = percentile(dev, 50) * 1.4826 + 1e-6;
  let mask = new Uint8Array(P);
  for (let i = 0; i < P; i++) if (mag[i] > med + 9 * mad) mask[i] = 255;
  mask = closeMask(mask, w, h, odd(Math.max(5, Math.min(w, h) / 100)));
  removeSmall(mask, w, h, Math.max(6, Math.round(5e-5 * P)));
  mask = fillHoles(mask, w, h, 0.002);
  mask = dilateMask(mask, w, h, 2);
  let count = 0; for (let i = 0; i < P; i++) if (mask[i]) count++;
  const results = images.map((d) => Uint8ClampedArray.from(d));
  const alpha = new Float32Array(P), hiMask = new Uint8Array(P);
  if (!count) return { results, mask, alpha, hiMask, found: false };
  if (count > 0.4 * P) throw new Error("Could not isolate a shared watermark. Check that the images carry the same mark in the same place.");

  // 2. content priors J_n by smooth fill (only near the mask, for speed)
  progress(0.3, "Building content priors");
  const { boxes } = regions({ mask, w, h, marginFrac: 0.3, minMargin: 16 });
  const priors = images.map(() => new Float32Array(P * 3));
  images.forEach((d, n) => {
    for (const box of boxes) {
      const [x0, y0, x1, y1] = box, cw = x1 - x0, ch = y1 - y0, f = new Float32Array(cw * ch * 3), k = new Float32Array(cw * ch);
      for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
        const i = y * cw + x, p = (y0 + y) * w + x0 + x;
        f[i * 3] = d[p * 4]; f[i * 3 + 1] = d[p * 4 + 1]; f[i * 3 + 2] = d[p * 4 + 2]; k[i] = mask[p] ? 0 : 1;
      }
      const filled = pushPull(f, cw, ch, k);
      for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
        const i = y * cw + x, p = (y0 + y) * w + x0 + x;
        priors[n][p * 3] = filled[i * 3]; priors[n][p * 3 + 1] = filled[i * 3 + 1]; priors[n][p * 3 + 2] = filled[i * 3 + 2];
      }
    }
    progress(0.3 + 0.3 * ((n + 1) / N), "Building content priors");
  });

  // 3. per-pixel least squares: I - J = u - a*J  (u = a*W)
  progress(0.65, "Solving alpha matte");
  const idx = []; for (let i = 0; i < P; i++) if (mask[i]) idx.push(i);
  const M = idx.length, U = new Float32Array(M * 3), A = new Float32Array(M);
  const Ib = new Float32Array(N * 3), Jb = new Float32Array(N * 3);
  for (let m = 0; m < M; m++) {
    const p = idx[m];
    for (let n = 0; n < N; n++) for (let c = 0; c < 3; c++) { Ib[n * 3 + c] = images[n][p * 4 + c] / 255; Jb[n * 3 + c] = priors[n][p * 3 + c] / 255; }
    // normal equations 4x4
    const S = [[N, 0, 0, 0], [0, N, 0, 0], [0, 0, N, 0], [0, 0, 0, 0]], b = [0, 0, 0, 0];
    for (let n = 0; n < N; n++) for (let c = 0; c < 3; c++) {
      const J = Jb[n * 3 + c], D = Ib[n * 3 + c] - J;
      S[c][3] -= J; S[3][c] -= J; S[3][3] += J * J; b[c] += D; b[3] -= J * D;
    }
    for (let i = 0; i < 4; i++) S[i][i] += 1e-4;
    const sol = solve4(S, b);
    U[m * 3] = sol[0]; U[m * 3 + 1] = sol[1]; U[m * 3 + 2] = sol[2]; A[m] = sol[3];
  }
  // flat-colour refinement
  const wr = [], wg = [], wb = [];
  for (let m = 0; m < M; m++) if (A[m] > 0.15) { wr.push(U[m * 3] / A[m]); wg.push(U[m * 3 + 1] / A[m]); wb.push(U[m * 3 + 2] / A[m]); }
  if (wr.length >= 20) {
    const Wc = [median(wr), median(wg), median(wb)];
    const spread = (median(wr.map((v) => Math.abs(v - Wc[0]))) + median(wg.map((v) => Math.abs(v - Wc[1]))) + median(wb.map((v) => Math.abs(v - Wc[2])))) / 3;
    if (spread < 0.12) {
      for (let c = 0; c < 3; c++) Wc[c] = Math.min(1, Math.max(0, Wc[c]));
      for (let m = 0; m < M; m++) {
        const p = idx[m]; let num = 0, den = 0;
        for (let n = 0; n < N; n++) for (let c = 0; c < 3; c++) {
          const J = priors[n][p * 3 + c] / 255, G = Wc[c] - J, D = images[n][p * 4 + c] / 255 - J;
          num += D * G; den += G * G;
        }
        const a = Math.min(0.99, Math.max(0, num / (den + 1e-6)));
        A[m] = a; U[m * 3] = a * Wc[0]; U[m * 3 + 1] = a * Wc[1]; U[m * 3 + 2] = a * Wc[2];
      }
    }
  }
  // 4. invert the blend; high-alpha pixels lean on the prior and are flagged for real inpainting
  progress(0.85, "Recovering pixels");
  const lo = maxAlpha - 0.15;
  for (let m = 0; m < M; m++) {
    const p = idx[m], a = Math.min(0.99, Math.max(0, A[m]));
    alpha[p] = a;
    let t = Math.min(1, Math.max(0, (a - lo) / (maxAlpha - lo)));
    const wp = t * t * (3 - 2 * t), den = Math.max(1 - a, 1 - maxAlpha);
    if (wp > 0) hiMask[p] = 255;
    for (let n = 0; n < N; n++) for (let c = 0; c < 3; c++) {
      const u = Math.min(1, Math.max(0, U[m * 3 + c]));
      const inv = Math.min(1, Math.max(0, (images[n][p * 4 + c] / 255 - u) / den));
      results[n][p * 4 + c] = (inv * (1 - wp) + (priors[n][p * 3 + c] / 255) * wp) * 255 + 0.5;
    }
  }
  progress(1, "Done");
  return { results, mask, alpha, hiMask, found: true };
}

function solve4(S, b) {
  const M = S.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < 4; c++) {
    let piv = c; for (let r = c + 1; r < 4; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c] || 1e-12;
    for (let r = 0; r < 4; r++) if (r !== c) { const f = M[r][c] / d; for (let k = c; k < 5; k++) M[r][k] -= f * M[c][k]; }
  }
  return [M[0][4] / M[0][0], M[1][4] / M[1][1], M[2][4] / M[2][2], M[3][4] / M[3][3]];
}

const OPS = { detect, inpaint, regions, composite, batch };
const API = { detect, inpaint, regions, composite, compositeInto, batch, patchMatchFill, pushPull, cropRGBA, cropMask, dilateMask };
if (typeof module !== "undefined") module.exports = API; // Node tests
else if (!IN_WORKER) self.UnmarkEngine = API; // main-thread helpers
