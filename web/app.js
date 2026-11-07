/* Unmark Studio: UI, canvas editing, AI (LaMa via onnxruntime-web) and worker orchestration. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const E = window.UnmarkEngine;

  const MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx";
  const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const MODEL_CACHE = "unmark-models-v1";
  const LAMA_SIZE = 512;
  const RUBY = "rgb(207,61,42)";
  const IOS = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const store = {
    get(k) { try { return localStorage.getItem("unmark." + k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem("unmark." + k, v); } catch { /* storage unavailable */ } },
  };

  // ------------------------------------------------------------ worker RPC
  let worker = null, rpcId = 0;
  const pending = new Map();
  function getWorker() {
    if (worker) return worker;
    worker = new Worker("engine-worker.js");
    worker.onmessage = (e) => {
      const m = e.data, p = pending.get(m.id);
      if (!p) return;
      if ("progress" in m) { p.onProgress && p.onProgress(m.progress, m.label); return; }
      pending.delete(m.id);
      m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
    };
    worker.onerror = (e) => { for (const p of pending.values()) p.reject(new Error(e.message || "Engine crashed")); pending.clear(); worker = null; };
    return worker;
  }
  function rpc(op, args, transfer = [], onProgress) {
    return new Promise((resolve, reject) => {
      const id = ++rpcId;
      pending.set(id, { resolve, reject, onProgress });
      getWorker().postMessage({ id, op, args }, transfer);
    });
  }

  // ------------------------------------------------------------ state
  const stage = $("stage"), board = $("board");
  const imgC = $("imageCanvas"), maskC = $("maskCanvas"), beforeC = $("beforeCanvas");
  const ictx = imgC.getContext("2d", { willReadFrequently: true });
  const mctx = maskC.getContext("2d", { willReadFrequently: true });
  let W = 0, H = 0, original = null, fileName = "image";
  let tool = "brush", view = { s: 1, x: 0, y: 0 };
  let undoStack = [], redoStack = [], hasResult = false, busy = false, loadGen = 0;

  function setStatus(msg, err = false) { const s = $("status"); s.textContent = msg; s.classList.toggle("err", err); }
  function setBusy(on, title = "Working…") {
    busy = on;
    $("veil").classList.toggle("on", on); $("veilTitle").textContent = title; $("veilLabel").textContent = ""; $("veilBar").style.width = "0";
    ["btnRemove", "btnDetect", "btnOpen", "btnSample", "fabRemove"].forEach((id) => ($(id).disabled = on));
  }
  function veilProgress(p, label) { $("veilBar").style.width = Math.round(p * 100) + "%"; if (label) $("veilLabel").textContent = label; }
  const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

  // ------------------------------------------------------------ loading
  function loadSource(src, name) {
    W = src.width; H = src.height; fileName = name; loadGen++;
    for (const c of [imgC, maskC, beforeC]) { c.width = W; c.height = H; }
    ictx.drawImage(src, 0, 0);
    original = ictx.getImageData(0, 0, W, H);
    mctx.clearRect(0, 0, W, H);
    undoStack = []; redoStack = []; hasResult = false;
    board.style.width = W + "px"; board.style.height = H + "px";
    $("dims").textContent = `${W} × ${H}`;
    setCompare(false); syncButtons(); fit();
  }
  async function openFile(file) {
    if (busy) { setStatus("Wait for the current step to finish, then open the image again.", true); return; }
    if (!file || !file.type.startsWith("image/")) { setStatus("That file is not an image. Try a JPEG, PNG or WebP.", true); return; }
    try {
      const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
      // iPhone and iPad Safari cannot draw canvases larger than 16.7 megapixels.
      const MAX = IOS ? 16e6 : 40e6;
      if (bmp.width * bmp.height > MAX) { setStatus(`This image is ${(bmp.width * bmp.height / 1e6).toFixed(0)} megapixels. Unmark handles up to ${MAX / 1e6} on this device.`, true); bmp.close(); return; }
      loadSource(bmp, file.name.replace(/\.[^.]+$/, "") || "image");
      bmp.close();
      setStatus("Paint over the watermark, drag a box around it, or press Auto-detect.");
    } catch { setStatus("This image could not be decoded by your browser.", true); }
  }
  $("btnOpen").onclick = () => $("fileInput").click();
  $("fileInput").onchange = (e) => { openFile(e.target.files[0]); e.target.value = ""; };
  $("btnSample").onclick = () => loadSample(true);
  $("btnRevert").onclick = () => { if (!original) return; pushUndo(true); ictx.putImageData(original, 0, 0); hasResult = false; setCompare(false); syncButtons(); setStatus("Back to the image as opened."); };
  stage.addEventListener("dragover", (e) => { e.preventDefault(); stage.classList.add("dragover"); });
  stage.addEventListener("dragleave", () => stage.classList.remove("dragover"));
  stage.addEventListener("drop", (e) => { e.preventDefault(); stage.classList.remove("dragover"); openFile(e.dataTransfer.files[0]); });
  // a file dropped anywhere else would make the browser open it and leave the page
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
  window.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/")); if (f) openFile(f); });

  // Procedural sample: a landscape with fine texture and a stock-style preview mark.
  function makeSample() {
    const w = 1200, h = 800, c = document.createElement("canvas"); c.width = w; c.height = h;
    const g = c.getContext("2d");
    let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const sky = g.createLinearGradient(0, 0, 0, h * 0.6);
    sky.addColorStop(0, "#5d86c4"); sky.addColorStop(0.6, "#c9b9c8"); sky.addColorStop(1, "#f2c79a");
    g.fillStyle = sky; g.fillRect(0, 0, w, h);
    g.fillStyle = "rgba(255,236,200,.9)"; g.beginPath(); g.arc(w * 0.84, h * 0.22, 40, 0, 7); g.fill();
    for (let i = 0; i < 26; i++) { g.fillStyle = `rgba(255,255,255,${0.05 + rnd() * 0.08})`; g.beginPath(); g.ellipse(rnd() * w, rnd() * h * 0.35, 60 + rnd() * 120, 8 + rnd() * 14, 0, 0, 7); g.fill(); }
    const ridge = (base, amp, col, jag) => {
      g.fillStyle = col; g.beginPath(); g.moveTo(0, h);
      for (let x = 0; x <= w; x += 6) g.lineTo(x, base - amp * (Math.sin(x / 140 + base) * 0.6 + Math.sin(x / 47 + base * 2) * 0.25) - rnd() * jag);
      g.lineTo(w, h); g.fill();
    };
    ridge(h * 0.58, 70, "#6f7f9a", 6); ridge(h * 0.64, 50, "#4f6a63", 5); ridge(h * 0.72, 30, "#3d5a3b", 4);
    const water = g.createLinearGradient(0, h * 0.74, 0, h);
    water.addColorStop(0, "#7b8fb0"); water.addColorStop(1, "#2f4560");
    g.fillStyle = water; g.fillRect(0, h * 0.76, w, h * 0.24);
    for (let i = 0; i < 900; i++) { g.fillStyle = `rgba(255,255,255,${rnd() * 0.18})`; g.fillRect(rnd() * w, h * 0.76 + rnd() * h * 0.24, 6 + rnd() * 26, 1.5); }
    for (let i = 0; i < 5000; i++) { g.fillStyle = `rgba(${20 + rnd() * 40},${50 + rnd() * 50},${20 + rnd() * 30},.5)`; g.fillRect(rnd() * w, h * 0.7 + rnd() * h * 0.06, 2, 3 + rnd() * 5); }
    const id = g.getImageData(0, 0, w, h), d = id.data;
    for (let i = 0; i < d.length; i += 4) { const n = (rnd() - 0.5) * 10; d[i] += n; d[i + 1] += n; d[i + 2] += n; }
    g.putImageData(id, 0, 0);
    // watermark: semi-transparent white text plus a corner badge
    g.save(); g.globalAlpha = 0.5; g.fillStyle = "#fff"; g.textAlign = "center"; g.textBaseline = "middle";
    g.font = "800 110px 'Bricolage Grotesque', 'Arial Black', sans-serif"; g.fillText("© SAMPLE", w * 0.5, h * 0.47);
    g.font = "600 34px 'IBM Plex Sans', Arial, sans-serif"; g.fillText("PREVIEW · NOT FOR USE", w * 0.5, h * 0.47 + 78);
    g.globalAlpha = 0.55; g.lineWidth = 5; g.strokeStyle = "#fff"; g.beginPath(); g.arc(w - 80, h - 70, 34, 0, 7); g.stroke();
    g.font = "700 30px 'IBM Plex Sans', Arial, sans-serif"; g.fillText("S", w - 80, h - 69);
    g.restore();
    return c;
  }
  async function loadSample(announce) {
    try { await document.fonts.ready; } catch { /* fonts optional */ }
    loadSource(makeSample(), "sample");
    const gen = loadGen;
    if (announce !== false) setStatus("Sample loaded. Detecting the watermark…");
    await detect(null, true);
    if (gen === loadGen) setStatus("Sample loaded with the watermark detected (red). Press Remove watermark to see it go.");
  }

  // ------------------------------------------------------------ view: zoom & pan
  function applyView() {
    board.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.s})`;
    $("zoomOut2").textContent = Math.round(view.s * 100) + "%";
    updateCursorSize(); placeSplit();
  }
  function fit() {
    const r = stage.getBoundingClientRect(), pad = 24;
    view.s = Math.min((r.width - pad * 2) / W, (r.height - pad * 2) / H, 4);
    view.x = (r.width - W * view.s) / 2; view.y = (r.height - H * view.s) / 2; applyView();
  }
  function zoomAt(f, cx, cy) {
    const ns = Math.min(16, Math.max(0.05, view.s * f));
    view.x = cx - (cx - view.x) * (ns / view.s); view.y = cy - (cy - view.y) * (ns / view.s); view.s = ns; applyView();
  }
  const stageCenter = () => { const r = stage.getBoundingClientRect(); return [r.width / 2, r.height / 2]; };
  $("zoomIn").onclick = () => zoomAt(1.25, ...stageCenter());
  $("zoomOut").onclick = () => zoomAt(0.8, ...stageCenter());
  $("zoomFit").onclick = fit;
  $("zoom100").onclick = () => zoomAt(1 / view.s, ...stageCenter());
  stage.addEventListener("wheel", (e) => {
    e.preventDefault(); const r = stage.getBoundingClientRect();
    if (e.ctrlKey || e.metaKey || Math.abs(e.deltaY) > Math.abs(e.deltaX)) zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    else { view.x -= e.deltaX; applyView(); }
  }, { passive: false });
  window.addEventListener("resize", () => W && fit());

  // ------------------------------------------------------------ tools & painting
  function setTool(t) {
    tool = t; stage.dataset.tool = t;
    document.querySelectorAll("[data-tool]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.tool === t)));
    if (hasResult) setCompare(false);
  }
  document.querySelectorAll("[data-tool]").forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
  const sizeIn = $("size");
  sizeIn.value = store.get("size") || 32;
  const syncSize = () => { $("sizeOut").textContent = sizeIn.value + " px"; store.set("size", sizeIn.value); updateCursorSize(); };
  sizeIn.oninput = syncSize;
  $("sens").oninput = () => ($("sensOut").textContent = $("sens").value);
  $("grow").oninput = () => ($("growOut").textContent = $("grow").value + " px");

  function toImage(e) { const r = stage.getBoundingClientRect(); return { x: (e.clientX - r.left - view.x) / view.s, y: (e.clientY - r.top - view.y) / view.s, sx: e.clientX - r.left, sy: e.clientY - r.top }; }
  function updateCursorSize() { const c = $("cursor"), d = +sizeIn.value; c.style.width = c.style.height = d + "px"; }
  function stroke(a, b) {
    mctx.globalCompositeOperation = tool === "erase" ? "destination-out" : "source-over";
    mctx.strokeStyle = RUBY; mctx.lineCap = mctx.lineJoin = "round"; mctx.lineWidth = +sizeIn.value / view.s;
    mctx.beginPath(); mctx.moveTo(a.x, a.y); mctx.lineTo(b.x, b.y); mctx.stroke();
    mctx.globalCompositeOperation = "source-over";
  }

  let drag = null; const pointers = new Map();
  stage.addEventListener("pointerdown", (e) => {
    if (!W || busy) return;
    stage.setPointerCapture(e.pointerId); pointers.set(e.pointerId, e);
    const p = toImage(e);
    if (pointers.size === 2) { // pinch start: cancel any stroke
      if (drag && drag.kind === "paint") undo(true);
      const [a, b] = [...pointers.values()];
      drag = { kind: "pinch", d: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY), cx: (a.clientX + b.clientX) / 2, cy: (a.clientY + b.clientY) / 2 };
      return;
    }
    if (tool === "hand" || e.button === 1 || spaceDown) { drag = { kind: "pan", sx: e.clientX, sy: e.clientY, vx: view.x, vy: view.y }; stage.classList.add("panning"); return; }
    if (hasResult && splitOn) setCompare(false);
    if (tool === "box") { drag = { kind: "box", a: p, b: p }; return; }
    pushUndo(false); drag = { kind: "paint", last: p }; stroke(p, p);
  });
  stage.addEventListener("pointermove", (e) => {
    const c = $("cursor"), p = toImage(e);
    if (tool === "brush" || tool === "erase") { c.style.display = "block"; c.style.left = p.sx + "px"; c.style.top = p.sy + "px"; } else c.style.display = "none";
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, e);
    if (!drag) return;
    if (drag.kind === "pinch" && pointers.size === 2) {
      const [a, b] = [...pointers.values()], d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      const r = stage.getBoundingClientRect(), cx = (a.clientX + b.clientX) / 2, cy = (a.clientY + b.clientY) / 2;
      view.x += cx - drag.cx; view.y += cy - drag.cy; zoomAt(d / drag.d, cx - r.left, cy - r.top);
      drag.d = d; drag.cx = cx; drag.cy = cy; return;
    }
    if (drag.kind === "pan") { view.x = drag.vx + e.clientX - drag.sx; view.y = drag.vy + e.clientY - drag.sy; applyView(); return; }
    if (drag.kind === "box") {
      drag.b = p; const m = $("marquee"), a = drag.a;
      m.style.display = "block";
      m.style.left = view.x + Math.min(a.x, p.x) * view.s + "px"; m.style.top = view.y + Math.min(a.y, p.y) * view.s + "px";
      m.style.width = Math.abs(p.x - a.x) * view.s + "px"; m.style.height = Math.abs(p.y - a.y) * view.s + "px"; return;
    }
    if (drag.kind === "paint") { stroke(drag.last, p); drag.last = p; }
  });
  const endDrag = async (e) => {
    pointers.delete(e.pointerId);
    if (!drag) return;
    const d = drag; drag = null; stage.classList.remove("panning");
    if (d.kind === "box") {
      $("marquee").style.display = "none";
      const x = Math.max(0, Math.min(d.a.x, d.b.x)), y = Math.max(0, Math.min(d.a.y, d.b.y));
      const w = Math.min(W, Math.max(d.a.x, d.b.x)) - x, h = Math.min(H, Math.max(d.a.y, d.b.y)) - y;
      if (w > 3 && h > 3) await detect([x, y, w, h]);
    }
    if (d.kind === "paint") syncButtons();
  };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);
  stage.addEventListener("pointerleave", () => ($("cursor").style.display = "none"));

  // ------------------------------------------------------------ history
  function snapshot(withImage) { return { mask: mctx.getImageData(0, 0, W, H), image: withImage ? ictx.getImageData(0, 0, W, H) : null, hasResult }; }
  function trim(stack) {
    // keep memory bounded: roughly 600 MB of snapshots
    let bytes = 0;
    for (let i = stack.length - 1; i >= 0; i--) {
      bytes += W * H * 4 * (stack[i].image ? 2 : 1);
      if (bytes > 600e6 || stack.length - i > 40) { stack.splice(0, i + 1); break; }
    }
  }
  function pushUndo(withImage) { undoStack.push(snapshot(withImage)); trim(undoStack); redoStack = []; syncButtons(); }
  function restore(snap, into) {
    into.push(snapshot(!!snap.image));
    mctx.putImageData(snap.mask, 0, 0);
    if (snap.image) { ictx.putImageData(snap.image, 0, 0); hasResult = snap.hasResult; setCompare(false); }
  }
  function undo(silent) { const s = undoStack.pop(); if (!s) return; restore(s, redoStack); if (silent) redoStack.pop(); syncButtons(); }
  function redo() { const s = redoStack.pop(); if (!s) return; restore(s, undoStack); syncButtons(); }
  $("btnUndo").onclick = () => undo(); $("btnRedo").onclick = redo;
  $("btnClear").onclick = () => { pushUndo(false); mctx.clearRect(0, 0, W, H); syncButtons(); };
  function syncButtons() {
    $("btnUndo").disabled = !undoStack.length; $("btnRedo").disabled = !redoStack.length;
    $("btnRevert").disabled = !hasResult && !undoStack.some((s) => s.image);
    for (const id of ["btnCompare", "btnDownload", "btnDownloadJpg"]) $(id).disabled = !hasResult;
    $("btnShowResult").hidden = !hasResult;
  }

  // ------------------------------------------------------------ mask I/O
  function readMask() {
    const d = mctx.getImageData(0, 0, W, H).data, m = new Uint8Array(W * H);
    let any = false;
    for (let i = 0; i < m.length; i++) if (d[i * 4 + 3] > 20) { m[i] = 255; any = true; }
    return any ? m : null;
  }
  function addMask(m, mw, mh) {
    // m: Uint8 mask at (mw, mh); scaled up to the image and merged with the painted mask
    const t = document.createElement("canvas"); t.width = mw; t.height = mh;
    const tc = t.getContext("2d"), id = tc.createImageData(mw, mh);
    for (let i = 0; i < m.length; i++) if (m[i]) { id.data[i * 4] = 207; id.data[i * 4 + 1] = 61; id.data[i * 4 + 2] = 42; id.data[i * 4 + 3] = 255; }
    tc.putImageData(id, 0, 0);
    if (mw === W && mh === H) { mctx.drawImage(t, 0, 0); return; }
    // upscale smoothly then harden the edge
    const u = document.createElement("canvas"); u.width = W; u.height = H;
    const uc = u.getContext("2d"); uc.imageSmoothingEnabled = true; uc.imageSmoothingQuality = "high"; uc.drawImage(t, 0, 0, W, H);
    const ud = uc.getImageData(0, 0, W, H);
    for (let i = 3; i < ud.data.length; i += 4) { const on = ud.data[i] >= 110; ud.data[i] = on ? 255 : 0; if (on) { ud.data[i - 3] = 207; ud.data[i - 2] = 61; ud.data[i - 1] = 42; } }
    uc.putImageData(ud, 0, 0); mctx.drawImage(u, 0, 0);
  }

  // ------------------------------------------------------------ detection
  async function detect(box, quiet) {
    if (!W) return;
    if (!quiet) setBusy(true, box ? "Selecting strokes in box" : "Detecting watermark");
    try {
      const WORK = 720, sc = Math.min(1, WORK / Math.min(W, H));
      const ww = Math.max(1, Math.round(W * sc)), wh = Math.max(1, Math.round(H * sc));
      let data;
      if (sc < 1) { const t = document.createElement("canvas"); t.width = ww; t.height = wh; const tc = t.getContext("2d"); tc.imageSmoothingQuality = "high"; tc.drawImage(imgC, 0, 0, ww, wh); data = tc.getImageData(0, 0, ww, wh).data; }
      else data = ictx.getImageData(0, 0, W, H).data;
      const grow = Math.max(0, Math.round(+$("grow").value * sc));
      const gen = loadGen;
      const res = await rpc("detect", { data, w: ww, h: wh, box: box && box.map((v) => v * sc), sensitivity: +$("sens").value / 100, grow }, [data.buffer], veilProgress);
      if (gen !== loadGen) return; // another image was opened meanwhile: this mask belongs to the old one
      if (res.found) { pushUndo(false); addMask(res.mask, ww, wh); }
      if (!quiet) setStatus(res.found ? "Watermark marked in red. Touch up with Brush or Erase, then press Remove watermark." : "No watermark found. Drag a box around it, raise the sensitivity, or paint it with the brush.", !res.found);
      syncButtons();
    } catch (err) { setStatus("Detection failed: " + err.message, true); }
    if (!quiet) setBusy(false);
  }
  $("btnDetect").onclick = () => detect(null);

  // ------------------------------------------------------------ AI model (LaMa)
  let lama = null, lamaLoading = null, aiFallback = ""; // aiFallback: why the last removal could not use AI
  function aiState(state, text) { const c = $("aiChip"); c.dataset.state = state; c.textContent = text; }
  async function fetchModel(onProgress) {
    let cache = null;
    try { cache = await caches.open(MODEL_CACHE); const hit = await cache.match(MODEL_URL); if (hit) return new Uint8Array(await hit.arrayBuffer()); } catch { cache = null; }
    const res = await fetch(MODEL_URL, { mode: "cors" });
    if (!res.ok) throw new Error(`model download failed (HTTP ${res.status})`);
    const total = +res.headers.get("content-length") || 208e6, reader = res.body.getReader(), chunks = [];
    let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; onProgress(got / total, got); }
    const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
    if (cache) { try { await cache.put(MODEL_URL, new Response(buf, { headers: { "content-type": "application/octet-stream" } })); } catch { /* quota: fine */ } }
    return buf;
  }
  async function createSession(bytes) {
    const ort = window.ort;
    ort.env.wasm.wasmPaths = ORT_WASM;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    const tries = navigator.gpu ? [["webgpu"], ["wasm"]] : [["wasm"]];
    let last;
    for (const eps of tries) {
      try {
        const s = await ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: "all" });
        // smoke test so a GPU that cannot run an op falls back now, not mid-edit
        await runLama(s, new Float32Array(3 * LAMA_SIZE * LAMA_SIZE), new Float32Array(LAMA_SIZE * LAMA_SIZE));
        return { session: s, ep: eps[0] };
      } catch (e) { last = e; }
    }
    throw last;
  }
  async function loadLama(fromFile) {
    if (lama) return lama;
    if (lamaLoading) return lamaLoading;
    if (!window.ort) { aiState("error", "AI runtime blocked here"); $("aiNote").textContent = "This page cannot load the AI runtime, so PatchMatch is used. On the deployed site the AI engine works."; throw new Error("AI runtime unavailable"); }
    lamaLoading = (async () => {
      const bar = $("aiBar"), fill = bar.firstElementChild; bar.hidden = false; $("btnLoadAI").disabled = true;
      try {
        aiState("busy", fromFile ? "Reading model file" : "Downloading model");
        const bytes = fromFile ? new Uint8Array(await fromFile.arrayBuffer()) : await fetchModel((p, got) => { fill.style.width = Math.round(p * 100) + "%"; aiState("busy", `Downloading ${(got / 1e6).toFixed(0)} / 208 MB`); });
        fill.style.width = "100%"; aiState("busy", "Starting AI engine");
        const { session, ep } = await createSession(bytes);
        lama = session;
        aiState("ready", `AI ready (${ep === "webgpu" ? "GPU" : "CPU"})`);
        $("aiNote").textContent = ep === "webgpu" ? "Running on your GPU through WebGPU." : "Running on your CPU. Each region takes a few seconds.";
        $("btnLoadAI").hidden = true; $("btnModelFile").hidden = true; store.set("aiWanted", "1");
        return lama;
      } catch (e) {
        aiState("error", "AI model unavailable");
        const blocked = e instanceof TypeError || /fetch|network|wasm|import/i.test(e.message || "");
        $("aiNote").textContent = blocked
          ? "This page cannot reach the model download or AI runtime (a network or security setting is blocking it). PatchMatch will be used instead. You can also pick a local .onnx file."
          : `Could not start the AI model (${e.message}). PatchMatch will be used instead.`;
        $("btnLoadAI").disabled = false; throw e;
      } finally { bar.hidden = true; lamaLoading = null; }
    })();
    return lamaLoading;
  }
  $("btnLoadAI").onclick = () => loadLama().catch(() => {});
  $("btnModelFile").onclick = () => $("modelInput").click();
  $("modelInput").onchange = (e) => { const f = e.target.files[0]; if (f) loadLama(f).catch(() => {}); e.target.value = ""; };

  async function runLama(session, img, mask) {
    const ort = window.ort, [iName, mName] = session.inputNames.some((n) => /mask/i.test(n)) ? [session.inputNames.find((n) => !/mask/i.test(n)), session.inputNames.find((n) => /mask/i.test(n))] : session.inputNames;
    const out = await session.run({
      [iName]: new ort.Tensor("float32", img, [1, 3, LAMA_SIZE, LAMA_SIZE]),
      [mName]: new ort.Tensor("float32", mask, [1, 1, LAMA_SIZE, LAMA_SIZE]),
    });
    return out[session.outputNames[0]].data;
  }

  async function inpaintAI(data, mask, W, H) {
    const { boxes } = E.regions({ mask, w: W, h: H, marginFrac: 0.6, minMargin: 48 });
    const src = document.createElement("canvas"); src.width = W; src.height = H;
    src.getContext("2d").putImageData(new ImageData(data, W, H), 0, 0);
    const sq = document.createElement("canvas"); sq.width = sq.height = LAMA_SIZE;
    const sctx = sq.getContext("2d", { willReadFrequently: true }); sctx.imageSmoothingQuality = "high";
    const N = LAMA_SIZE * LAMA_SIZE;
    for (let bi = 0; bi < boxes.length; bi++) {
      const box = boxes[bi], [x0, y0, x1, y1] = box, cw = x1 - x0, ch = y1 - y0;
      veilProgress(bi / boxes.length, `AI filling region ${bi + 1} of ${boxes.length}`); await nextFrame();
      sctx.clearRect(0, 0, LAMA_SIZE, LAMA_SIZE); sctx.drawImage(src, x0, y0, cw, ch, 0, 0, LAMA_SIZE, LAMA_SIZE);
      const px = sctx.getImageData(0, 0, LAMA_SIZE, LAMA_SIZE).data;
      // nearest-neighbour mask resample
      const cmask = E.cropMask(mask, W, box), m = new Float32Array(N);
      for (let y = 0; y < LAMA_SIZE; y++) { const sy = Math.min(ch - 1, Math.floor((y + 0.5) * ch / LAMA_SIZE)); for (let x = 0; x < LAMA_SIZE; x++) { const sx = Math.min(cw - 1, Math.floor((x + 0.5) * cw / LAMA_SIZE)); m[y * LAMA_SIZE + x] = cmask[sy * cw + sx] ? 1 : 0; } }
      const t = new Float32Array(3 * N);
      for (let i = 0; i < N; i++) { const k = 1 - m[i]; t[i] = px[i * 4] / 255 * k; t[N + i] = px[i * 4 + 1] / 255 * k; t[2 * N + i] = px[i * 4 + 2] / 255 * k; }
      const y = await runLama(lama, t, m);
      let mx = 0; for (let i = 0; i < y.length; i += 97) if (y[i] > mx) mx = y[i];
      const sc = mx <= 1.5 ? 255 : 1, outId = sctx.createImageData(LAMA_SIZE, LAMA_SIZE);
      for (let i = 0; i < N; i++) { outId.data[i * 4] = y[i] * sc; outId.data[i * 4 + 1] = y[N + i] * sc; outId.data[i * 4 + 2] = y[2 * N + i] * sc; outId.data[i * 4 + 3] = 255; }
      sctx.putImageData(outId, 0, 0);
      const back = document.createElement("canvas"); back.width = cw; back.height = ch;
      const bctx = back.getContext("2d", { willReadFrequently: true }); bctx.imageSmoothingQuality = "high"; bctx.drawImage(sq, 0, 0, cw, ch);
      E.compositeInto(data, W, box, bctx.getImageData(0, 0, cw, ch).data, cmask, 3);
    }
    return data;
  }

  function wantAI() { const e = $("engine").value; return e === "ai" || (e === "auto" && lama); }

  async function inpaint(data, mask, w = W, h = H) {
    if (wantAI()) {
      try { await loadLama(); return await inpaintAI(data, mask, w, h); }
      catch (e) { if ($("engine").value === "ai") setStatus(aiFallback = `AI engine failed (${e.message}). Used PatchMatch instead.`, true); }
    }
    const res = await rpc("inpaint", { data, w, h, mask }, [data.buffer], veilProgress);
    return res.data;
  }

  // ------------------------------------------------------------ remove
  $("btnRemove").onclick = async () => {
    if (!W || busy) return;
    const mask = readMask();
    if (!mask) { setStatus("Mark the watermark first: paint it, drag a box around it, or press Auto-detect.", true); return; }
    setBusy(true, wantAI() ? "Removing with AI" : "Removing with PatchMatch");
    const t0 = performance.now(); aiFallback = "";
    try {
      pushUndo(true);
      const before = ictx.getImageData(0, 0, W, H);
      beforeC.getContext("2d").putImageData(before, 0, 0);
      const out = await inpaint(before.data.slice(), mask);
      ictx.putImageData(new ImageData(new Uint8ClampedArray(out.buffer || out), W, H), 0, 0);
      mctx.clearRect(0, 0, W, H); hasResult = true; syncButtons(); setCompare(true);
      setStatus(`${aiFallback ? aiFallback + " " : ""}Removed in ${((performance.now() - t0) / 1000).toFixed(1)} s. Drag the line to compare. Leftovers? Pick Brush, paint them, and remove again.`, !!aiFallback);
    } catch (err) { undo(true); setStatus("Removal failed: " + err.message, true); }
    setBusy(false);
  };

  $("fabRemove").onclick = () => $("btnRemove").click();
  $("fabRemove").addEventListener("pointerdown", (e) => e.stopPropagation());

  // ------------------------------------------------------------ compare
  let splitOn = false, splitF = 0.5;
  function setCompare(on) {
    splitOn = on && hasResult; beforeC.hidden = !splitOn; $("split").hidden = !splitOn; placeSplit();
  }
  function placeSplit() {
    if (!splitOn) return;
    const x = view.x + W * view.s * splitF;
    $("split").style.left = x + "px";
    beforeC.style.clipPath = `inset(0 ${(1 - splitF) * 100}% 0 0)`;
  }
  $("split").style.pointerEvents = "none";
  stage.addEventListener("pointerdown", (e) => {
    if (!splitOn || tool === "hand") return;
    const r = stage.getBoundingClientRect(), sx = view.x + W * view.s * splitF;
    if (Math.abs(e.clientX - r.left - sx) < 18) {
      e.stopImmediatePropagation();
      const move = (ev) => { splitF = Math.min(1, Math.max(0, (ev.clientX - r.left - view.x) / (W * view.s))); placeSplit(); };
      const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
      window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    }
  }, true);
  const holdCompare = (on) => { if (!hasResult) return; beforeC.hidden = !on && !splitOn; beforeC.style.clipPath = on ? "none" : beforeC.style.clipPath; if (!on) placeSplit(); };
  $("btnCompare").onpointerdown = () => holdCompare(true);
  $("btnCompare").onpointerup = $("btnCompare").onpointerleave = () => holdCompare(false);
  $("btnCompare").onclick = () => { setCompare(!splitOn); };

  // ------------------------------------------------------------ export
  function blobOf(canvas, type, q) { return new Promise((r) => canvas.toBlob(r, type, q)); }
  function save(blob, name) { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 30000); }
  $("btnDownload").onclick = async () => save(await blobOf(imgC, "image/png"), `${fileName}_unmarked.png`);
  $("btnDownloadJpg").onclick = async () => save(await blobOf(imgC, "image/jpeg", 0.95), `${fileName}_unmarked.jpg`);
  $("btnShowResult").onclick = async () => { $("saveImg").src = URL.createObjectURL(await blobOf(imgC, "image/png")); $("saveDialog").showModal(); };
  $("saveClose").onclick = () => $("saveDialog").close();
  $("saveDialog").addEventListener("close", () => { URL.revokeObjectURL($("saveImg").src); $("saveImg").removeAttribute("src"); });

  // ------------------------------------------------------------ keyboard
  let spaceDown = false;
  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input:not([type=range]), select, textarea") || !$("viewBatch").hidden || $("saveDialog").open) return;
    const k = e.key.toLowerCase();
    // let Enter and Space press the focused button instead of also running a shortcut
    if ((k === "enter" || k === " ") && e.target.closest("button, a, summary")) return;
    if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if ((e.ctrlKey || e.metaKey) && k === "y") { e.preventDefault(); redo(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (k === " ") { spaceDown = true; stage.classList.add("panning"); e.preventDefault(); }
    else if (k === "b") setTool("brush"); else if (k === "e") setTool("erase"); else if (k === "r") setTool("box"); else if (k === "h") setTool("hand");
    else if (k === "[") { sizeIn.value = Math.max(4, +sizeIn.value - 6); syncSize(); } else if (k === "]") { sizeIn.value = +sizeIn.value + 6; syncSize(); }
    else if (k === "c" && !e.repeat) holdCompare(true);
    else if (k === "0") fit(); else if (k === "enter") $("btnRemove").click();
  });
  window.addEventListener("keyup", (e) => { if (e.key === " ") { spaceDown = false; stage.classList.remove("panning"); } if (e.key.toLowerCase() === "c") holdCompare(false); });

  // ------------------------------------------------------------ tabs
  function showTab(batch) {
    $("tabSingle").setAttribute("aria-selected", String(!batch)); $("tabBatch").setAttribute("aria-selected", String(batch));
    $("viewSingle").hidden = batch; $("viewBatch").hidden = !batch;
    if (!batch && W) fit();
    try { history.replaceState(null, "", batch ? "#batch" : "#single"); } catch { /* sandboxed */ }
  }
  $("tabSingle").onclick = () => showTab(false); $("tabBatch").onclick = () => showTab(true);

  // ------------------------------------------------------------ batch
  let bFiles = [], bImages = [], bResults = [], bW = 0, bH = 0;
  $("btnBatchOpen").onclick = () => $("batchInput").click();
  $("batchInput").onchange = async (e) => {
    bFiles = [...e.target.files].filter((f) => f.type.startsWith("image/")); e.target.value = "";
    bResults = []; bImages = []; $("btnBatchZip").disabled = true; $("btnBatchRun").disabled = true; $("batchStatsBox").hidden = true;
    const st = $("batchStatus"); st.classList.remove("err");
    if (!bFiles.length) return;
    st.textContent = "Reading images…";
    const bmps = [];
    for (const f of bFiles) { try { bmps.push(await createImageBitmap(f, { imageOrientation: "from-image" })); } catch { st.textContent = `Could not read ${f.name}.`; st.classList.add("err"); return; } }
    bW = bmps[0].width; bH = bmps[0].height;
    const bad = bmps.findIndex((b) => b.width !== bW || b.height !== bH), sizes = bmps.map((b) => [b.width, b.height]);
    $("batchGrid").innerHTML = ""; $("batchEmpty").hidden = true; bImages = [];
    bmps.forEach((b, i) => {
      const c = document.createElement("canvas"); c.width = b.width; c.height = b.height; const cx = c.getContext("2d", { willReadFrequently: true }); cx.drawImage(b, 0, 0);
      bImages.push(cx.getImageData(0, 0, c.width, c.height).data); b.close();
      const tile = document.createElement("div"); tile.className = "tile";
      const meta = document.createElement("div"); meta.className = "meta";
      const name = document.createElement("span"); name.textContent = bFiles[i].name; meta.appendChild(name);
      tile.append(c, meta); $("batchGrid").appendChild(tile);
    });
    if (bad >= 0) { st.textContent = `${bFiles[bad].name} is ${sizes[bad][0]} × ${sizes[bad][1]}, but the first image is ${bW} × ${bH}. All images must be the same size.`; st.classList.add("err"); $("btnBatchRun").disabled = true; return; }
    if (bFiles.length < 3) { st.textContent = "Choose at least 3 images. The shared mark is found by comparing them."; st.classList.add("err"); $("btnBatchRun").disabled = true; return; }
    st.textContent = `${bFiles.length} images, ${bW} × ${bH}. Ready.`; $("btnBatchRun").disabled = false;
  };
  $("btnBatchRun").onclick = async () => {
    const st = $("batchStatus"); st.classList.remove("err"); $("btnBatchRun").disabled = true;
    const t0 = performance.now();
    try {
      const imgs = bImages.map((d) => d.slice());
      const r = await rpc("batch", { images: imgs, w: bW, h: bH }, imgs.map((d) => d.buffer), (p, l) => (st.textContent = `${l}… ${Math.round(p * 100)}%`));
      if (!r.found) { st.textContent = "No shared watermark found. Check that the mark is in the same place on every image."; st.classList.add("err"); return; }
      let hi = 0, mc = 0, as = 0; for (let i = 0; i < r.mask.length; i++) { if (r.hiMask[i]) hi++; if (r.mask[i]) { mc++; as += r.alpha[i]; } }
      // opaque parts of the mark cannot be inverted: fill them with the inpainting engine
      if (hi) for (let i = 0; i < r.results.length; i++) {
        st.textContent = `Filling opaque parts ${i + 1}/${r.results.length}…`;
        r.results[i] = await inpaint(r.results[i], r.hiMask.slice(), bW, bH);
      }
      bResults = r.results;
      const tiles = $("batchGrid").children;
      r.results.forEach((d, i) => {
        const c = tiles[i].querySelector("canvas"); c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(d.buffer || d), bW, bH), 0, 0);
        const meta = tiles[i].querySelector(".meta"); meta.querySelector("button")?.remove();
        const b = document.createElement("button"); b.className = "btn"; b.textContent = "Save";
        b.onclick = async () => save(await blobOf(c, "image/png"), bFiles[i].name.replace(/\.[^.]+$/, "") + "_unmarked.png");
        meta.appendChild(b);
      });
      // stats + mask preview
      const mcv = $("batchMask"); mcv.width = bW; mcv.height = bH; const mid = mcv.getContext("2d").createImageData(bW, bH);
      for (let i = 0; i < r.mask.length; i++) { const v = r.mask[i] ? Math.round(80 + 175 * Math.min(1, r.alpha[i] / 0.9)) : 0; mid.data[i * 4] = v; mid.data[i * 4 + 1] = v * 0.35; mid.data[i * 4 + 2] = v * 0.3; mid.data[i * 4 + 3] = 255; }
      mcv.getContext("2d").putImageData(mid, 0, 0);
      $("batchStats").innerHTML = `<span>Mark area</span><b>${(100 * mc / r.mask.length).toFixed(2)}% of image</b><span>Mean opacity</span><b>${Math.round(100 * as / Math.max(1, mc))}%</b><span>Inpainted</span><b>${(100 * hi / Math.max(1, mc)).toFixed(1)}% of mark</b><span>Time</span><b>${((performance.now() - t0) / 1000).toFixed(1)} s</b>`;
      $("batchStatsBox").hidden = false; $("btnBatchZip").disabled = false;
      st.textContent = `Done. ${r.results.length} images cleaned.`;
    } catch (err) { st.textContent = err.message; st.classList.add("err"); }
    finally { $("btnBatchRun").disabled = false; }
  };
  $("btnBatchZip").onclick = async () => {
    const tiles = $("batchGrid").children, files = [];
    for (let i = 0; i < bResults.length; i++) {
      const blob = await blobOf(tiles[i].querySelector("canvas"), "image/png");
      files.push({ name: bFiles[i].name.replace(/\.[^.]+$/, "") + "_unmarked.png", data: new Uint8Array(await blob.arrayBuffer()) });
    }
    save(zipStore(files), "unmarked.zip");
  };

  // Minimal ZIP writer (stored, no compression: PNGs are already compressed).
  const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  const crc32 = (d) => { let c = 0xffffffff; for (let i = 0; i < d.length; i++) c = CRC[(c ^ d[i]) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  function zipStore(files) {
    const enc = new TextEncoder(), parts = [], central = [], used = new Set(); let off = 0;
    for (const f of files) {
      // two inputs like a.jpg and a.png would both become a_unmarked.png and overwrite each other on extract
      let nm = f.name; for (let i = 2; used.has(nm); i++) nm = f.name.replace(/(\.[^.]+)?$/, `-${i}$1`);
      used.add(nm);
      const name = enc.encode(nm), crc = crc32(f.data), h = new DataView(new ArrayBuffer(30));
      h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); h.setUint32(14, crc, true); h.setUint32(18, f.data.length, true); h.setUint32(22, f.data.length, true); h.setUint16(26, name.length, true);
      parts.push(new Uint8Array(h.buffer), name, f.data);
      const c = new DataView(new ArrayBuffer(46));
      c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true); c.setUint32(16, crc, true); c.setUint32(20, f.data.length, true); c.setUint32(24, f.data.length, true); c.setUint16(28, name.length, true); c.setUint32(42, off, true);
      central.push(new Uint8Array(c.buffer), name); off += 30 + name.length + f.data.length;
    }
    const csize = central.reduce((s, p) => s + p.length, 0), end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true); end.setUint32(12, csize, true); end.setUint32(16, off, true);
    return new Blob([...parts, ...central, new Uint8Array(end.buffer)], { type: "application/zip" });
  }

  // ------------------------------------------------------------ boot
  async function boot() {
    syncSize();
    const eng = store.get("engine"); if (eng) $("engine").value = eng;
    $("engine").onchange = () => { store.set("engine", $("engine").value); if ($("engine").value === "ai") loadLama().catch(() => {}); };
    if (!window.ort) { aiState("error", "AI runtime blocked here"); $("aiNote").textContent = "This page cannot load the AI runtime, so the PatchMatch engine is used. The deployed site runs the AI model."; $("btnLoadAI").disabled = true; }
    else {
      // load silently if the model is already cached
      try { const c = await caches.open(MODEL_CACHE); if (await c.match(MODEL_URL)) loadLama().catch(() => {}); } catch { /* no Cache API */ }
    }
    if (location.hash === "#batch") showTab(true);
    await loadSample(false);
  }
  window.__unmark = { rpc, get state() { return { W, H, hasResult, tool }; }, readMask, loadLama }; // for tests
  boot();
})();
