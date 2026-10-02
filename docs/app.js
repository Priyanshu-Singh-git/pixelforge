// PixelForge browser demo: x4 super-resolution with ONNX Runtime Web, tiled, fully client-side.
const $ = (id) => document.getElementById(id);
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
const TILE = 128, PAD = 8, SCALE = 4, MAX_SIDE = 512;

const state = { meta: null, session: null, lr: null, sr: null, truth: null, mode: "bicubic", split: 0.5, results: {}, tileable: false };
window.pixelforge = state; // exposed for automated checks

function imgToCanvas(img, maxSide = MAX_SIDE) {
  const s = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(img.naturalWidth * s)); c.height = Math.max(1, Math.round(img.naturalHeight * s));
  const ctx = c.getContext("2d"); ctx.imageSmoothingQuality = "high"; ctx.drawImage(img, 0, 0, c.width, c.height);
  return c;
}

// run the model on one tile (ImageData) -> Float32 CHW output
async function runTile(data, w, h) {
  const n = w * h, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) x[c * n + i] = data[i * 4 + c] / 255;
  const out = await state.session.run({ lr: new ort.Tensor("float32", x, [1, 3, h, w]) });
  return out.sr.data;
}

// tiled x4 upscale with overlap so tile seams don't show
async function upscale(src, onProgress) {
  const W = src.width, H = src.height, OW = W * SCALE, OH = H * SCALE;
  const sctx = src.getContext("2d");
  const out = new ImageData(OW, OH);
  const tiles = [];
  for (let ty = 0; ty < H; ty += TILE) for (let tx = 0; tx < W; tx += TILE) tiles.push([tx, ty]);
  let done = 0;
  for (const [tx, ty] of tiles) {
    const x0 = Math.max(tx - PAD, 0), y0 = Math.max(ty - PAD, 0);
    const x1 = Math.min(tx + TILE + PAD, W), y1 = Math.min(ty + TILE + PAD, H);
    const tw = x1 - x0, th = y1 - y0;
    const y = await runTile(sctx.getImageData(x0, y0, tw, th).data, tw, th);
    const OTW = tw * SCALE, OTH = th * SCALE, on = OTW * OTH;
    const cx0 = (tx - x0) * SCALE, cy0 = (ty - y0) * SCALE;
    const cw = Math.min(TILE, W - tx) * SCALE, ch = Math.min(TILE, H - ty) * SCALE;
    for (let yy = 0; yy < ch; yy++) for (let xx = 0; xx < cw; xx++) {
      const si = (cy0 + yy) * OTW + (cx0 + xx), di = ((ty * SCALE + yy) * OW + (tx * SCALE + xx)) * 4;
      for (let c = 0; c < 3; c++) out.data[di + c] = Math.min(255, Math.max(0, Math.round(y[c * on + si] * 255)));
      out.data[di + 3] = 255;
    }
    onProgress && onProgress(++done / tiles.length);
    await new Promise((r) => setTimeout(r, 0)); // keep the page responsive
  }
  const c = document.createElement("canvas"); c.width = OW; c.height = OH;
  c.getContext("2d").putImageData(out, 0, 0);
  return c;
}

// fill colours hidden under alpha=0 from opaque neighbours (prevents coloured halos at transparent edges)
function bleed(data, w, h) {
  const known = new Uint8Array(w * h);
  let any = false, all = true;
  for (let i = 0; i < w * h; i++) { known[i] = data[i * 4 + 3] > 0 ? 1 : 0; any = any || !!known[i]; all = all && !!known[i]; }
  if (!any || all) return;
  for (let pass = 0; pass < 64; pass++) {
    const next = known.slice(); let changed = false;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (known[i]) continue;
      let r = 0, g = 0, b = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || xx < 0 || yy >= h || xx >= w) continue;
        const j = yy * w + xx; if (!known[j]) continue;
        r += data[j * 4]; g += data[j * 4 + 1]; b += data[j * 4 + 2]; n++;
      }
      if (n) { data[i * 4] = r / n; data[i * 4 + 1] = g / n; data[i * 4 + 2] = b / n; next[i] = 1; changed = true; }
    }
    known.set(next); if (!changed) break;
  }
}

// full engine pipeline: alpha-aware, optional seamless (wrap-around) mode, tiled x4 upscale
async function engine(src, onProgress) {
  const W = src.width, H = src.height;
  const img = src.getContext("2d").getImageData(0, 0, W, H);
  let hasAlpha = false;
  for (let i = 3; i < img.data.length; i += 4) if (img.data[i] < 255) { hasAlpha = true; break; }
  const rgb = new ImageData(new Uint8ClampedArray(img.data), W, H);
  if (hasAlpha) bleed(rgb.data, W, H);
  for (let i = 3; i < rgb.data.length; i += 4) rgb.data[i] = 255;
  const P = state.tileable ? Math.min(16, W, H) : 0;
  const t = document.createElement("canvas"); t.width = W; t.height = H; t.getContext("2d").putImageData(rgb, 0, 0);
  const pc = document.createElement("canvas"); pc.width = W + 2 * P; pc.height = H + 2 * P;
  const pctx = pc.getContext("2d");
  for (const dy of [-1, 0, 1]) for (const dx of [-1, 0, 1]) pctx.drawImage(t, P + dx * W, P + dy * H);
  const up = await upscale(pc, onProgress);
  const out = document.createElement("canvas"); out.width = W * SCALE; out.height = H * SCALE;
  const octx = out.getContext("2d");
  octx.drawImage(up, P * SCALE, P * SCALE, out.width, out.height, 0, 0, out.width, out.height);
  if (hasAlpha) {  // alpha upscaled separately (smooth), then recombined
    const a = document.createElement("canvas"); a.width = out.width; a.height = out.height;
    const actx = a.getContext("2d"); actx.imageSmoothingQuality = "high"; actx.drawImage(src, 0, 0, out.width, out.height);
    const ad = actx.getImageData(0, 0, out.width, out.height).data, od = octx.getImageData(0, 0, out.width, out.height);
    for (let i = 3; i < od.data.length; i += 4) od.data[i] = ad[i];
    octx.putImageData(od, 0, 0);
  }
  return out;
}

function draw() {
  const L = $("cLeft"), R = $("cRight");
  if (!state.sr) return;
  for (const cv of [L, R]) { cv.width = state.sr.width; cv.height = state.sr.height; }
  const lctx = L.getContext("2d");
  if (state.mode === "truth" && state.truth) {
    lctx.drawImage(state.truth, 0, 0, L.width, L.height);
  } else {
    lctx.imageSmoothingEnabled = state.mode !== "nearest";
    lctx.imageSmoothingQuality = "high";
    lctx.drawImage(state.lr, 0, 0, L.width, L.height);
  }
  R.getContext("2d").drawImage(state.sr, 0, 0);
  R.style.clipPath = `inset(0 0 0 ${state.split * 100}%)`;
  $("divider").style.left = state.split * 100 + "%";
  $("compare").style.aspectRatio = `${state.sr.width} / ${state.sr.height}`;
  $("tagL").textContent = { bicubic: "Bicubic", nearest: "Original pixels", truth: "True HD" }[state.mode];
}

async function process(img, key, truthSrc) {
  $("dl").disabled = true; state.zip = null; $("dl").textContent = "Download HD PNG";
  state.lr = imgToCanvas(img);
  state.truth = truthSrc ? await loadImage(truthSrc) : null;
  $("seg").querySelector('[data-mode="truth"]').disabled = !state.truth;
  if (!state.truth && state.mode === "truth") setMode("bicubic");
  const t0 = performance.now();
  $("status").textContent = "Upscaling…";
  try {
    state.sr = await engine(state.lr, (p) => { $("status").textContent = `Upscaling… ${Math.round(p * 100)}%`; });
    const ms = performance.now() - t0;
    $("status").textContent = `${state.lr.width}×${state.lr.height} → ${state.sr.width}×${state.sr.height} in ${(ms / 1000).toFixed(1)} s, in your browser`;
    if (key) state.results[key] = { ms, w: state.sr.width, h: state.sr.height };
    $("dl").disabled = false;
    draw();
  } catch (e) {
    $("status").innerHTML = `<span class="err">Error: ${e.message}</span>`; console.error(e);
  }
}

function loadImage(src) {
  return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
}

function setMode(m) {
  state.mode = m;
  $("seg").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === m)));
  draw();
}

function bindSlider() {
  const box = $("compare");
  const move = (e) => {
    const r = box.getBoundingClientRect();
    state.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    draw();
  };
  box.addEventListener("pointerdown", (e) => { box.setPointerCapture(e.pointerId); move(e); });
  box.addEventListener("pointermove", (e) => { if (e.buttons) move(e); });
}

async function main() {
  state.meta = await (await fetch("meta.json")).json();
  const m = state.meta;
  $("statsNote").textContent = `${m.stats.n} held-out game textures never used in training, damaged like old compressed assets (blur, downscale, noise, JPEG), upscaled 4×.`;
  $("stats").innerHTML = m.stats.rows.map((r) => `<tr class="${r.mine ? "mine" : ""}"><td>${r.method}</td><td class="num">${r.psnr.toFixed(2)} dB</td><td class="num">${r.lpips.toFixed(3)}</td><td class="num">${r.size}</td></tr>`).join("");
  state.session = await ort.InferenceSession.create("models/" + m.model, { executionProviders: ["wasm"] });
  $("seg").querySelectorAll("button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  bindSlider();
  $("dl").onclick = () => {
    const a = document.createElement("a");
    if (state.zip) { a.download = "pixelforge_x4.zip"; a.href = URL.createObjectURL(state.zip); }
    else { a.download = "pixelforge_x4.png"; a.href = state.sr.toDataURL("image/png"); }
    a.click();
  };
  $("tileable").onchange = (e) => {
    state.tileable = e.target.checked;
    const sel = $("thumbs").querySelector('[aria-pressed="true"]'); if (sel) sel.click();
  };
  $("file").onchange = async (e) => {
    const files = [...e.target.files].filter((f) => f.size <= 20e6);
    if (!files.length) { $("status").textContent = "Image too large (max 20 MB)"; return; }
    $("thumbs").querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
    if (files.length === 1) { await process(await loadImage(URL.createObjectURL(files[0])), null, null); return; }
    // batch: upscale every file, then offer one ZIP
    $("dl").disabled = true;
    const zip = new JSZip(); state.batch = [];
    for (const [i, f] of files.entries()) {
      $("status").textContent = `Batch ${i + 1}/${files.length}: ${f.name}`;
      state.lr = imgToCanvas(await loadImage(URL.createObjectURL(f)), 1024);
      state.sr = await engine(state.lr);
      const blob = await new Promise((r) => state.sr.toBlob(r, "image/png"));
      zip.file(f.name.replace(/\.[^.]+$/, "") + "_x4.png", blob);
      state.batch.push({ name: f.name, w: state.sr.width, h: state.sr.height });
      state.truth = null; draw();
    }
    state.zip = await zip.generateAsync({ type: "blob" });
    $("status").textContent = `Batch done: ${files.length} textures upscaled 4×, in your browser`;
    $("dl").textContent = `Download all (${files.length}) as ZIP`; $("dl").disabled = false;
  };
  const box = $("thumbs");
  m.samples.forEach((s) => {
    const b = document.createElement("button"); b.className = "thumb"; b.type = "button";
    b.innerHTML = `<img src="${s.lr}" alt="${s.name}">${s.name}`;
    b.onclick = async () => {
      box.querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      await process(await loadImage(s.lr), s.lr, s.hr);
    };
    box.appendChild(b);
  });
  const q = new URLSearchParams(location.search).get("sample");
  const i = Math.max(0, m.samples.findIndex((s) => s.slug === q));
  box.children[i].click();
}
main().catch((e) => { $("status").innerHTML = `<span class="err">Failed to load: ${e.message}</span>`; });
