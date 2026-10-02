// PixelForge browser demo: x4 super-resolution with ONNX Runtime Web, tiled, fully client-side.
const $ = (id) => document.getElementById(id);
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
const TILE = 128, PAD = 8, SCALE = 4, MAX_SIDE = 512;

const state = { meta: null, session: null, lr: null, sr: null, truth: null, mode: "bicubic", split: 0.5, results: {} };
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
  $("dl").disabled = true;
  state.lr = imgToCanvas(img);
  state.truth = truthSrc ? await loadImage(truthSrc) : null;
  $("seg").querySelector('[data-mode="truth"]').disabled = !state.truth;
  if (!state.truth && state.mode === "truth") setMode("bicubic");
  const t0 = performance.now();
  $("status").textContent = "Upscaling…";
  try {
    state.sr = await upscale(state.lr, (p) => { $("status").textContent = `Upscaling… ${Math.round(p * 100)}%`; });
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
  $("dl").onclick = () => { const a = document.createElement("a"); a.download = "pixelforge_x4.png"; a.href = state.sr.toDataURL("image/png"); a.click(); };
  $("file").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    if (f.size > 20e6) { $("status").textContent = "Image too large (max 20 MB)"; return; }
    $("thumbs").querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
    await process(await loadImage(URL.createObjectURL(f)), null, null);
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
