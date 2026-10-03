// PixelForge Live: real-time x2 upscaling of a video source in the browser (WebGPU, wasm fallback).
const $ = (id) => document.getElementById(id);
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);

const S = { meta: null, sessions: {}, backend: "", model: "live", inH: 360, src: "demo", video: null, running: false,
            split: 0.5, times: [], frames: 0, demoT: 0, demoImgs: [] };
window.pflive = S; // exposed for automated checks

const grab = document.createElement("canvas"), gctx = grab.getContext("2d", { willReadFrequently: true });

async function session(name) {
  const key = name;
  if (S.sessions[key]) return S.sessions[key];
  const url = "models/" + S.meta.models[name].file;
  let sess, backend;
  try {
    if (!navigator.gpu) throw new Error("no WebGPU");
    sess = await ort.InferenceSession.create(url, { executionProviders: ["webgpu"] });
    backend = "WebGPU";
  } catch (e) {
    sess = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
    backend = `CPU (wasm, ${ort.env.wasm.numThreads} threads)`;
  }
  S.sessions[key] = { sess, backend };
  return S.sessions[key];
}

// built-in demo scene: slow pan + zoom across sample textures, so the page works without any input
function drawDemo(ctx, w, h) {
  S.demoT += 1 / 60;
  const imgs = S.demoImgs; if (!imgs.length) return;
  const cols = 3, tile = w / 2.2;
  const ox = -((S.demoT * 40) % (tile * cols)), oy = Math.sin(S.demoT * 0.4) * tile * 0.15 - tile * 0.2;
  for (let r = -1; r < 3; r++) for (let c = -1; c < cols * 2 + 1; c++) {
    const img = imgs[((c % imgs.length) + imgs.length + r * 2) % imgs.length];
    ctx.drawImage(img, ox + c * tile, oy + r * tile, tile + 1, tile + 1);
  }
}

function sourceFrame(w, h) {
  if (S.src === "demo") { drawDemo(gctx, w, h); return true; }
  const v = S.video; if (!v || v.readyState < 2) return false;
  const vw = v.videoWidth, vh = v.videoHeight, a = 16 / 9;
  let sw = vw, sh = vw / a; if (sh > vh) { sh = vh; sw = vh * a; }
  gctx.drawImage(v, (vw - sw) / 2, (vh - sh) / 2, sw, sh, 0, 0, w, h);
  return true;
}

async function loop() {
  if (!S.running) return;
  const h = S.inH, w = Math.round(h * 16 / 9);
  if (grab.width !== w || grab.height !== h) { grab.width = w; grab.height = h; }
  if (sourceFrame(w, h)) {
    const px = gctx.getImageData(0, 0, w, h).data, n = w * h, x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { x[i] = px[i * 4] / 255; x[n + i] = px[i * 4 + 1] / 255; x[2 * n + i] = px[i * 4 + 2] / 255; }
    const { sess, backend } = await session(S.model);
    $("backend").textContent = backend; S.backend = backend;
    const t0 = performance.now();
    const out = await sess.run({ lr: new ort.Tensor("float32", x, [1, 3, h, w]) });
    const y = out.sr.data;
    const ms = performance.now() - t0;
    const W = w * 2, H = h * 2, N = W * H, img = new ImageData(W, H);
    for (let i = 0; i < N; i++) {
      img.data[i * 4] = y[i] * 255; img.data[i * 4 + 1] = y[N + i] * 255; img.data[i * 4 + 2] = y[2 * N + i] * 255; img.data[i * 4 + 3] = 255;
    }
    const co = $("cOut"), cb = $("cBase");
    if (co.width !== W) { co.width = cb.width = W; co.height = cb.height = H; }
    co.getContext("2d").putImageData(img, 0, 0);
    const bctx = cb.getContext("2d"); bctx.imageSmoothingEnabled = true; bctx.imageSmoothingQuality = "low";
    bctx.drawImage(grab, 0, 0, W, H);
    S.times.push(ms); if (S.times.length > 30) S.times.shift();
    S.frames++;
    const avg = S.times.reduce((a, b) => a + b, 0) / S.times.length;
    $("ms").textContent = avg.toFixed(1); $("fps").textContent = (1000 / avg).toFixed(1);
    S.stats = { ms: avg, fps: 1000 / avg, w, h, frames: S.frames, backend };
  }
  requestAnimationFrame(loop);
}

async function setSource(src, file) {
  document.querySelectorAll(".bar [data-src]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.src === src)));
  if (S.video && S.video.srcObject) S.video.srcObject.getTracks().forEach((t) => t.stop());
  S.video = null; S.src = src; $("status").textContent = "";
  try {
    if (src === "screen" || src === "camera") {
      const stream = src === "screen" ? await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30 } })
                                      : await navigator.mediaDevices.getUserMedia({ video: true });
      S.video = document.createElement("video"); S.video.muted = true; S.video.srcObject = stream; await S.video.play();
    } else if (src === "file" && file) {
      S.video = document.createElement("video"); S.video.muted = true; S.video.loop = true; S.video.src = URL.createObjectURL(file); await S.video.play();
    }
  } catch (e) { $("status").innerHTML = `<span class="err">${e.message}</span>`; S.src = "demo"; }
}

function bindSplit() {
  const st = $("stage");
  const mv = (e) => { const r = st.getBoundingClientRect(); S.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    $("cOut").style.clipPath = `inset(0 0 0 ${S.split * 100}%)`; $("divider").style.left = S.split * 100 + "%"; };
  st.addEventListener("pointerdown", (e) => { st.setPointerCapture(e.pointerId); mv(e); });
  st.addEventListener("pointermove", (e) => { if (e.buttons) mv(e); });
}

async function main() {
  S.meta = await (await fetch("live_meta.json")).json();
  $("pLive").textContent = S.meta.models.live.params.toLocaleString("en-US");
  $("pLite").textContent = S.meta.models.lite.params.toLocaleString("en-US");
  $("speed").innerHTML = S.meta.speed.map((r) => `<tr><td>${r.device}</td><td class="n">${r.p360}</td><td class="n">${r.p540}</td></tr>`).join("");
  $("speedNote").textContent = S.meta.speed_note;
  S.demoImgs = await Promise.all(S.meta.demo.map((src) => new Promise((res) => { const i = new Image(); i.onload = () => res(i); i.src = src; })));
  const q = new URLSearchParams(location.search);
  if (q.get("model")) { S.model = q.get("model"); $("model").value = S.model; }
  if (q.get("res")) { S.inH = +q.get("res"); $("res").value = q.get("res"); }
  $("res").onchange = (e) => { S.inH = +e.target.value; S.times = []; };
  $("model").onchange = (e) => { S.model = e.target.value; S.times = []; };
  document.querySelectorAll(".bar button[data-src]").forEach((b) => { if (b.dataset.src !== "file") b.onclick = () => setSource(b.dataset.src); });
  $("vfile").onchange = (e) => e.target.files[0] && setSource("file", e.target.files[0]);
  bindSplit();
  S.running = true; loop();
}
main().catch((e) => { $("status").innerHTML = `<span class="err">Failed to load: ${e.message}</span>`; });
