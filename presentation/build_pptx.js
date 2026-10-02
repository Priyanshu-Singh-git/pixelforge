// Build deck.pptx (16:9, 13.333 x 7.5 in) from deck_data.json written by build_deck.py.
// Light theme with warm accents; buyer-first slides, engineering appendix.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const THEME = {
  name: "PixelForge", headFontFace: "Arial", bodyFontFace: "Calibri",
  colors: { dk1: "1F2937", lt1: "FFFFFF", dk2: "4B5563", lt2: "F6F5F4", accent1: "EA580C", accent2: "DC2626",
            accent3: "F59E0B", accent4: "B91C1C", accent5: "6B7280", accent6: "E5E7EB", hlink: "EA580C", folHlink: "B91C1C" },
};
const H = THEME.colors;

async function applyTheme(file, theme) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const part = "ppt/theme/theme1.xml";
  let xml = await zip.file(part).async("string");
  for (const [slot, hex] of Object.entries(theme.colors)) {
    xml = xml.replace(new RegExp("<a:" + slot + ">[\\s\\S]*?</a:" + slot + ">"), `<a:${slot}><a:srgbClr val="${hex}"/></a:${slot}>`);
  }
  xml = xml.replace(/<a:clrScheme name="[^"]*">/, `<a:clrScheme name="${theme.name}">`);
  zip.file(part, xml);
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "PixelForge: 4x game texture upscaler";
pres.author = "Priyanshu Singh";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, M = 0.6;

function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function fit(file, x, y, w, h) {
  const s = pngSize(file), r = Math.min(w / s.w, h / s.h), iw = s.w * r, ih = s.h * r;
  return { path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih };
}

pres.defineSlideMaster({ title: "COVER", background: { color: C.background1 }, objects: [] });
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 }, margin: [0.5, M, 0.6, M],
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: M, y: 0.42, w: 12.1, h: 0.4, fontFace: "Arial", fontSize: 13,
        bold: true, color: C.accent1, charSpacing: 2, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.82, w: 12.1, h: 0.95, fontFace: "Arial", fontSize: 34,
        bold: true, color: C.text1, margin: 0, valign: "top", align: "left" }, text: "" } },
    { text: { text: "PixelForge · 4× game texture upscaler", options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 11, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.1, y: 7.0, w: 0.63, h: 0.3, fontSize: 11, color: C.accent5, align: "right" },
});

function content(section, kicker, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(kicker.toUpperCase(), { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  return s;
}
function card(s, x, y, w, h, name, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 },
    line: { color: opts.hl ? C.accent1 : C.accent6, width: opts.hl ? 2 : 1 }, objectName: name });
}
function badge(s, x, y, n) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { type: "none" } });
  s.addText(String(n), { x, y, w: 0.5, h: 0.5, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);

// ================= 1. cover / thumbnail =================
pres.addSection({ title: "Hook" });
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Hook" });
  const img = 5.6, ix = W - img - 0.5, iy = (7.5 - img) / 2;
  s.addImage({ path: D.assets.hero, x: ix, y: iy, w: img, h: img, objectName: "Before/after hero",
    altText: "A game texture: left half bicubic upscale, right half PixelForge 4x" });
  const tw = ix - M - 0.35;
  s.addText("AI FOR GAME ARTISTS & MODDERS", { x: M, y: 1.2, w: tw, h: 0.45, fontFace: "Arial", fontSize: 18, bold: true,
    color: C.accent2, charSpacing: 3, margin: 0, isTextBox: true });
  s.addText("PixelForge", { x: M, y: 1.75, w: tw, h: 1.15, fontFace: "Arial", fontSize: 66, bold: true, color: C.text1,
    margin: 0, valign: "top", isTextBox: true, objectName: "Product name" });
  s.addText([{ text: "Old game textures in.", options: { color: C.text1, breakLine: true } },
             { text: "4× HD out.", options: { color: C.accent1 } }],
    { x: M, y: 3.1, w: tw, h: 1.7, fontFace: "Arial", fontSize: 30, bold: true, margin: 0, valign: "top", isTextBox: true, objectName: "USP" });
  let cx = M;
  ["4× upscale", "Trained on game art", "Runs in browser"].forEach((t) => {
    const cw = 0.1 * t.length + 0.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: 5.0, w: cw, h: 0.55, rectRadius: 0.27, fill: { color: C.background2 }, line: { color: C.accent1, width: 1.25 } });
    s.addText(t, { x: cx, y: 5.0, w: cw, h: 0.55, fontSize: 17, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    cx += cw + 0.15;
  });
  s.addNotes("0-8s: drag the before/after slider on an old texture. 'Same texture, 4x the detail.'");
}

// ================= 2. problem =================
pres.addSection({ title: "Problem" });
{
  const s = content("Problem", "The problem", "HD screens expose every old texture.");
  const items = [
    ["Remasters need thousands of textures", "Redrawing every wall, floor and prop by hand is the slowest, most expensive part of a remaster or HD mod."],
    ["Simple resizing just blurs", "Stretching a texture 4× with bicubic gives soft, muddy surfaces that look worse on a modern monitor."],
    ["Generic AI upscalers aren't tuned for games", "Tools trained on photos can smear tiling patterns or invent the wrong kind of detail."],
  ];
  const cw = (12.13 - 2 * 0.35) / 3;
  items.forEach(([h, p], i) => {
    const x = M + i * (cw + 0.35), y = 2.2;
    card(s, x, y, cw, 3.15, `Problem ${i + 1}`);
    badge(s, x + 0.35, y + 0.35, i + 1);
    s.addText(h, { x: x + 0.35, y: y + 1.0, w: cw - 0.7, h: 0.75, fontFace: "Arial", fontSize: 18, bold: true, color: C.text1, margin: 0, valign: "top", isTextBox: true });
    s.addText(p, { x: x + 0.35, y: y + 1.85, w: cw - 0.7, h: 1.2, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText([{ text: "PixelForge does the first pass for you: ", options: { color: C.text2 } },
             { text: "seconds per texture, in your browser.", options: { color: C.accent1, bold: true } }],
    { x: M, y: 5.8, w: 12.1, h: 0.7, fontSize: 24, margin: 0, isTextBox: true });
  s.addNotes("Remastering means redrawing thousands of textures; resizing blurs; generic AI isn't tuned for game art.");
}

// ================= 3. demo =================
pres.addSection({ title: "Solution" });
{
  const s = content("Solution", "Try it yourself", "Drop in a texture. Drag the slider.");
  const box = { x: M, y: 1.95, w: 7.6, h: 4.85 };
  card(s, box.x, box.y, box.w, box.h, "Screenshot frame");
  s.addImage({ ...fit(D.assets.shot, box.x + 0.15, box.y + 0.15, box.w - 0.3, box.h - 0.3), objectName: "Demo screenshot",
    altText: "PixelForge web app with a before/after slider" });
  const pts = [["Before/after slider", "Compare against bicubic, the original pixels, or the true HD texture."],
               ["Runs in your browser", "No upload, no server, no install. Your art never leaves your machine."],
               ["Download the HD file", `Large images are processed in tiles; get a 4× PNG in seconds (${D.browser_s.toFixed(1)} s for a 512×512 result in a laptop browser).`]];
  const x = M + box.w + 0.35, w = W - M - x;
  pts.forEach(([h, p], i) => {
    const y = 1.95 + i * 1.7;
    card(s, x, y, w, 1.45, `Point ${i + 1}`);
    badge(s, x + 0.25, y + 0.25, i + 1);
    s.addText(h, { x: x + 0.95, y: y + 0.2, w: w - 1.15, h: 0.45, fontFace: "Arial", fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(p, { x: x + 0.95, y: y + 0.65, w: w - 1.15, h: 0.72, fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("Live demo: pick a texture, drag the slider, toggle 'vs True HD', download the PNG.");
}

// ================= 4. results in plain numbers =================
pres.addSection({ title: "Results" });
{
  const s = content("Results", "Results", "Looks like real HD. Small enough for a browser.");
  const stats = D.stats;  // [value, line 1, line 2], computed from eval results in build_deck.py
  const cw = (12.13 - 2 * 0.35) / 3;
  stats.forEach(([v, c1, c2], i) => {
    const x = M + i * (cw + 0.35), y = 1.95;
    card(s, x, y, cw, 1.95, `Stat ${i + 1}`);
    s.addText(v, { x: x + 0.3, y: y + 0.2, w: cw - 0.6, h: 0.85, fontFace: "Arial", fontSize: 38, bold: true, color: C.accent1, margin: 0, isTextBox: true });
    s.addText([{ text: c1, options: { bold: true, color: C.text1, breakLine: true } }, { text: c2, options: { color: C.accent5 } }],
      { x: x + 0.3, y: y + 1.08, w: cw - 0.6, h: 0.8, fontSize: 13.5, margin: 0, valign: "top", isTextBox: true });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 4.15, w: 12.13, h: 2.65, rectRadius: 0.12, fill: { color: "FFFFFF" }, line: { color: C.accent6, width: 1 } });
  s.addImage({ ...fit(D.assets.zoom_row, M + 0.1, 4.2, 11.93, 2.55), objectName: "Zoom comparison", altText: "Zoomed crops: input, bicubic, Real-ESRGAN, PixelForge, true HD" });
  s.addNotes("One number that proves it works: closer to the real HD texture than bicubic, at a fraction of the big model's size.");
}

// ================= 5. how it works =================
{
  const s = content("Results", "How it works", "Trained to undo old-texture damage.");
  const steps = [["Collect game textures", `${D.n_train} free (CC0) textures: bricks, wood, metal, ground, fabric, stone.`],
                 ["Damage them on purpose", "Blur, downscale, noise and JPEG compression, applied at random, the way old assets get degraded."],
                 ["Learn to undo it", "A compact generator learns to rebuild the missing detail at 4×, sharpened with a GAN stage."]];
  const gap = 0.55, bw = (12.13 - 2 * gap) / 3, y = 2.25, bh = 2.9;
  steps.forEach(([h, p], i) => {
    const x = M + i * (bw + gap);
    card(s, x, y, bw, bh, `Step ${i + 1}`, { hl: i === 2 });
    badge(s, x + 0.3, y + 0.3, i + 1);
    s.addText(h, { x: x + 0.3, y: y + 1.0, w: bw - 0.6, h: 0.5, fontFace: "Arial", fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(p, { x: x + 0.3, y: y + 1.6, w: bw - 0.6, h: 1.2, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true });
    if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.12, y: y + bh / 2 - 0.15, w: gap - 0.24, h: 0.3, fill: { color: C.accent1 }, line: { type: "none" } });
  });
  s.addText([{ text: "For engineers: ", options: { bold: true, color: C.text1 } },
             { text: `compact ESRGAN-style RRDB generator (${D.params_m.toFixed(1)}M params) trained from scratch in PyTorch: L1 stage, then VGG-perceptual + U-Net-discriminator GAN stage with EMA (Real-ESRGAN recipe), exported to ONNX. Benchmarks are in the appendix.`, options: { color: C.text2 } }],
    { x: M, y: 5.6, w: 12.1, h: 0.9, fontSize: 14, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Technical peek: the training trick is damaging clean textures on purpose so the model learns to undo it.");
}

// ================= 6. gallery =================
{
  const s = content("Results", "Real results", "Representative, not cherry-picked.");
  s.addText("Median-improvement textures, different materials, zoomed. PixelForge clears noise and JPEG blocks; detail the tiny input no longer has can't be fully recovered.",
    { x: M, y: 1.65, w: 12.1, h: 0.4, fontSize: 14, color: C.text2, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 2.2, w: 12.13, h: 4.6, rectRadius: 0.12, fill: { color: "FFFFFF" }, line: { color: C.accent6, width: 1 } });
  s.addImage({ ...fit(D.assets.zoom, M + 0.15, 2.3, 11.83, 4.4), objectName: "Gallery", altText: "Zoomed comparisons for three textures" });
  s.addNotes("Run three textures live; toggle 'vs True HD' to show it's honest.");
}

// ================= 7. working together =================
pres.addSection({ title: "Working together" });
{
  const s = content("Working together", "How we'd work together", "Start with a test pack. Scale to the whole library.");
  const steps = [["Send 10–20 textures", "A representative sample of your game's textures or asset pack."],
                 ["Test pack back", "I upscale them, optionally fine-tune on your art style, and send before/after comparisons."],
                 ["Batch the library", "Folder in, folder out, file names kept, ready to drop into your mod or engine."]];
  const cw = (12.13 - 2 * 0.35) / 3;
  steps.forEach(([h, p], i) => {
    const x = M + i * (cw + 0.35), y = 2.2;
    card(s, x, y, cw, 2.75, `Work step ${i + 1}`);
    s.addText(`STEP ${i + 1}`, { x: x + 0.35, y: y + 0.3, w: 2, h: 0.35, fontSize: 13, bold: true, color: C.accent1, charSpacing: 2, margin: 0, isTextBox: true });
    s.addText(h, { x: x + 0.35, y: y + 0.7, w: cw - 0.7, h: 0.5, fontFace: "Arial", fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(p, { x: x + 0.35, y: y + 1.3, w: cw - 0.7, h: 1.35, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  });
  card(s, M, 5.25, 12.13, 1.45, "What you get");
  s.addText([{ text: "What you get:  ", options: { bold: true, color: C.text1 } },
             { text: "the HD texture set · a batch tool you can rerun on new assets · optional model fine-tuned on your art style · before/after report", options: { color: C.text2 } }],
    { x: M + 0.35, y: 5.4, w: 11.4, h: 1.15, fontSize: 16, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("Lower the risk: a small test pack first.");
}

// ================= 8. CTA =================
pres.addSection({ title: "Call to action" });
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Call to action" });
  s.addText("NEXT STEP", { x: M, y: 0.9, w: 6, h: 0.4, fontFace: "Arial", fontSize: 14, bold: true, color: C.accent2, charSpacing: 2, margin: 0, isTextBox: true });
  s.addText([{ text: "Send me 10 of your textures.", options: { color: C.text1, breakLine: true } },
             { text: "I'll send them back in HD.", options: { color: C.accent1 } }],
    { x: M, y: 1.35, w: 12.1, h: 1.9, fontFace: "Arial", fontSize: 42, bold: true, margin: 0, valign: "top", isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 3.55, w: 2.9, h: 2.9, rectRadius: 0.12, fill: { color: "FFFFFF" }, line: { color: C.accent6, width: 1 } });
  s.addImage({ path: D.assets.qr, x: M + 0.15, y: 3.7, w: 2.6, h: 2.6, objectName: "QR code", altText: "QR code to the live demo" });
  const links = [["Try the live demo", D.demo_url]];
  if (D.video_url) links.push(["Watch the walkthrough", D.video_url]);
  links.push(["Code and full benchmark", D.repo_url]);
  const runs = [];
  links.forEach(([lab, url], i) => {
    runs.push({ text: lab, options: { bold: true, color: C.text1, breakLine: true } });
    runs.push({ text: url.replace(/^https:\/\//, ""), options: { color: C.accent1, hyperlink: { url }, breakLine: i < links.length - 1 } });
    if (i < links.length - 1) runs.push({ text: " ", options: { fontSize: 6, breakLine: true } });
  });
  s.addText(runs, { x: M + 3.4, y: 3.55, w: 8.6, h: 2.9, fontSize: 18, margin: 0, valign: "top", isTextBox: true });
  s.addText("Priyanshu Singh · Freelance AI / ML Engineer · Trained on Poly Haven CC0 textures. Commercial use of the code needs a licence; see the repo.",
    { x: M, y: 6.75, w: 12.1, h: 0.4, fontSize: 12, color: C.accent5, margin: 0, isTextBox: true });
  s.addNotes("The live demo is linked. Send me ten textures and I'll send them back in HD.");
}

// ================= 9. appendix =================
pres.addSection({ title: "Appendix" });
{
  const s = content("Appendix", "Appendix · for engineers", "Benchmarked, not guessed.");
  s.addText(`Held-out test set: ${D.n_test} game textures, realistic degradation, ×4. PSNR/SSIM higher is better; LPIPS lower is better.`,
    { x: M, y: 1.65, w: 12.1, h: 0.4, fontSize: 13, color: C.text2, margin: 0, isTextBox: true });
  const hdr = ["Method", "Params", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "GPU ms"].map((t) => ({ text: t, options: { bold: true, color: C.accent1, fill: { color: C.background2 } } }));
  const rows = D.table.map((r) => [r.method, r.params, f2(r.psnr), f3(r.ssim), f3(r.lpips), r.gpu_ms].map((t) => ({ text: String(t),
    options: { color: r.mine ? C.text1 : C.text2, bold: r.mine, fill: { color: C.background1 } } })));
  s.addTable([hdr, ...rows], { x: M, y: 2.15, w: 12.13, colW: [5.33, 1.3, 1.3, 1.3, 1.6, 1.3], fontSize: 12.5, fontFace: "Calibri",
    border: { type: "solid", pt: 0.75, color: H.accent6 }, rowH: 0.34, margin: 0.06, objectName: "Benchmark table" });
  s.addText(D.findings.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < D.findings.length - 1 } })),
    { x: M, y: 5.05, w: 12.1, h: 1.7, fontSize: 13.5, color: C.text2, margin: 0, valign: "top", paraSpaceAfter: 4, isTextBox: true });
  s.addNotes("For technical buyers: architectures compared, L1 vs GAN, and the official Real-ESRGAN models as references.");
}

(async () => {
  const out = path.join(__dirname, "deck.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
