// PixelForge Live (v2) - 5-slide BUSINESS deck. Green theme, script headings, curved cards.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));

const FIG = path.join(__dirname, "..", "assets", "figures");
const C = { bg: "FFFFFF", card: "F0FDF4", ink: "14532D", ink2: "166534", muted: "4B6B58", line: "86EFAC", neon: "39FF14", accent: "16A34A", dark: "0B2E1A" };
const THEME = { name: "PixelForge Neon", colors: { dk1: C.ink, lt1: C.bg, dk2: C.ink2, lt2: C.card, accent1: C.accent, accent2: C.neon, accent3: "4ADE80", accent4: "15803D", accent5: C.muted, accent6: C.line, hlink: "15803D", folHlink: C.ink2 } };
const SCRIPT = "Segoe Script", SANS = "Segoe UI";

async function applyTheme(file) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  let xml = await zip.file("ppt/theme/theme1.xml").async("string");
  for (const [k, v] of Object.entries(THEME.colors)) xml = xml.replace(new RegExp("<a:" + k + ">[\\s\\S]*?</a:" + k + ">"), `<a:${k}><a:srgbClr val="${v}"/></a:${k}>`);
  zip.file("ppt/theme/theme1.xml", xml.replace(/<a:clrScheme name="[^"]*">/, `<a:clrScheme name="${THEME.name}">`));
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "PixelForge Live (v2)";
const W = 13.333, M = 0.6;
function png(f) { const b = fs.readFileSync(f); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function fit(f, x, y, w, h) { const s = png(f), r = Math.min(w / s.w, h / s.h); return { path: f, x: x + (w - s.w * r) / 2, y: y + (h - s.h * r) / 2, w: s.w * r, h: s.h * r }; }
function cover(f, x, y, w, h) { const s = png(f), r = Math.max(w / s.w, h / s.h), iw = s.w * r, ih = s.h * r; return { path: f, x, y, w, h, sizing: { type: "crop", x: (iw - w) / 2, y: (ih - h) / 2, w, h } }; }
function slide() { const s = pres.addSlide(); s.background = { color: C.bg }; s.addText("PixelForge Live", { x: M, y: 7.02, w: 5, h: 0.3, fontFace: SANS, fontSize: 10, color: C.muted, margin: 0 }); return s; }
function card(s, x, y, w, h, fill = C.card, neon = false) { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.22, fill: { color: fill }, line: neon ? { color: C.neon, width: 2.5 } : { type: "none" } }); }
function kicker(s, t) { s.addText(t, { x: M, y: 0.7, w: 11, h: 0.4, fontFace: SANS, fontSize: 15, bold: true, color: C.accent, charSpacing: 3, margin: 0 }); }
function title(s, a, b, y = 1.15, size = 46) { s.addText([{ text: a + " ", options: { color: C.ink } }, { text: b, options: { color: C.accent } }], { x: M, y, w: 12.1, h: 1.1, fontFace: SCRIPT, fontSize: size, bold: true, margin: 0 }); }

// ============ 1. COVER ============
{ const s = slide();
  kicker(s, "AI GAME UPSCALER · DESKTOP APP");
  s.addText([{ text: "PixelForge ", options: { color: C.ink } }, { text: "Live", options: { color: C.accent } }], { x: M, y: 1.2, w: 12, h: 1.5, fontFace: SCRIPT, fontSize: 76, bold: true, margin: 0 });
  s.addText([{ text: "Play your games in crisp HD — ", options: { color: C.ink } }, { text: "even on an old GPU.", options: { color: C.ink, highlight: C.neon } }],
    { x: M, y: 2.9, w: 12, h: 0.6, fontFace: SANS, fontSize: 25, bold: true, margin: 0 });
  card(s, M, 3.85, W - 2 * M, 3.05, C.dark, true);
  s.addImage(fit(path.join(FIG, "live_app_shot.png"), M + 0.18, 3.97, W - 2 * M - 0.36, 2.8));
  s.addText("One click. Any game. No settings to tweak.", { x: M, y: 3.5, w: 12, h: 0.33, fontFace: SANS, fontSize: 14, color: C.muted, margin: 0 });
}

// ============ 2. PROBLEM ============
{ const s = slide();
  kicker(s, "THE PROBLEM");
  title(s, "Blurry games on", "modest PCs", 1.15, 44);
  const items = [["Old or budget GPUs", "Millions of gamers run laptops and entry cards that can't drive sharp, high-res visuals."],
                 ["Upscaling is clunky", "Built-in options are per-game, hard to find, or simply absent in older titles."],
                 ["The best tool costs money", "The popular desktop upscaler is paid and uses simple filters, not a trained AI model."]];
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  items.forEach(([h, p], i) => { const x = M + i * (cw + 0.3), y = 2.5; card(s, x, y, cw, 3.3, C.card, false);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.3, y: y + 0.3, w: 0.55, h: 0.55, rectRadius: 0.1, fill: { color: C.neon }, line: { type: "none" } });
    s.addText(`${i + 1}`, { x: x + 0.3, y: y + 0.3, w: 0.55, h: 0.55, fontFace: SANS, fontSize: 22, bold: true, color: C.ink, align: "center", valign: "middle", margin: 0 });
    s.addText([{ text: h, options: { bold: true, breakLine: true } }, { text: p, options: { color: C.ink2 } }], { x: x + 0.3, y: y + 1.15, w: cw - 0.6, h: 2, fontFace: SANS, fontSize: 15, color: C.ink, margin: 0, valign: "top", paraSpaceAfter: 6 }); });
  s.addText([{ text: "PixelForge Live fixes all three: ", options: { color: C.ink, bold: true } }, { text: "free, one-click, and AI-powered.", options: { color: C.accent, bold: true } }],
    { x: M, y: 6.1, w: 12, h: 0.5, fontFace: SANS, fontSize: 18, margin: 0 });
}

// ============ 3. SOLUTION ============
{ const s = slide();
  kicker(s, "THE SOLUTION");
  title(s, "Sharper games,", "instantly", 1.1, 46);
  card(s, M, 2.3, 6.9, 4.5, C.dark, true);
  s.addImage(fit(path.join(FIG, "live_app_shot.png"), M + 0.15, 2.45, 6.6, 4.2));
  const pts = [["Works with any game", "Launch your game, launch PixelForge Live, play. No per-game configuration."],
               ["Trained on game art", "An AI model that understands textures and edges — not just a blur-and-stretch filter."],
               ["Runs on your GPU", "Smooth, real-time upscaling. Your game and your data stay on your machine."]];
  const x = M + 7.3, cw = W - M - x;
  pts.forEach(([h, p], i) => { const y = 2.4 + i * 1.5; card(s, x, y, cw, 1.3, C.card, false);
    s.addText([{ text: h, options: { bold: true, breakLine: true } }, { text: p, options: { color: C.ink2 } }], { x: x + 0.3, y: y + 0.17, w: cw - 0.6, h: 1, fontFace: SANS, fontSize: 14, color: C.ink, margin: 0, valign: "top" }); });
}

// ============ 4. WHY BETTER ============
{ const s = slide();
  kicker(s, "WHY IT STANDS OUT");
  title(s, "Smoother AND", "smarter", 1.1, 46);
  // comparison cards
  const cols = [
    ["Full HD, 60+ FPS", `${D.lite_fps} FPS at 1080p on a laptop RTX 3050 — fluid, not a slideshow.`, true],
    ["Real AI model", "Trained on game textures; keeps text and edges clean where simple filters smear."],
    ["Free & open", "The leading alternative is paid. PixelForge Live is free and the code is open."]];
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  cols.forEach(([h, p, hl], i) => { const x = M + i * (cw + 0.3), y = 2.4; card(s, x, y, cw, 2.1, C.card, !!hl);
    s.addText(h, { x: x + 0.3, y: y + 0.25, w: cw - 0.6, h: 0.8, fontFace: SANS, fontSize: 18, bold: true, color: C.accent, margin: 0, valign: "top" });
    s.addText(p, { x: x + 0.3, y: y + 1.0, w: cw - 0.6, h: 1, fontFace: SANS, fontSize: 13.5, color: C.ink2, margin: 0, valign: "top" }); });
  // proof strip
  card(s, M, 4.75, W - 2 * M, 2.0, C.card);
  s.addImage(fit(path.join(FIG, "live_compare.png"), M + 0.25, 4.9, W - 2 * M - 0.5, 1.5));
  s.addText("Same frame, same moment — PixelForge (right) keeps the HUD text and detail crisper than a plain filter (middle).", { x: M, y: 6.45, w: 12, h: 0.3, fontFace: SANS, fontSize: 12, color: C.muted, margin: 0, align: "center" });
}

// ============ 5. CTA ============
{ const s = slide();
  kicker(s, "GET IT");
  s.addText([{ text: "Try it, or let's ", options: { color: C.ink } }, { text: "build your version", options: { color: C.accent } }], { x: M, y: 1.2, w: 12, h: 1.3, fontFace: SCRIPT, fontSize: 52, bold: true, margin: 0 });
  const cards = [["For gamers", "Free download, run it on any offline game. One click to sharper visuals."],
                 ["For studios & devs", "Want AI upscaling tuned to your game's art, or built into your engine? I take that on."],
                 ["For me", "Freelance AI / ML engineer — real-time model deployment, from research to shipped app."]];
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  cards.forEach(([h, p], i) => { const x = M + i * (cw + 0.3), y = 2.9; card(s, x, y, cw, 2.6, i === 1 ? C.dark : C.card, i === 1);
    s.addText(h, { x: x + 0.3, y: y + 0.3, w: cw - 0.6, h: 0.6, fontFace: SANS, fontSize: 18, bold: true, color: i === 1 ? C.neon : C.accent, margin: 0 });
    s.addText(p, { x: x + 0.3, y: y + 1.0, w: cw - 0.6, h: 1.5, fontFace: SANS, fontSize: 14, color: i === 1 ? "CFEAD9" : C.ink2, margin: 0, valign: "top" }); });
  s.addText([{ text: "github.com/Priyanshu-Singh-git/pixelforge", options: { bold: true, color: C.accent } }, { text: "   ·   Priyanshu Singh, Freelance AI / ML Engineer", options: { color: C.muted } }],
    { x: M, y: 6.1, w: 12, h: 0.4, fontFace: SANS, fontSize: 15, margin: 0 });
}

(async () => { const out = path.join(__dirname, "deck_v2.pptx"); await pres.writeFile({ fileName: out }); await applyTheme(out); console.log("wrote", out); })();
