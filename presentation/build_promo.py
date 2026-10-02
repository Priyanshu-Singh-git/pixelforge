"""Promo image (1920x1080): hook lines + PixelForge Engine selling points, every number from eval/.

  python presentation/build_promo.py  ->  presentation/upwork/pixelforge_engine_promo.png
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "upwork"


def b64(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def main():
    rows = [json.loads(l) for l in (ROOT / "eval" / "results.jsonl").read_text().splitlines()]
    real = {r["method"]: r for r in rows if r["condition"] == "real"}
    mine, bic, ref = real["PixelForge RRDB-S, GAN (mine)"], real["Bicubic"], real["Real-ESRGAN x4plus (pretrained)"]
    tk = json.loads((ROOT / "eval" / "toolkit_results.json").read_text())
    lp = round(100 * (bic["lpips"] - mine["lpips"]) / bic["lpips"])
    small = round(ref["params_m"] / mine["params_m"])
    t = tk["tiling"]
    cards = [
        (f"{lp}%", "closer to real HD", "than plain resizing"),
        (f"{small}×", "smaller than Real-ESRGAN", "and closer to real HD on game art"),
        (f"{t['textures_with_visible_seam_normal']}→{t['textures_with_visible_seam_tileable']}", "tiling seams",
         f"out of {t['n']} repeating textures"),
        (f"−{round(tk['alpha']['reduction_pct'])}%", "halo on transparent edges", "foliage, decals, UI"),
    ]
    card_html = "".join(f'<div class="card"><div class="v">{v}</div><div class="l1">{a}</div><div class="l2">{b}</div></div>'
                        for v, a, b in cards)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
  * {{ box-sizing: border-box; margin: 0; }}
  body {{ width: 1920px; height: 1080px; background: #fff; font-family: "Segoe UI", sans-serif; color: #14532d; overflow: hidden; }}
  .wrap {{ position: absolute; inset: 0; padding: 78px 96px; display: grid; grid-template-columns: 1.08fr 0.92fr; gap: 64px; }}
  .kick {{ font-weight: 700; letter-spacing: .16em; color: #16a34a; font-size: 24px; }}
  h1 {{ font-family: "Segoe Script", cursive; font-size: 66px; line-height: 1.12; margin-top: 20px; color: #14532d; }}
  h1 span {{ color: #16a34a; }}
  .sub {{ font-size: 30px; margin-top: 26px; color: #166534; line-height: 1.35; }}
  .sub b {{ color: #15803d; }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin-top: 44px; }}
  .card {{ border: 2.5px solid #39ff14; border-radius: 34px; padding: 22px 28px; background: #f0fdf4; }}
  .v {{ font-size: 54px; font-weight: 800; color: #16a34a; line-height: 1; }}
  .l1 {{ font-size: 23px; font-weight: 700; margin-top: 10px; }}
  .l2 {{ font-size: 19px; color: #4b6b58; margin-top: 2px; }}
  .right {{ display: flex; flex-direction: column; justify-content: center; gap: 26px; }}
  .right img {{ width: 100%; border-radius: 36px; border: 3px solid #39ff14; }}
  .name {{ font-family: "Segoe Script", cursive; font-size: 46px; font-weight: 700; text-align: center; }}
  .name span {{ color: #16a34a; }}
  .bar {{ position: absolute; left: 96px; right: 96px; bottom: 44px; border-radius: 999px; background: #14532d; color: #fff;
          display: flex; justify-content: space-between; align-items: center; padding: 18px 40px; font-size: 24px; }}
  .bar b {{ color: #39ff14; }}
  .bar code {{ font-family: Consolas, monospace; font-size: 21px; color: #bbf7d0; }}
</style></head><body><div class="wrap">
  <div>
    <div class="kick">FREE FOR MODDERS · RUNS IN YOUR BROWSER</div>
    <h1>Your favourite old game.<br><span>Now in HD.</span></h1>
    <p class="sub">Upscale a <b>whole texture pack 4×</b> in minutes: seamless tiles stay seamless, transparent edges stay clean, nothing gets uploaded.</p>
    <div class="cards">{card_html}</div>
  </div>
  <div class="right">
    <div class="name">PixelForge <span>Engine</span></div>
    <img src="{b64(ROOT / 'assets' / 'figures' / 'hero_grid.png')}">
  </div>
</div>
<div class="bar"><span><b>Try it:</b> priyanshu-singh-git.github.io/pixelforge</span><code>pip install git+https://github.com/Priyanshu-Singh-git/pixelforge</code></div>
</body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        pg.set_content(html, wait_until="load")
        pg.screenshot(path=str(OUT / "pixelforge_engine_promo.png"))
        b.close()
    print("wrote", OUT / "pixelforge_engine_promo.png")


if __name__ == "__main__":
    main()
