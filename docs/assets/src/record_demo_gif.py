"""Record docs/assets/loci-demo.gif from the warehouse demo.

    pip install -e . fastapi "uvicorn[standard]" playwright pillow
    playwright install chromium
    uvicorn demo.app.main:app --port 8765 &
    python docs/assets/src/record_demo_gif.py

Drives the demo's guided steps (build memory, place + time search, similar
moments), captures frames with a fixed layout, and encodes a ~1 MB GIF.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from PIL import Image
from playwright.async_api import Page, async_playwright

URL = os.environ.get("LOCI_DEMO_URL", "http://localhost:8765/")
OUT = Path(__file__).resolve().parent.parent / "loci-demo.gif"
FPS = 6
GIF_WIDTH = 1000

# Hide recording-irrelevant chrome and pin the map so it doesn't move between frames.
CSS = """
.header,.map-legend,.canvas-hint{display:none!important}
.guide-panel h2,.guide-panel > p{display:none!important}
.guide-panel ~ *{display:none!important}
.canvas-wrap{justify-content:flex-end!important;align-items:flex-start!important;
  padding:30px 20px 20px!important}
.story-overlay{top:30px!important;left:20px!important;width:330px}
#rec-caption{position:absolute;left:20px;bottom:28px;width:330px;z-index:5;
  padding:16px 16px 14px;border-radius:12px;background:rgba(0,188,212,.12);
  border:1px solid rgba(0,188,212,.45);font-family:inherit}
#rec-caption .n{font-size:12px;letter-spacing:.18em;text-transform:uppercase;
  color:#00bcd4;margin-bottom:6px}
#rec-caption .t{font-size:19px;line-height:1.35;color:#fff;font-weight:bold}
"""

# (caption number, caption text, guide button to click, seconds to record, keep every Nth frame)
STEPS = [
    (
        "Step 1 / 3",
        "A robot patrols and LOCI remembers what it saw, where, and when.",
        "guide-build",
        14,
        2,
    ),
    (
        "Step 2 / 3",
        "Ask memory: what happened in this aisle in the last few seconds?",
        "guide-spatial",
        5,
        1,
    ),
    (
        "Step 3 / 3",
        "Find moments that looked like this one, nearby in space and time.",
        "guide-similar",
        5,
        1,
    ),
]

# Colours that must survive palette quantisation (robot, docks, anchors, accents).
KEY_COLORS = [
    "#ff5757", "#ff4444", "#e04848", "#5ad17b", "#4caf50", "#3e9b58", "#ffc857", "#ffc107",
    "#00bcd4", "#0a8fa3", "#ffffff", "#e0e0e0", "#b9f7ff", "#9c27b0", "#ce93d8", "#4a6fa5",
]  # fmt: skip


async def js_click(page: Page, element_id: str) -> None:
    # A JS click avoids Playwright scrolling the target into view mid-recording.
    await page.evaluate(f"document.getElementById('{element_id}').click()")


async def record(frame_dir: Path) -> list[Path]:
    kept: list[Path] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH"))
        page = await browser.new_page(viewport={"width": 1400, "height": 660})
        await page.goto(URL)
        await page.add_style_tag(content=CSS)
        await page.evaluate(
            """() => { const d = document.createElement('div'); d.id = 'rec-caption';
            d.innerHTML = '<div class=n></div><div class=t></div>';
            document.querySelector('.canvas-wrap').appendChild(d); }"""
        )
        await page.wait_for_timeout(800)
        await js_click(page, "btn-reset")
        await page.wait_for_timeout(800)
        n = 0
        for number, text, button, seconds, stride in STEPS:
            await page.evaluate(
                """([a, b]) => { const c = document.getElementById('rec-caption');
                c.querySelector('.n').textContent = a; c.querySelector('.t').textContent = b; }""",
                [number, text],
            )
            await js_click(page, button)
            for i in range(seconds * FPS):
                await page.evaluate("window.scrollTo(0, 0)")
                path = frame_dir / f"f{n:05d}.png"
                await page.screenshot(path=str(path))
                if i % stride == 0:
                    kept.append(path)
                n += 1
                await page.wait_for_timeout(1000 // FPS - 60)
        await browser.close()
    return kept


def encode(paths: list[Path]) -> None:
    frames = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        size = (GIF_WIDTH, round(im.height * GIF_WIDTH / im.width))
        frames.append(im.resize(size, Image.LANCZOS))
    height = frames[0].height
    picks = [int(len(frames) * k / 8) for k in range(8)] + [len(frames) - 1]
    sample = Image.new("RGB", (GIF_WIDTH, height * len(picks) + 200))
    for k, idx in enumerate(picks):
        sample.paste(frames[idx], (0, height * k))
    swatch = GIF_WIDTH // len(KEY_COLORS)
    for k, color in enumerate(KEY_COLORS):
        sample.paste(Image.new("RGB", (swatch, 200), color), (k * swatch, height * len(picks)))
    palette = sample.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    durations = [1000 // FPS] * len(quantized)
    durations[-1] = 2500  # hold the final frame
    quantized[0].save(
        OUT,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(quantized)} frames)")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        encode(asyncio.run(record(Path(tmp))))


if __name__ == "__main__":
    main()
