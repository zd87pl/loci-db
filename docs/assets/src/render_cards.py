"""Render the README banner and the GitHub social-preview card from card.html.

    pip install playwright pillow && playwright install chromium
    python docs/assets/src/render_cards.py

Writes docs/assets/loci-banner.png (1280x360 @2x) and
docs/assets/social-preview.png (1280x640, upload it under
Settings -> General -> Social preview). Edit the copy in card.html.
Set CHROMIUM_PATH to use a specific Chromium binary.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

SRC = Path(__file__).resolve().parent
ASSETS = SRC.parent

TARGETS = [
    # (mode, width, height, device scale, output)
    ("social", 1280, 640, 1, ASSETS / "social-preview.png"),
    ("banner", 1280, 360, 2, ASSETS / "loci-banner.png"),
]


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH"))
        for mode, width, height, scale, out in TARGETS:
            page = await browser.new_page(
                viewport={"width": width, "height": height}, device_scale_factor=scale
            )
            await page.goto(f"{(SRC / 'card.html').as_uri()}#mode={mode}")
            await page.evaluate("document.fonts.ready")
            await page.wait_for_timeout(400)
            await page.locator("#card").screenshot(path=str(out), omit_background=True)
            await page.close()
            # Lossless re-save: palette quantisation visibly bands the gradients.
            img = Image.open(out)
            (img.convert("RGB") if mode == "social" else img).save(out, optimize=True)
            print(f"wrote {out.relative_to(ASSETS.parent.parent)}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
