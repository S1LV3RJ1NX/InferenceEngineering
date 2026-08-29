"""Copy a post into the portfolio's content collection, rewriting what differs.

The README in this repo is the source of truth and has to stay readable on
GitHub, so anything the website renders differently lives here as a rewrite
rather than in the markdown:

  - image paths become site-absolute
  - sibling-post links become site routes
  - `draft: true` is cleared, since publishing is this script's job
  - `<!--walkthrough-->` becomes an iframe, so GitHub readers get the plain
    link that follows it and site readers get the embedded widget

Figures and the standalone walkthrough page are copied across too, so this repo
stays the single source for everything the post needs.

Usage: uv run python blogs/assets/sync.py inference-03-kernels-and-flashattention
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SITE = Path("/Users/prathamesh/portfolio/site")

WALKTHROUGH = (
    '<div class="not-prose" style="width:min(56rem,92vw);margin-left:50%;'
    'transform:translateX(-50%);margin-top:1.75rem;margin-bottom:0.5rem">'
    '<iframe src="/blogs/{slug}/walkthrough.html" title="Interactive FlashAttention walkthrough" '
    'loading="lazy" style="width:100%;height:780px;border:1px solid #E5E5E5;'
    'border-radius:8px;background:#FAFAFA"></iframe></div>'
)


def sync(slug: str) -> Path:
    src = REPO / "blogs" / slug / "README.md"
    if not src.exists():
        raise SystemExit(f"no post at {src}")

    text = src.read_text()
    text = text.replace("./images/", f"/blogs/{slug}/")
    text = re.sub(r"\.\./([a-z0-9-]+)/README\.md", r"/blogs/\1", text)
    text = re.sub(r"^draft: true$", "draft: false", text, flags=re.MULTILINE)
    text = text.replace("<!--walkthrough-->", WALKTHROUGH.format(slug=slug))

    dst = SITE / "src" / "content" / "blog" / f"{slug}.md"
    dst.write_text(text)

    assets = SITE / "public" / "blogs" / slug
    assets.mkdir(parents=True, exist_ok=True)
    for image in sorted((REPO / "blogs" / slug / "images").glob("*")):
        shutil.copy2(image, assets / image.name)

    walkthrough = REPO / "blogs" / slug / "walkthrough.html"
    if walkthrough.exists():
        shutil.copy2(walkthrough, assets / "walkthrough.html")

    return dst


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: sync.py <slug>")
    out = sync(sys.argv[1])
    print(f"synced -> {out}")
