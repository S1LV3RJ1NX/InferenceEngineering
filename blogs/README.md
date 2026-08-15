# Inference Engineering Blogs

Long-form, intuition-first writeups on the engineering that sits between a model checkpoint and
a chat window. Each post takes you from intuition → math → a worked number → the engineering
consequence. The markdown here is the **source of truth**; posts are synced to the portfolio for
hosting (see [`BLOG_RULES.md`](./BLOG_RULES.md) §7).

> **The series throughline:** *Arithmetic is cheap. Moving bytes is expensive. Every technique in
> this series is a way of buying back bandwidth.*

## Posts

| # | Title | Source | Live |
|---|---|---|---|
| 01 | The Memory Wall: Where the 30 Milliseconds Actually Go | [Source](./inference-01-memory-wall/) | [Read](https://prathameshsaraf.com/blogs/inference-01-memory-wall/) |
| 02 | Inside the GPU: Why Moving a Byte Costs More Than Multiplying One | [Source](./inference-02-inside-the-gpu/) | [Read](https://prathameshsaraf.com/blogs/inference-02-inside-the-gpu/) |

## Writing a new post

Read [`BLOG_RULES.md`](./BLOG_RULES.md): it has the 4-part spine, the color semantics, math and
code conventions, the figure pipeline, publishing steps, and a copy-paste skeleton.

## Figures

All code-generated figures come from one reproducible script:

```bash
uv run python blogs/assets/figures.py            # all posts
uv run python blogs/assets/figures.py --blog 01  # one post
```

- `ai-*.png`: AI-generated conceptual illustrations (no precise labels).
- `fig-*.svg`: code-generated diagrams with numbers, axes, or math (vector, no JavaScript).

Figures obey one series-wide color code: **warm terracotta `#C8421A` means compute, cool teal
`#1F7A8C` means memory and bandwidth.** Never use those two colors decoratively.

## Reference frames

`reference/frames/` (gitignored) holds stills pulled from the source videos, used as design
reference only. Regenerate with `yt-dlp` plus `ffmpeg`; see the plan notes in
[`BLOG_RULES.md`](./BLOG_RULES.md) §1c on attribution.
