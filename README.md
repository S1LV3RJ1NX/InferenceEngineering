# Inference Engineering

Long-form, first-principles writeups on the engineering that sits between a model checkpoint
and a chat window: why a token takes 30 milliseconds, where those milliseconds actually go, and
what production systems do about it.

> **The throughline:** *Arithmetic is cheap. Moving bytes is expensive. Every technique in this
> series is a way of buying back bandwidth.*

Every post derives its numbers rather than quoting them, ships reproducible figures generated
from one script, and works each formula through against the same reference hardware (an NVIDIA
H100) and the same reference model (Llama-3-70B) so the numbers stay comparable across the
series.

## Posts

| # | Post | Read it |
|---|---|---|
| 01 | **The Memory Wall: Where the 30 Milliseconds Actually Go**<br/>Derives the H100's ridge point of 295 ops/byte, places prefill and decode on a roofline 2,000x apart, and computes the KV cache down to the byte. | [Live](https://prathameshsaraf.com/blogs/inference-01-memory-wall/) · [Source](./blogs/inference-01-memory-wall/) |
| 02 | **Inside the GPU: Why Moving a Byte Costs More Than Multiplying One**<br/>Opens the machine that creates the bottleneck: 270,000 threads, a four-level memory ladder, latency hiding, and the controlled experiment that proves inference is bandwidth bound. | [Live](https://prathameshsaraf.com/blogs/inference-02-inside-the-gpu/) · [Source](./blogs/inference-02-inside-the-gpu/) |

More coming: kernels, quantization, multi-GPU sharding, mixture-of-experts, production serving,
and speculative decoding.

## What is in this repo

```
blogs/
  BLOG_RULES.md          the spec: workflow, math and figure conventions, publishing
  README.md              series index
  assets/figures.py      one script that regenerates every fig-*.svg
  inference-NN-slug/
    README.md            the post, and the source of truth for it
    images/              ai-*.png heroes and fig-*.svg diagrams
```

The markdown here is the source of truth. The site at
[prathameshsaraf.com/blogs](https://prathameshsaraf.com/blogs/) hosts a synced copy.

## Reproducing the figures

Every diagram with a number in it is generated, not drawn, so it can be regenerated from
scratch and audited:

```bash
uv sync
uv run python blogs/assets/figures.py            # every post
uv run python blogs/assets/figures.py --blog 02  # one post
```

Figures obey one series-wide color code: **warm terracotta `#C8421A` means compute, cool teal
`#1F7A8C` means memory and bandwidth.** Learn it once in post 01 and every later diagram reads
at a glance.

## Writing a new post

Read [`blogs/BLOG_RULES.md`](./blogs/BLOG_RULES.md). It has the end-to-end workflow, the 4-part
spine, notation, the hand-math-over-code rule, the figure pipeline, and the publish command.

## Related

If you want to build these models rather than read about them, I wrote a book on exactly that:
[**My Adventures with Large Language Models**](https://leanpub.com/adventures-with-llms) goes
from a vanilla Transformer through GPT-2, Llama 3, and DeepSeek in PyTorch, loading real
pretrained weights at each step.

Portfolio: [prathameshsaraf.com](https://prathameshsaraf.com)
