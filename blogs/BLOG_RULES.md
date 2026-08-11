# BLOG_RULES.md: how to write a blog in the Inference Engineering series

This is the reusable recipe for every post in `blogs/`. Drop the next video's transcript
beside it, follow this spec, and the output will match the rest of the series in structure,
math style, figures, and code. The goal of each post: take a reader from
**intuition → math → a worked number → the engineering consequence**.

> **The series throughline (repeat it in every post):**
> *Arithmetic is cheap. Moving bytes is expensive. Every technique in this series is a way of buying back bandwidth.*
> Tie each new idea back to this sentence.

---

## 1. The 4-part spine (every post, in this order)

1. **The intuition.** Explain the idea in plain language first, with one AI hero illustration
   and a Mermaid diagram where a flow or a stack helps. No equations before the reader has a
   mental picture. Open with a concrete puzzle (a number that does not add up) and carry it
   through the post.
2. **The math you need, with the numbers next to each idea.** The key relationships in good
   LaTeX, one clean derivation, every symbol defined on first use. **Put the worked number
   immediately after the formula**, not batched at the end. Inference is a quantitative
   discipline: a formula without a plugged-in number teaches nothing.
3. **Worked examples inline.** Concretise every important formula against a real chip and a
   real model (the series standard is an NVIDIA H100 and Llama-3-70B, so numbers stay
   comparable across posts). Add a code-generated figure when the numbers tell a story.
4. **Putting it all together: a recap table.** Close with a short
   `Concept → Formula → Number` table summarising the pieces already shown inline.

Close with a short **"Where this goes next"** that sets up the following post and **ends with a
markdown link to it by its `shortName`**.

### 1a. Depth rule: expand the new, recall-and-link the old

Transcripts are spoken and terse. **Expand every one-liner about a _new_ idea into a full
explanation:** state it, give the intuition, work the *why*. No new term is used before it is
defined. Where the source states a number without deriving it, derive it.

**But never re-teach a concept an earlier post already covered.** For anything established in a
previous post (arithmetic intensity, the roofline, the KV cache, TTFT/TPOT), give a **one-line
recall plus a markdown link** and immediately use it. The depth budget belongs to what is new.

**Reference other posts by their `shortName`, never by number.** Write "the
[Memory Wall](../inference-01-memory-wall/README.md) post", not "post 1".

### 1b. Understanding checks

Render self-checks as hidden-answer blocks so the reader can self-test first:

```markdown
<details>
<summary><strong>Check:</strong> one-sentence question?</summary>

**Answer.** The answer, 1–3 sentences, in this post's own wording.
</details>
```

- **Summary stays clean:** the visible line is just `Check:` plus the question.
- **Revealed answer leads with a bold keyword:** `Answer.` or `Explanation.`
- **No `$…$` math in the `<summary>` line.** It is raw HTML, so KaTeX never processes it. Use
  plain text and Unicode instead (`3.35 TB/s`, `ops/byte`, `T−1`).

Place each check immediately after the section it tests.

### 1c. Attribution

Each post is built from a source video or paper. Credit it once, near the top or in a closing
note, with a link. Take inspiration for framing and structure, but **redraw every figure** in
the house palette and write every explanation in our own words.

---

## 2. Folder & file layout

```
blogs/
  BLOG_RULES.md                     <- this file
  README.md                         <- index of all posts
  assets/figures.py                 <- ONE script regenerates every fig-*.svg
  inference-NN-slug/
    README.md                       <- the post (markdown, source of truth)
    images/
      ai-*.png                      <- AI-generated conceptual illustrations
      fig-*.svg                     <- code-generated diagrams (matplotlib)
reference/                          <- gitignored: frames pulled from source videos
transcripts/                        <- source transcripts
```

- One folder per post, prefixed `inference-NN-`. The prefix namespaces this series away from
  the RL series, which already occupies `01-` through `12-` in the portfolio.
- The post body is `README.md` so it renders on GitHub as the folder's landing page.
- **This repo is the source of truth.** Publishing = sync a copy into the portfolio (§7).

---

## 3. Frontmatter (required)

```yaml
---
title: "..."          # sentence case, specific, no clickbait
shortName: "..."      # 2-4 word handle other posts use to link here
date: "YYYY-MM-DD"    # the actual publish date (today when you ship)
summary: "1-3 sentences. What the reader will be able to do after reading."
tags: ["llm-inference", "..."]   # llm-inference first: it is the series filter
order: N              # 100 + the NN prefix, so 101, 102, ...
---
```

`order` starts at 101 because the portfolio's RL series holds 1 through 12. The portfolio's
blog index filters client-side on exact tag strings, so **`llm-inference` doubles as the series
filter** and must be the first tag on every post.

`date` must be the current date on the day you publish. Never inherit it from a sibling post.

---

## 4. Math / LaTeX conventions

- Inline math with `$...$`, display math with `$$...$$`.
- **Define every symbol the first time it appears.**
- **Every display equation gets a two-step explanation: read it, then interpret it.** Right
  after `$$…$$`, first read the equation aloud symbol by symbol, then give the plain-English
  meaning: what the result means and what moves when something changes.
- **Every formula gets a number.** Plug in the H100 and Llama-3-70B values immediately.
- Keep equations small and frequent rather than one giant block.
### Notation

| Symbol | Meaning | Units |
| --- | --- | --- |
| $C$ | peak compute throughput | FLOP/s |
| $B$ | memory bandwidth | bytes/s |
| $I$ | arithmetic intensity | ops/byte |
| $I_{\text{ridge}}$ | ridge point, $C/B$ | ops/byte |
| $W$ | weight footprint | bytes |
| $N$ | parameter count | count |
| $l$ | transformer blocks | count |
| $n$ | key-value heads | count |
| $h$ | head dimension | count |
| $s$ | context length | tokens |

Three conventions worth stating, because the obvious choices collide:

- **Bytes never get a symbol.** Write "2 bytes per weight" or name the precision (fp16, fp8).
  There is no standard letter for it in either the ML or the HPC literature, and inventing one
  costs the reader more than it saves.
- **$B$ is bandwidth, and batch size stays out of display math.** In ML papers $B$ is almost
  always batch size, while the roofline literature this series builds on uses $\beta$ for
  bandwidth and $\pi$ for peak compute. We keep the plain-ASCII $B$ for bandwidth because it
  appears in nearly every formula here, and we express batch in words ("per request", "multiply
  by the number of concurrent requests"). A $b$ sitting next to a $B$ differs only by case and
  is misread at body-text size.
- **Model geometry follows [My Adventures with Large Language Models](https://leanpub.com/adventures-with-llms)**
  ($l$, $n$, $h$, $s$) so the blog and the book agree arithmetically. Where the book's letters
  would collide with a hardware symbol here, prefer the clearer local choice: the arithmetic has
  to match, the alphabet does not.

### Units

Be explicit and consistent. Use `FLOP/s` for a rate and `FLOPs` for a count. Write bandwidth in
`TB/s` and memory in `GB`. When a factor of 1000 vs 1024 matters, say which you used.

---

## 5. Image conventions

Two kinds of images, two clear jobs. **Default to SVG.**

- **`ai-*.png`, conceptual illustrations** (AI-generated). Hero and "feel" images with **no
  numbers or precise labels** (generators garble text). Prompt for the house palette: paper
  `#FAFAFA`, near-black `#0A0A0A` line work, terracotta `#C8421A` accent, teal `#1F7A8C`
  secondary. Flat editorial, lots of negative space, explicitly "no text or letters."
- **`fig-*.svg`, anything with numbers, axes, or math** (code-generated by
  `blogs/assets/figures.py`).
- **Mermaid** for flows, stacks, and decision trees, as a ```mermaid code block.
- Reference images with **relative paths**: `![alt](./images/fig-foo.svg)`.
- Every image needs **descriptive alt text** (a sentence, not "figure 1").
- **Every diagram gets an explanatory paragraph right after it** (including Mermaid), saying
  what it shows and how it ties to the concept just taught.

### The color semantics (series-wide)

The single most important visual rule. Every figure in this series obeys one code:

| Meaning | Constant | Hex |
|---|---|---|
| Compute, arithmetic, FLOPs | `COMPUTE` | `#C8421A` (terracotta, the house accent) |
| Memory, bandwidth, bytes | `MEMORY` | `#1F7A8C` (teal) |
| Neutral ink, structure | `INK` | `#0A0A0A` |
| Secondary text, axes | `MUTED` | `#525252` |
| Gridlines, borders | `DIVIDER` | `#E5E5E5` |
| Canvas | `CANVAS` | `#FAFAFA` |

Warm always means compute; cool always means memory. A reader who learns this in post one can
read any diagram in the series at a glance. Never use these two colors decoratively.

### `figures.py` rules

- Headless (`matplotlib.use("Agg")`), `svg.fonttype = "none"` so text stays selectable.
- Two style modes:
  - `house_style()` for anything with real axes and numbers.
  - `sketch_style()` for conceptual schematics: hand-drawn wobble via `path.sketch`, no grid,
    no spines. This is the "excalidraw" look.
- Save with `save(fig, blog_folder, name)` → `blogs/inference-NN-slug/images/fig-name.svg`.
- Register each post's builder in `BUILDERS`; support `--blog NN` to rebuild one post.
- Run: `uv run python blogs/assets/figures.py [--blog NN]`.

---

## 6. Code conventions

- **Python only.** Every snippet must be **runnable** as written.
- **Prefer hand math. Reach for code only when it earns the space.** If a number can be worked
  out in a line or two of arithmetic that the reader could follow themselves, write it out by
  hand instead. Use a code block only when at least one of these is true:
  - it needs a **library**: NumPy, PyTorch, transformers, a tokenizer, a profiler
  - it needs **non-trivial math**: logs, exponentials, softmax, matrix operations, sampling,
    anything iterative like a convergence loop
  - it needs **iteration or state** that is genuinely awkward on paper: a simulation, a
    benchmark, a loop over many steps
  - the **code itself is the point**, for example showing what a kernel or an API call looks like

  A block that assigns three constants, multiplies them, and prints the answer is worse than the
  same multiplication in prose: it buries one line of arithmetic inside ten lines of ceremony,
  and it invites the reader to skim past a number they should be checking.
- **How to write hand math.** Build the number up in named steps so each factor's role stays
  visible ("8 heads of 128 dimensions gives 1,024 numbers; keys and values doubles it to
  2,048"). When the same formula is evaluated at several inputs, use a small markdown table
  rather than a loop. Round in the prose and say so.
- **Code follows the explanation**, immediately after the concept it implements.
- **Comments go on the line above the code, never beside it.**
- **Show real output.** After every runnable snippet add a ` ```text title="Output" ` block with
  the **actual captured stdout**, then a one-sentence interpretation.
- Prefer pure Python and NumPy so snippets run anywhere. If a snippet needs a GPU, say so and
  give the captured output from the machine it ran on.
- Run snippets with `uv run python ...`.

---

## 7. Publishing to the portfolio (Astro → Cloudflare Pages)

The portfolio at `/Users/prathamesh/portfolio/site` is Astro 4 + Tailwind on Cloudflare Pages.
It already supports everything this series needs: KaTeX (`remark-math` + `rehype-katex`),
Mermaid (a remark plugin plus a client runtime), and Expressive Code for fenced blocks
including `text title="Output"`. **No site changes are required.**

The content collection schema in `src/content/config.ts` is the contract:
`title`, `shortName?`, `date`, `summary`, `tags[]`, `order`, `hero?`, `heroAlt?`, `draft`.
Note it is `summary`, not `description`. The slug comes from the **filename** in
`src/content/blog/`, and the render page does **not** print `title` as an H1, so the post body
must repeat the title as its own `#` heading.

**Per-post publish steps.** Run from this repo root; set `SLUG` to the post folder:

```bash
SLUG="inference-01-memory-wall"
SITE="/Users/prathamesh/portfolio/site"

mkdir -p "$SITE/public/blogs/$SLUG" "$SITE/src/content/blog"
cp blogs/$SLUG/images/* "$SITE/public/blogs/$SLUG/"
sed -e "s#\./images/#/blogs/$SLUG/#g" \
    -e 's#\.\./\([a-z0-9-]*\)/README\.md#/blogs/\1#g' \
    blogs/$SLUG/README.md \
    > "$SITE/src/content/blog/$SLUG.md"

cd "$SITE" && npm run build
```

Re-running it is idempotent, so it doubles as the "update a published post" command. Then
commit and push; Cloudflare Pages auto-deploys.

Two things to know: the portfolio homepage features the top four posts by `order` descending,
so `order: 101` and up will occupy that strip. And keep `draft: true` in the frontmatter until
the post is reviewed, since a draft builds no page and appears in no listing.

Keep the palette and fonts aligned (canvas `#FAFAFA`, ink `#0A0A0A`, accent `#C8421A`,
Geist / Geist Mono) so figures and site share one visual language.

---

## 8. Tone & style

- Intuition-first, conversational but precise. Short paragraphs. One idea per paragraph.
- **Keep sentences simple.** Prefer short, plain sentences over long, multi-clause ones.
- **Flow between sections.** End a section by motivating what is coming, or open one by
  recalling what just came, so the post reads as one thread of thought.
- Lead with the "why" before the "what." Use a vivid concrete number, then generalize.
- Bold the one takeaway sentence per section. Don't pad.
- No emojis unless asked. American spelling. Define jargon on first use.
- **No em dashes (`—`).** Use a comma for an aside, a colon to introduce, parentheses for a true
  parenthetical, or a period to split into two sentences. (Prose only; the minus sign `−` in
  math and the en dash `–` in numeric ranges are fine.)

---

## 9. Pre-publish checklist

- [ ] 4-part spine present and in order; throughline sentence referenced.
- [ ] Every new one-liner expanded; prior-post concepts recalled in one line + linked (§1a).
- [ ] Other posts referenced by `shortName` + link, never "post N" (§1a).
- [ ] Source video or paper credited with a link (§1c).
- [ ] No em dashes anywhere in prose (§8).
- [ ] Checks are untagged; revealed answers lead with a bold `Answer.` (§1b).
- [ ] Frontmatter complete; `llm-inference` is the first tag; `order` is 100 + NN; `date` is
      today's real publish date (§3).
- [ ] Every symbol defined on first use; every formula followed by a plugged-in number (§4).
- [ ] Every display equation read symbol by symbol, then interpreted (§4).
- [ ] Figures obey the color semantics: warm = compute, cool = memory (§5).
- [ ] Every figure and Mermaid diagram has an explanatory paragraph after it (§5).
- [ ] `figures.py` regenerates every `fig-*.svg` for the post with no errors.
- [ ] Every code block clears the §6 bar (library, non-trivial math, iteration, or the code is
      the point); anything simpler is written as hand math instead.
- [ ] Every snippet runs and has a `text title="Output"` block with real captured stdout (§6).
- [ ] Code comments are on the line above, never trailing inline (§6).
- [ ] "Where this goes next" ends with a `shortName` link to the next post (§1).
- [ ] (If publishing) synced to the portfolio, image paths rewritten, `npm run build` passes.

---

## 10. Copy-paste skeleton

```markdown
---
title: "..."
shortName: "..."
date: "YYYY-MM-DD"
summary: "..."
tags: ["llm-inference", "..."]
order: 10N
draft: true
---

# Title

![hero alt text](./images/ai-hero.png)

> **The throughline:** *Arithmetic is cheap. Moving bytes is expensive. Every technique in this series is a way of buying back bandwidth.*

## 1. The intuition
... the puzzle, one concrete number that does not add up, a ```mermaid``` diagram ...

## 2. The math you need
### 2.1 ...
$$ ... $$
... read the equation symbol by symbol, then interpret it; then plug in H100 + Llama-3-70B ...
![fig alt](./images/fig-foo.svg)
... a paragraph explaining the figure and tying it to this concept ...

## 3. Putting it all together
| Concept | Formula | Number (H100, Llama-3-70B) |
|---|---|---|
| ... | $...$ | ... |

## Where this goes next
... one paragraph, ending with a link to the next post by its shortName ...
```
