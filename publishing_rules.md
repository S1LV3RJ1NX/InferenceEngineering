## 1. Publishing to the portfolio (Astro → Cloudflare Pages)

The portfolio at `/Users/prathamesh/portfolio/site` is Astro 4 + Tailwind, deployed to
Cloudflare Pages. Markdown bodies render via Astro's `<Content />` with Shiki code
highlighting. To host a math+figures post, the site needs (one-time):

1. A `blog` content collection in `src/content/config.ts` (schema = the frontmatter above).
2. Markdown integrations: `@astrojs/mdx`, `remark-math`, `rehype-katex` (+ KaTeX CSS),
   and a Mermaid integration; wired in `astro.config.mjs`.
3. A render page `src/pages/blog/[slug].astro` (clone `case-studies/[slug].astro`) and a
   `/blog` index.

**Per-post publish steps:**

1. Copy `blogs/NN-slug/README.md` → `site/src/content/blog/NN-slug.md`.
2. Copy `blogs/NN-slug/images/*` → `site/public/blogs/NN-slug/` and rewrite image paths
   from `./images/...` to `/blogs/NN-slug/...`.
3. `npm run build` to verify, then commit & push (Cloudflare Pages auto-deploys).
4. Subdomain `blogs.prathameshsaraf.com`: add it as a custom domain on the Pages project
   (or a dedicated Pages project) and add the `blogs` CNAME in Cloudflare DNS.

> Note the **plural** `public/blogs/<slug>/` directory and `/blogs/<slug>/...` URL prefix —
> that is what the existing posts (01–03) and the `[slug]` route use. Content markdown lives
> under the singular `src/content/blog/` collection.

**Concrete sync command (steps 1–2, copy-paste).** Run from the RL repo root; set `SLUG`
to the post folder. It copies the images, copies the markdown, and rewrites the relative
`./images/...` paths to the site's absolute `/blogs/<slug>/...` paths in one go:

```bash
SLUG="04-sarsa-qlearning-dqn"                                  # <- the only thing to change
SITE="/Users/prathamesh/portfolio/site"

mkdir -p "$SITE/public/blogs/$SLUG" "$SITE/src/content/blog"
cp blogs/$SLUG/images/* "$SITE/public/blogs/$SLUG/"            # figures, hero, gifs
sed -e "s#\./images/#/blogs/$SLUG/#g" \
    -e 's#\.\./\([0-9][0-9]-[^/]*\)/README\.md#/blogs/\1#g' \
    blogs/$SLUG/README.md \
    > "$SITE/src/content/blog/$SLUG.md"                        # markdown + path rewrite

cd "$SITE" && npm run build                                    # step 3: verify before pushing
```

Re-running it is idempotent (it overwrites), so it doubles as the "update a published post"
command.

Keep the palette/fonts aligned: canvas `#FAFAFA`, ink `#0A0A0A`, accent `#C8421A`,
Geist / Geist Mono — so figures and site share one visual language.

---

## 2. Tone & style

- Intuition-first, conversational but precise. Short paragraphs. One idea per paragraph.
- **Keep sentences simple.** Prefer short, plain sentences over long, multi-clause ones. If a
  sentence runs long or stacks several ideas, split it into two or three. Simple English beats
  clever phrasing; the reader is here for the concept, not the prose.
- **Flow between sections.** Each section should lead into the next: end a section by motivating
  what is coming, or open one by recalling what just came, so the post reads as one thread of
  thought rather than disconnected blocks.
- Lead with the "why" before the "what." Use a vivid concrete example, then generalize.
- Bold the one takeaway sentence per section. Don't pad.
- No emojis unless asked. American spelling. Define jargon on first use.
- **No em dashes (`—`).** They read as AI-generated. Use the punctuation the sentence actually
  wants instead: a comma for an aside, a colon to introduce, parentheses for a true parenthetical,
  or a period to split into two sentences. (This applies to prose only; the minus sign `−` in math
  and the en dash `–` in numeric ranges like `B1–B5` are fine.)

---
