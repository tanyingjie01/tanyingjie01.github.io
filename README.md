# yingjietan.me

Source code for Yingjie Tan's personal research website.

The English site lives at the project root and the Chinese mirror lives under
`zh/`. The language switch keeps readers on the equivalent page in the other
language, so translated posts should use the same slug in both trees.

## Stack

- Quarto Website
- QMD / Markdown
- SCSS
- Static HTML output

No Node.js framework, database, or server-side application is required.
KaTeX is vendored under `assets/katex/`, so equations and fonts do not depend
on a browser-time CDN request.

The five homepage research illustrations live under `assets/atlas/`. They are
self-contained animated SVG files with reduced-motion fallbacks and no external
image or script dependencies.

## Preview on the cluster

```bash
cd /data/zhangmy/tanyj/blog/yingjietan-site
quarto preview --timeout 0 --no-browser
```

The preview listens on `127.0.0.1:4200`. Open it through the IDE's port
forwarding feature or an SSH tunnel.

## Render the complete site

```bash
quarto render
```

The generated static website is written to `_site/`.

## Add a post

```bash
mkdir -p posts/my-new-post
$EDITOR posts/my-new-post/index.qmd
mkdir -p zh/posts/my-new-post
$EDITOR zh/posts/my-new-post/index.qmd
quarto render posts/my-new-post/index.qmd
quarto render
```

Each post should include at least:

```yaml
---
title: "Post title"
description: "One-sentence summary"
date: 2026-08-29
categories:
  - Research
body-classes: post-research
---
```

Use exactly one top-level Blog category per post: `Research`, `Thinking`, or
`Observation`. Its Chinese counterpart should use `研究`, `思考`, or `观察`,
respectively. These values drive the colored category tree and the matching
archive-card accents. Pair the category with `body-classes: post-research`,
`body-classes: post-reflection`, or `body-classes: post-observation` so the
article title, outline, margin rules, and other accents use the same blue,
orange, or green visual language.

The two directories ending in `-preview` are deliberately short layout
previews. Replace or remove them when real articles are ready.

English posts inherit `lang: en`; posts under `zh/posts/` inherit `lang:
zh-CN`. The two blog listings filter on that metadata so the languages do not
mix. Keep paired translations at `posts/<slug>/index.qmd` and
`zh/posts/<slug>/index.qmd` to preserve same-page language switching.

Do not edit `_site/` manually. It is generated from the QMD, YAML, SCSS, and
asset source files.
