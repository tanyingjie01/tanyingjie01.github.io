# Welcome to My Research Notes

Thinking

A visual tour of the layout I will use for future technical essays and experiment reports.

Author

Yingjie Tan

Published

August 29, 2026

This is a placeholder article: its job is to demonstrate the reading experience before the first real research note arrives.

Reading note

The right margin holds optional context: definitions, caveats, small figures, and observations that should not interrupt the main argument.

## A layout for technical thinking

Research writing asks a page to do several things at once. The central argument must remain easy to follow, while code, equations, figures, qualifications, and cross-references stay close enough to be useful. This template gives each kind of information a deliberate place.

The page has three layers on wide screens: a navigable outline on the left, the primary narrative in the center, and authored annotations on the right. On a narrow screen, those layers become one continuous reading flow.

Margin notes are written explicitly in the source. They are not comments from readers, and they remain part of the article.

### Equations stay in the reading flow

For example, a schematic residual update can be written as

x\_{\ell + 1} = x\_\ell + \mathcal{A}\_\ell(x\_\ell) + \mathcal{M}\_\ell(x\_\ell).

The typesetting is intentionally quiet: the equation is prominent enough to inspect without turning the page into a slide deck.

Notation

\mathcal{A}\_\ell and \mathcal{M}\_\ell stand in for attention and MLP updates. The example is illustrative rather than a claim about a particular model.

## Code and evidence

Code blocks include syntax highlighting, sensible line spacing, horizontal overflow protection, and a copy button.

``` python
def residual_update(x, attention_out, mlp_out):
    """A small display-only example."""
    return x + attention_out + mlp_out
```

Reproducibility

Expensive computation will be frozen after an explicit render, so rebuilding the navigation will not silently rerun old experiments.

### Room for wide results

Some evidence needs more horizontal space than prose. A wide region can extend beyond the text column for an ablation table, an attention map, or a systems diagram.

WIDE FIGURE · PLACEHOLDER

A future figure, table, or information-flow diagram can occupy this space.

## Designed to get out of the way

The visual system uses a warm paper-like background, strong but restrained typography, and a single blue-violet accent. Navigation and metadata are easy to find; the article itself remains the loudest object on the page.

The template is now ready to be replaced by real notes one post at a time.

Back to top
