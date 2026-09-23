# Yingjie Tan

AI RESEARCH · NOTES IN PROGRESS

# Understanding how models move information.

I’m Yingjie Tan. I study the mechanisms that let language models retrieve, route, compress, and transform information—and how to make those mechanisms more interpretable and efficient.

[Explore the writing](blog.llms.md)

![](./assets/tan-solar-system.svg)

01

## Current questions

The themes that organize my experiments, essays, and reading notes.

A

### Information flow

Which representations and computational paths actually control a model’s answer?

B

### Sparse attention

What gets selected, what gets missed, and why can reasoning still succeed?

C

### Efficient context

How can long trajectories and KV caches be compressed without losing evidence?

02

## Research systems, visualized

Five animated maps of the systems and questions that shape my work.

![Animated Transformer circuit with signals moving through attention and MLP blocks.](./assets/atlas/transformer-circuit.svg)

01 · CIRCUITS

### Transformer circuits

Information does not simply pass through a model—it is routed, rewritten, amplified, and combined along a computational circuit.

![Animated sparse-attention stack where index heads select tokens before main heads reason over them.](./assets/atlas/sparse-attention-routing.svg)

02 · ROUTING

### DeepSeek sparse-attention routing

At every layer, an index head chooses a small token set; the main head then performs focused reasoning over that routed evidence.

![Animated KV-cache compression funnel that condenses a large memory bank into a compact representation.](./assets/atlas/kv-cache-compression.svg)

03 · COMPRESSION

### Compression as intelligence

KV-cache compression asks what can be discarded, what must remain addressable, and whether useful abstraction is itself a form of intelligence.

![Animated retrieval heads scanning a long context and converging on a small set of evidence tokens.](./assets/atlas/retrieval-heads.svg)

04 · RETRIEVAL

### Retrieval across long context

A useful head behaves like a telescope: it scans a wide memory field, resolves a few distant signals, and brings the right evidence into focus.

![Animated causal intervention map that highlights one computational path through a layered model.](./assets/atlas/causal-intervention.svg)

05 · INTERVENTION

### Finding the causal path

Interpretability becomes actionable when an observed representation can be intervened on and connected to a measurable change in behavior.

03

## Recent writing

[View all posts ↗](blog.llms.md)

### [From a Model’s Random-Number Fingerprint: A Study of LLM Reasoning Circuits](posts/random-number-fingerprint-circuit/index.llms.md)

Starting from a stable fingerprint in random-number generation, this post identifies attention heads that transport candidate information. Their causal effect appears when the answer is unresolved, generalizes across tasks and phrasings, and persists under interventions that interrupt normal reasoning. Head TV can then detect unreliable reasoning and trigger additional reasoning to improve performance.

Sep 22, 2026

38 min

Back to top
