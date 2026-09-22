# Yingjie Tan

AI 研究 · 持续更新的笔记

# 理解模型如何传递信息。

我是 Yingjie Tan。我的研究关注语言模型如何检索、路由、压缩和转换信息， 以及如何让这些机制更容易理解、更高效。

[阅读博客](../zh/blog.llms.md) [关于我](../zh/about.llms.md)

![](../assets/tan-solar-system.svg)

01

## 当前关注的问题

这些主题串联起我的实验、文章与阅读笔记。

A

### 信息流

哪些表征与计算路径真正决定了模型最终给出的答案？

B

### 稀疏注意力

模型选择了什么、遗漏了什么，又为什么仍然可能完成正确推理？

C

### 高效上下文

如何压缩长轨迹与 KV cache，同时保留与当前问题相关的证据？

02

## 用图像理解研究系统

五幅动态地图，串联起我关注的模型结构、信息流与研究问题。

![动态 Transformer 电路图：信号流经注意力与 MLP 模块。](../assets/atlas/transformer-circuit.svg)

01 · 计算电路

### Transformer 中的信息电路

信息并不是简单地穿过模型；它沿着计算电路被路由、改写、放大，并与其他证据组合。

![动态稀疏注意力堆栈：每层由 index head 选择 token，再由主 head 进行推理。](../assets/atlas/sparse-attention-routing.svg)

02 · 稀疏路由

### DeepSeek 稀疏注意力架构

在每一层，index head 先选择少量 token，主 head 再围绕这些被路由的证据进行集中推理。

![动态 KV Cache 压缩漏斗：大规模记忆被压缩成紧凑表示。](../assets/atlas/kv-cache-compression.svg)

03 · 上下文压缩

### 压缩即智能

KV Cache 压缩的核心，是判断什么可以舍弃、什么必须仍可寻址，以及有效抽象本身是否就是一种智能。

![动态检索头图：多个注意力头扫描长上下文，并汇聚到少量证据 token。](../assets/atlas/retrieval-heads.svg)

04 · 长程检索

### 跨越长上下文的检索

一个有效的检索头像望远镜：扫描广阔的记忆空间，分辨少数远端信号，并把正确证据带回当前计算。

![动态因果干预图：一条关键计算路径在多层模型中被高亮。](../assets/atlas/causal-intervention.svg)

05 · 因果干预

### 寻找真正起作用的路径

当一个被观察到的表征可以被干预，并能与模型行为的可测变化联系起来，可解释性才真正具有因果意义。

03

## 近期文章

[查看全部文章 ↗](../zh/blog.llms.md)

### [从模型的随机数指纹出发：\
一项关于 LLM 推理回路的研究](../zh/posts/random-number-fingerprint-circuit/index.llms.md)

2026年8月30日

11 分钟

### [为什么我决定开博客：论我对AI科研与论文的看法](../zh/posts/welcome/index.llms.md)

2026年8月29日

1 分钟

返回顶部
