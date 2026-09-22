(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const legacyTasks = [
    {
      color: "#4e79a7",
      group: { zh: "局部位置", en: "Local position" },
      label: { zh: "输出第 2 项", en: "Output item 2" },
      description: { zh: "直接输出列表中的第 2 个元素。", en: "Directly output the second element in the displayed list." },
      accuracy: 100.0, tv: 0.0134,
      ordered: { accuracy: 100.0, tv: 0.0022 }, shuffled: { accuracy: 100.0, tv: 0.0156 }
    },
    {
      color: "#f28e2b",
      group: { zh: "局部位置", en: "Local position" },
      label: { zh: "输出第 4 项", en: "Output item 4" },
      description: { zh: "直接输出列表中的第 4 个元素。", en: "Directly output the fourth element in the displayed list." },
      accuracy: 89.5833, tv: 0.3157,
      ordered: { accuracy: 100.0, tv: 0.0141 }, shuffled: { accuracy: 87.5, tv: 0.3760 }
    },
    {
      color: "#e15759",
      group: { zh: "局部位置", en: "Local position" },
      label: { zh: "输出第 6 项", en: "Output item 6" },
      description: { zh: "直接输出列表中的第 6 个元素。", en: "Directly output the sixth element in the displayed list." },
      accuracy: 70.8333, tv: 0.8492,
      ordered: { accuracy: 100.0, tv: 1.9303 }, shuffled: { accuracy: 65.0, tv: 0.6330 }
    },
    {
      color: "#76b7b2",
      group: { zh: "局部位置", en: "Local position" },
      label: { zh: "锚点后一项", en: "Item after an anchor" },
      description: { zh: "给定列表中的一个锚点元素，输出紧跟在它后面的元素。", en: "Given an anchor element, output the element immediately after it." },
      accuracy: 91.6667, tv: 0.2756,
      ordered: { accuracy: 100.0, tv: 0.0361 }, shuffled: { accuracy: 90.0, tv: 0.3235 }
    },
    {
      color: "#59a14f",
      group: { zh: "局部位置", en: "Local position" },
      label: { zh: "锚点前一项", en: "Item before an anchor" },
      description: { zh: "给定列表中的一个锚点元素，输出紧挨在它前面的元素。", en: "Given an anchor element, output the element immediately before it." },
      accuracy: 64.5833, tv: 1.4017,
      ordered: { accuracy: 75.0, tv: 1.5783 }, shuffled: { accuracy: 62.5, tv: 1.3663 }
    },
    {
      color: "#c49a00",
      group: { zh: "纯全局结构", en: "Pure global structure" },
      label: { zh: "统计列表项数", en: "Count list items" },
      description: { zh: "忽略元素内容，统计列表一共有多少项。", en: "Ignore element identities and count all list items." },
      accuracy: 100.0, tv: 0.0058,
      ordered: { accuracy: 100.0, tv: 0.0100 }, shuffled: { accuracy: 100.0, tv: 0.0050 }
    },
    {
      color: "#b07aa1",
      group: { zh: "纯全局结构", en: "Pure global structure" },
      label: { zh: "统计相邻间隔", en: "Count adjacent gaps" },
      description: { zh: "忽略元素内容，统计相邻列表项之间的间隔数。", en: "Ignore element identities and count gaps between adjacent items." },
      accuracy: 100.0, tv: 0.1119,
      ordered: { accuracy: 100.0, tv: 0.0756 }, shuffled: { accuracy: 100.0, tv: 0.1191 }
    },
    {
      color: "#d45087",
      group: { zh: "纯全局结构", en: "Pure global structure" },
      label: { zh: "统计奇数位置", en: "Count odd positions" },
      description: { zh: "忽略元素内容，统计编号为奇数的位置数量。", en: "Ignore element identities and count positions with odd indices." },
      accuracy: 100.0, tv: 0.0904,
      ordered: { accuracy: 100.0, tv: 0.1009 }, shuffled: { accuracy: 100.0, tv: 0.0884 }
    },
    {
      color: "#9c755f",
      group: { zh: "纯全局结构", en: "Pure global structure" },
      label: { zh: "去首尾后计数", en: "Count without endpoints" },
      description: { zh: "忽略元素内容，去掉首项和末项后统计剩余项数。", en: "Ignore identities and count items remaining after removing both endpoints." },
      accuracy: 91.6667, tv: 0.9465,
      ordered: { accuracy: 100.0, tv: 0.9689 }, shuffled: { accuracy: 90.0, tv: 0.9421 }
    },
    {
      color: "#79706e",
      group: { zh: "纯全局结构", en: "Pure global structure" },
      label: { zh: "统计第 2–6 项", en: "Count items 2–6" },
      description: { zh: "忽略元素内容，统计从第 2 项到第 6 项共有多少项。", en: "Ignore identities and count how many items lie from positions 2 through 6." },
      accuracy: 100.0, tv: 0.0492,
      ordered: { accuracy: 100.0, tv: 0.0232 }, shuffled: { accuracy: 100.0, tv: 0.0543 }
    },
    {
      color: "#2f4b7c",
      group: { zh: "全局元素内容", en: "Global element content" },
      label: { zh: "规范顺序最早项", en: "Canonically earliest item" },
      description: { zh: "忽略展示位置，比较全部元素并输出规范顺序最早的一项。", en: "Ignore display positions and output the canonically earliest element." },
      accuracy: 93.75, tv: 0.6656,
      ordered: { accuracy: 100.0, tv: 0.6306 }, shuffled: { accuracy: 92.5, tv: 0.6726 }
    },
    {
      color: "#ef6c00",
      group: { zh: "全局元素内容", en: "Global element content" },
      label: { zh: "规范顺序最晚项", en: "Canonically latest item" },
      description: { zh: "忽略展示位置，比较全部元素并输出规范顺序最晚的一项。", en: "Ignore display positions and output the canonically latest element." },
      accuracy: 81.25, tv: 0.8014,
      ordered: { accuracy: 87.5, tv: 0.6841 }, shuffled: { accuracy: 80.0, tv: 0.8248 }
    },
    {
      color: "#7a5195",
      group: { zh: "全局元素内容", en: "Global element content" },
      label: { zh: "规范顺序中位项", en: "Canonical median" },
      description: { zh: "按规范顺序重排全部元素，并输出排序后的中位元素。", en: "Reorder all elements canonically and output the median element." },
      accuracy: 33.3333, tv: 8.6447,
      ordered: { accuracy: 12.5, tv: 8.3530 }, shuffled: { accuracy: 37.5, tv: 8.7031 }
    },
    {
      color: "#008c95",
      group: { zh: "全局元素内容", en: "Global element content" },
      label: { zh: "最接近序号均值", en: "Closest to mean rank" },
      description: { zh: "计算全部元素规范序号的均值，并输出序号最接近均值的一项。", en: "Average all canonical ranks and output the item closest to that mean." },
      accuracy: 22.9167, tv: 6.9306,
      ordered: { accuracy: 25.0, tv: 9.6910 }, shuffled: { accuracy: 22.5, tv: 6.3785 }
    },
    {
      color: "#a23b72",
      group: { zh: "全局元素内容", en: "Global element content" },
      label: { zh: "最大序号间隔后一项", en: "Item after largest rank gap" },
      description: { zh: "找出规范序号差最大且唯一的相邻元素对，并输出顺序较晚的一项。", en: "Find the unique adjacent pair with the largest rank gap and output its later item." },
      accuracy: 29.1667, tv: 5.7901,
      ordered: { accuracy: 37.5, tv: 7.5755 }, shuffled: { accuracy: 27.5, tv: 5.4330 }
    }
  ];

  // The original 15 tasks pool the ordered and shuffled conditions.
  [
    35.2233, 34.7130, 50.4940, 6.3294, 11.6975,
    5.4931, 1.0141, 2.3671, 1.3653, 8.4711,
    49.4656, 45.3465, 67.8969, 72.0774, 71.9159
  ].forEach((attention, index) => {
    legacyTasks[index].attention = attention;
  });

  legacyTasks.push(
    { label: { zh: "最接近首尾几何平均", en: "Closest to endpoint geometric mean" }, accuracy: 0.00, attention: 80.47, tv: 7.56 },
    { label: { zh: "删项后序号和命中目标 A", en: "Removal makes rank sum hit target A" }, accuracy: 0.00, attention: 67.84, tv: 5.09 },
    { label: { zh: "移除后使其余序号方差最小", en: "Removal minimizes remaining-rank variance" }, accuracy: 0.00, attention: 36.11, tv: 5.41 },
    { label: { zh: "交替序号和模 7 定位", en: "Alternating rank sum modulo 7" }, accuracy: 4.17, attention: 47.58, tv: 5.32 },
    { label: { zh: "上升次数加一定位", en: "Number of ascents plus one" }, accuracy: 4.17, attention: 42.56, tv: 3.79 },
    { label: { zh: "最小相邻跳跃右项", en: "Right item of smallest adjacent jump" }, accuracy: 4.17, attention: 70.66, tv: 4.56 },
    { label: { zh: "规范第 5 项", en: "Canonical item 5" }, accuracy: 8.33, attention: 56.29, tv: 5.31 },
    { label: { zh: "给定项向前两格", en: "Two positions before a given item" }, accuracy: 12.50, attention: 43.92, tv: 3.56 },
    { label: { zh: "逆序对数模 7 定位", en: "Inversion count modulo 7" }, accuracy: 12.50, attention: 35.74, tv: 5.36 },
    { label: { zh: "参与逆序对最多的元素", en: "Element in the most inversions" }, accuracy: 12.50, attention: 39.90, tv: 4.98 },
    { label: { zh: "最接近展示末三项序号均值", en: "Closest to mean rank of final three displayed items" }, accuracy: 20.83, attention: 73.23, tv: 5.90 },
    { label: { zh: "最近奇数位序号均值", en: "Closest odd-position rank mean" }, accuracy: 20.83, attention: 67.52, tv: 7.71 },
    { label: { zh: "最大向上跳跃右项", en: "Right item of largest upward jump" }, accuracy: 20.83, attention: 64.49, tv: 3.51 },
    { label: { zh: "首尾序号三分之一点", en: "One-third point between endpoint ranks" }, accuracy: 25.00, attention: 75.11, tv: 8.92 },
    { label: { zh: "唯一配对差的较晚项", en: "Later item of unique pairwise difference" }, accuracy: 25.00, attention: 54.81, tv: 3.42 },
    { label: { zh: "首尾序号三分之二点", en: "Two-thirds point between endpoint ranks" }, accuracy: 29.17, attention: 76.74, tv: 9.25 },
    { label: { zh: "偶数展示位最接近锚点", en: "Even-position item closest to anchor" }, accuracy: 29.17, attention: 27.06, tv: 4.67 },
    { label: { zh: "相邻对序号和接近两倍均值", en: "Adjacent rank sum closest to twice the mean" }, accuracy: 33.33, attention: 65.42, tv: 4.65 },
    { label: { zh: "规范第 3 项", en: "Canonical item 3" }, accuracy: 33.33, attention: 54.49, tv: 5.32 },
    { label: { zh: "偶数展示位规范最早", en: "Earliest canonical even-position item" }, accuracy: 33.33, attention: 50.66, tv: 3.36 },
    { label: { zh: "唯一配对和的较晚项", en: "Later item of unique pairwise sum" }, accuracy: 33.33, attention: 31.28, tv: 2.08 },
    { label: { zh: "规范后继", en: "Canonical successor" }, accuracy: 37.50, attention: 12.57, tv: 1.68 },
    { label: { zh: "规范向后两项", en: "Two items later canonically" }, accuracy: 41.67, attention: 30.63, tv: 2.52 },
    { label: { zh: "首中末三项规范最早", en: "Earliest of first, middle, and last" }, accuracy: 50.00, attention: 58.44, tv: 3.05 },
    { label: { zh: "给定项向后两格", en: "Two positions after a given item" }, accuracy: 54.17, attention: 16.75, tv: 3.50 },
    { label: { zh: "规范第 2 项", en: "Canonical item 2" }, accuracy: 62.50, attention: 53.15, tv: 3.19 },
    { label: { zh: "两锚点中规范较早者", en: "Earlier of two anchors canonically" }, accuracy: 91.67, attention: 6.03, tv: 0.53 },
    { label: { zh: "展示第 5 项", en: "Displayed item 5" }, accuracy: 95.83, attention: 37.77, tv: 1.12 },
    { label: { zh: "展示第 3 项", en: "Displayed item 3" }, accuracy: 95.83, attention: 15.82, tv: 0.32 },
    { label: { zh: "两锚点中规范较晚者", en: "Later of two anchors canonically" }, accuracy: 100.00, attention: 9.18, tv: 0.02 }
  );

  [
    { zh: "计算首项与末项规范序号的几何平均值，输出规范序号最接近该值的元素。", en: "Compute the geometric mean of the endpoint ranks and output the element whose canonical rank is closest to it." },
    { zh: "逐一移除元素，找出使剩余元素序号和命中给定目标 A 的一项。", en: "Remove each element in turn and find the one whose removal makes the remaining rank sum hit target A." },
    { zh: "逐一移除元素，找出使其余元素规范序号方差最小的一项。", en: "Remove each element in turn and find the one that minimizes the variance of the remaining canonical ranks." },
    { zh: "对展示序列的规范序号交替加减并取模 7，用所得位置定位元素。", en: "Alternately add and subtract displayed canonical ranks, take the result modulo 7, and use it to locate an element." },
    { zh: "统计相邻规范序号的上升次数，加一后用所得位置定位元素。", en: "Count ascents between adjacent canonical ranks, add one, and use the resulting position to locate an element." },
    { zh: "比较相邻元素的规范序号跳跃，输出最小跳跃对应的右侧元素。", en: "Compare canonical-rank jumps between adjacent elements and output the right item of the smallest jump." },
    { zh: "忽略展示顺序，根据规范顺序输出第 5 个元素。", en: "Ignore display order and output the fifth element in canonical order." },
    { zh: "从给定元素的位置向前移动两格，并输出到达的元素。", en: "Move two positions backward from a given element and output the element reached." },
    { zh: "统计展示序列中的逆序对数量并取模 7，用所得位置定位元素。", en: "Count inversions in the displayed sequence, take the result modulo 7, and use it to locate an element." },
    { zh: "统计各元素参与的逆序对数量，输出参与次数最多的元素。", en: "Count how many inversions involve each element and output the element involved most often." },
    { zh: "计算展示末三项的规范序号均值，输出序号最接近该均值的元素。", en: "Average the canonical ranks of the final three displayed items and output the closest-ranked element." },
    { zh: "计算奇数展示位元素的规范序号均值，输出序号最接近该均值的元素。", en: "Average the canonical ranks at odd display positions and output the closest-ranked element." },
    { zh: "比较相邻规范序号的向上跳跃，输出最大向上跳跃对应的右侧元素。", en: "Compare upward canonical-rank jumps and output the right item of the largest jump." },
    { zh: "在首尾规范序号之间取三分之一位置，输出最接近该位置的元素。", en: "Take the one-third point between the endpoint ranks and output the element closest to it." },
    { zh: "找出规范序号差唯一匹配目标的元素对，输出其中顺序较晚的一项。", en: "Find the pair whose canonical-rank difference uniquely matches the target and output its later item." },
    { zh: "在首尾规范序号之间取三分之二位置，输出最接近该位置的元素。", en: "Take the two-thirds point between the endpoint ranks and output the element closest to it." },
    { zh: "只考虑偶数展示位，输出规范序号最接近给定锚点的元素。", en: "Among even display positions, output the element whose canonical rank is closest to a given anchor." },
    { zh: "比较相邻元素对的序号和，找出最接近两倍全体均值的一对。", en: "Compare adjacent-pair rank sums and find the pair closest to twice the overall mean." },
    { zh: "忽略展示顺序，根据规范顺序输出第 3 个元素。", en: "Ignore display order and output the third element in canonical order." },
    { zh: "只考虑偶数展示位，输出其中规范顺序最早的元素。", en: "Among even display positions, output the canonically earliest element." },
    { zh: "找出规范序号和唯一匹配目标的元素对，输出其中顺序较晚的一项。", en: "Find the pair whose canonical-rank sum uniquely matches the target and output its later item." },
    { zh: "给定一个元素，输出它在规范顺序中的直接后继。", en: "Given an element, output its immediate successor in canonical order." },
    { zh: "给定一个元素，沿规范顺序向后移动两项并输出所得元素。", en: "Given an element, move two places later in canonical order and output the result." },
    { zh: "在展示首项、中间项和末项中，输出规范顺序最早的元素。", en: "Among the first, middle, and last displayed items, output the canonically earliest one." },
    { zh: "从给定元素的位置向后移动两格，并输出到达的元素。", en: "Move two positions forward from a given element and output the element reached." },
    { zh: "忽略展示顺序，根据规范顺序输出第 2 个元素。", en: "Ignore display order and output the second element in canonical order." },
    { zh: "比较两个给定锚点，输出规范顺序更早的一项。", en: "Compare two given anchors and output the canonically earlier one." },
    { zh: "直接输出展示列表中的第 5 个元素。", en: "Directly output the fifth element in the displayed list." },
    { zh: "直接输出展示列表中的第 3 个元素。", en: "Directly output the third element in the displayed list." },
    { zh: "比较两个给定锚点，输出规范顺序更晚的一项。", en: "Compare two given anchors and output the canonically later one." }
  ].forEach((description, index) => {
    legacyTasks[index + 15].description = description;
  });

  // Values default to n=48. Where n=48 produced tied success rates, selected
  // tasks use the matching success-rate/TV pair from n=46 or n=44.
  const tasks = [
    {
      label: { zh: "删项后序号和命中目标 A", en: "Removal makes rank sum hit target A" },
      description: { zh: "逐一移除元素，找出使剩余元素序号和命中给定目标 A 的一项。", en: "Remove each element in turn and find the one whose removal makes the remaining rank sum hit target A." },
      accuracy: 0.00, tv: 5.67
    },
    {
      label: { zh: "移除后方差最小", en: "Removal minimizes remaining-rank variance" },
      description: { zh: "逐一移除元素，找出使其余元素规范序号方差最小的一项。", en: "Remove each element in turn and find the one that minimizes the variance of the remaining canonical ranks." },
      accuracy: 0.00, tv: 6.02
    },
    {
      id: "closest_geometric_endpoint_mean",
      label: { zh: "最接近首尾几何平均", en: "Closest to endpoint geometric mean" },
      description: { zh: "计算首项与末项规范序号的几何平均值，输出规范序号最接近该值的元素。", en: "Compute the geometric mean of the endpoint ranks and output the element whose canonical rank is closest to it." },
      accuracy: 2.08, tv: 8.61
    },
    {
      label: { zh: "上升次数加一定位", en: "Number of ascents plus one" },
      description: { zh: "统计相邻规范序号的上升次数，加一后用所得位置定位元素。", en: "Count ascents between adjacent canonical ranks, add one, and use the resulting position to locate an element." },
      accuracy: 4.17, tv: 4.16
    },
    {
      label: { zh: "交替序号和模 7 定位", en: "Alternating rank sum modulo 7" },
      description: { zh: "对展示序列的规范序号交替加减并取模 7，用所得位置定位元素。", en: "Alternately add and subtract displayed canonical ranks, take the result modulo 7, and use it to locate an element." },
      accuracy: 6.25, tv: 5.50
    },
    {
      label: { zh: "最小相邻跳跃右项", en: "Right item of smallest adjacent jump" },
      description: { zh: "比较相邻元素的规范序号跳跃，输出最小跳跃对应的右侧元素。", en: "Compare canonical-rank jumps between adjacent elements and output the right item of the smallest jump." },
      accuracy: 8.33, tv: 5.28
    },
    {
      label: { zh: "给定项向前两格", en: "Two positions before a given item" },
      description: { zh: "从给定元素的位置向前移动两格，并输出到达的元素。", en: "Move two positions backward from a given element and output the element reached." },
      accuracy: 12.50, tv: 4.88
    },
    {
      label: { zh: "参与逆序对最多的元素", en: "Element in the most inversions" },
      description: { zh: "统计各元素参与的逆序对数量，输出参与次数最多的元素。", en: "Count how many inversions involve each element and output the element involved most often." },
      accuracy: 13.64, tv: 5.41, sourceN: 44
    },
    {
      label: { zh: "逆序对数模 7 定位", en: "Inversion count modulo 7" },
      description: { zh: "统计展示序列中的逆序对数量并取模 7，用所得位置定位元素。", en: "Count inversions in the displayed sequence, take the result modulo 7, and use it to locate an element." },
      accuracy: 14.58, tv: 5.62
    },
    {
      label: { zh: "偶数展示位最接近锚点", en: "Even-position item closest to anchor" },
      description: { zh: "只考虑偶数展示位，输出规范序号最接近给定锚点的元素。", en: "Among even display positions, output the element whose canonical rank is closest to a given anchor." },
      accuracy: 16.67, tv: 4.82
    },
    {
      id: "closest_mean_odd_positions",
      label: { zh: "最近奇数位序号均值", en: "Closest odd-position rank mean" },
      description: { zh: "计算奇数展示位元素的规范序号均值，输出序号最接近该均值的元素。", en: "Average the canonical ranks at odd display positions and output the closest-ranked element." },
      accuracy: 18.75, tv: 8.06
    },
    {
      label: { zh: "最大向上跳跃右项", en: "Right item of largest upward jump" },
      description: { zh: "比较相邻规范序号的向上跳跃，输出最大向上跳跃对应的右侧元素。", en: "Compare upward canonical-rank jumps and output the right item of the largest jump." },
      accuracy: 21.74, tv: 3.96, sourceN: 46
    },
    {
      label: { zh: "规范第 5 项", en: "Canonical item 5" },
      description: { zh: "忽略展示顺序，根据规范顺序输出第 5 个元素。", en: "Ignore display order and output the fifth element in canonical order." },
      accuracy: 22.92, tv: 5.48
    },
    {
      id: "closest_to_rank_mean",
      label: { zh: "最接近序号均值", en: "Closest to mean rank" },
      description: { zh: "计算全部元素规范序号的均值，并输出序号最接近均值的一项。", en: "Average all canonical ranks and output the item closest to that mean." },
      accuracy: 22.92, tv: 6.93
    },
    {
      label: { zh: "最接近展示末三项序号均值", en: "Closest to mean rank of final three displayed items" },
      description: { zh: "计算展示末三项的规范序号均值，输出序号最接近该均值的元素。", en: "Average the canonical ranks of the final three displayed items and output the closest-ranked element." },
      accuracy: 25.00, tv: 5.40
    },
    {
      label: { zh: "相邻对序号和接近两倍均值", en: "Adjacent rank sum closest to twice the mean" },
      description: { zh: "比较相邻元素对的序号和，找出最接近两倍全体均值的一对。", en: "Compare adjacent-pair rank sums and find the pair closest to twice the overall mean." },
      accuracy: 27.08, tv: 5.04
    },
    {
      id: "after_largest_rank_gap",
      label: { zh: "最大序号间隔后项", en: "Item after largest rank gap" },
      description: { zh: "找出规范序号差最大且唯一的相邻元素对，并输出顺序较晚的一项。", en: "Find the unique adjacent pair with the largest rank gap and output its later item." },
      accuracy: 29.17, tv: 5.79
    },
    {
      label: { zh: "规范第 3 项", en: "Canonical item 3" },
      description: { zh: "忽略展示顺序，根据规范顺序输出第 3 个元素。", en: "Ignore display order and output the third element in canonical order." },
      accuracy: 31.25, tv: 6.02
    },
    {
      label: { zh: "规范向后两项", en: "Two items later canonically" },
      description: { zh: "给定一个元素，沿规范顺序向后移动两项并输出所得元素。", en: "Given an element, move two places later in canonical order and output the result." },
      accuracy: 34.09, tv: 2.49, sourceN: 44
    },
    {
      label: { zh: "唯一配对和的较晚项", en: "Later item of unique pairwise sum" },
      description: { zh: "找出规范序号和唯一匹配目标的元素对，输出其中顺序较晚的一项。", en: "Find the pair whose canonical-rank sum uniquely matches the target and output its later item." },
      accuracy: 33.33, tv: 2.48
    },
    {
      label: { zh: "唯一配对差的较晚项", en: "Later item of unique pairwise difference" },
      description: { zh: "找出规范序号差唯一匹配目标的元素对，输出其中顺序较晚的一项。", en: "Find the pair whose canonical-rank difference uniquely matches the target and output its later item." },
      accuracy: 34.78, tv: 3.61, sourceN: 46
    },
    {
      id: "earliest_even_display_positions",
      label: { zh: "偶数展示位规范最早", en: "Earliest canonical even-position item" },
      description: { zh: "只考虑偶数展示位，输出其中规范顺序最早的元素。", en: "Among even display positions, output the canonically earliest element." },
      accuracy: 43.75, tv: 3.97
    },
    {
      label: { zh: "规范后继", en: "Canonical successor" },
      description: { zh: "给定一个元素，输出它在规范顺序中的直接后继。", en: "Given an element, output its immediate successor in canonical order." },
      accuracy: 50.00, tv: 1.28
    },
    {
      label: { zh: "给定项向后两格", en: "Two positions after a given item" },
      description: { zh: "从给定元素的位置向后移动两格，并输出到达的元素。", en: "Move two positions forward from a given element and output the element reached." },
      accuracy: 56.25, tv: 2.36
    },
    {
      label: { zh: "首中末三项规范最早", en: "Earliest of first, middle, and last" },
      description: { zh: "在展示首项、中间项和末项中，输出规范顺序最早的元素。", en: "Among the first, middle, and last displayed items, output the canonically earliest one." },
      accuracy: 64.58, tv: 2.21
    },
    {
      id: "canonical_second",
      label: { zh: "规范第 2 项", en: "Canonical item 2" },
      description: { zh: "忽略展示顺序，根据规范顺序输出第 2 个元素。", en: "Ignore display order and output the second element in canonical order." },
      accuracy: 70.83, tv: 3.15
    },
    {
      label: { zh: "两锚点中规范较早者", en: "Earlier of two anchors canonically" },
      description: { zh: "比较两个给定锚点，输出规范顺序更早的一项。", en: "Compare two given anchors and output the canonically earlier one." },
      accuracy: 89.58, tv: 0.37
    },
    {
      id: "display_fifth",
      label: { zh: "展示第 5 项", en: "Displayed item 5" },
      description: { zh: "直接输出展示列表中的第 5 个元素。", en: "Directly output the fifth element in the displayed list." },
      accuracy: 93.75, tv: 1.24
    },
    {
      label: { zh: "展示第 3 项", en: "Displayed item 3" },
      description: { zh: "直接输出展示列表中的第 3 个元素。", en: "Directly output the third element in the displayed list." },
      accuracy: 95.83, tv: 0.23
    },
    {
      label: { zh: "两锚点中规范较晚者", en: "Later of two anchors canonically" },
      description: { zh: "比较两个给定锚点，输出规范顺序更晚的一项。", en: "Compare two given anchors and output the canonically later one." },
      accuracy: 100.00, tv: 0.02
    }
  ];

  // A deterministic palette keeps every task visually distinct without a legend.
  tasks.forEach((task, index) => {
    const hue = (index * 137.508) % 360;
    const lightness = [42, 50, 36][index % 3];
    task.color = `hsl(${hue.toFixed(1)} 64% ${lightness}%)`;
  });

  function applyTaskColors() {
    const taskColors = new Map(tasks.filter((task) => task.id).map((task) => [task.id, task.color]));
    document.querySelectorAll("[data-h18-task-id]").forEach((element) => {
      const color = taskColors.get(element.dataset.h18TaskId);
      if (color) element.style.setProperty("--h18-task-color", color);
    });
  }

  const copy = {
    zh: {
      xAxis: "成功率（%）", yAxis: "列表边 TV（%）",
      success: "成功率", tv: "删列表边 TV",
      correlation: "相关系数",
      aria: "不同任务的成功率与 L82.H18 删列表边 TV 散点图"
    },
    en: {
      xAxis: "Success rate (%)", yAxis: "List-edge TV (%)",
      success: "Success rate", tv: "List-edge TV",
      correlation: "Correlation",
      aria: "Scatter plot of task success rate and L82.H18 list-edge TV"
    }
  };

  function svgElement(tag, attributes = {}) {
    const element = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  const percent = (value, precision = 2) => `${value.toFixed(precision)}%`;
  const tvPercent = (value) => percent(value, value < 0.1 ? 3 : 2);

  function initialize(root) {
    if (root.dataset.ready === "true") return;
    root.dataset.ready = "true";

    const lang = root.dataset.lang === "en" ? "en" : "zh";
    const labels = copy[lang];
    const width = 780;
    const height = 500;
    const margin = { top: 28, right: 28, bottom: 68, left: 76 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const x = (value) => margin.left + (value / 100) * plotWidth;
    const y = (value) => margin.top + plotHeight - (value / 10) * plotHeight;

    root.setAttribute("role", "figure");
    root.setAttribute("aria-label", labels.aria);

    const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": labels.aria });
    root.appendChild(svg);
    const grid = svgElement("g", { class: "h18-scatter-grid" });
    svg.appendChild(grid);

    for (let tick = 0; tick <= 100; tick += 20) {
      grid.appendChild(svgElement("line", { x1: x(tick), y1: margin.top, x2: x(tick), y2: margin.top + plotHeight }));
      const tickLabel = svgElement("text", { x: x(tick), y: margin.top + plotHeight + 28, class: "h18-scatter-tick", "text-anchor": "middle" });
      tickLabel.textContent = tick;
      svg.appendChild(tickLabel);
    }

    for (let tick = 0; tick <= 10; tick += 2) {
      grid.appendChild(svgElement("line", { x1: margin.left, y1: y(tick), x2: margin.left + plotWidth, y2: y(tick) }));
      const tickLabel = svgElement("text", { x: margin.left - 14, y: y(tick) + 5, class: "h18-scatter-tick", "text-anchor": "end" });
      tickLabel.textContent = tick;
      svg.appendChild(tickLabel);
    }

    svg.appendChild(svgElement("line", { x1: margin.left, y1: margin.top + plotHeight, x2: margin.left + plotWidth, y2: margin.top + plotHeight, class: "h18-scatter-axis" }));
    svg.appendChild(svgElement("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotHeight, class: "h18-scatter-axis" }));

    const xLabel = svgElement("text", { x: margin.left + plotWidth / 2, y: height - 12, class: "h18-scatter-axis-label", "text-anchor": "middle" });
    xLabel.textContent = labels.xAxis;
    svg.appendChild(xLabel);
    const yLabel = svgElement("text", { x: 20, y: margin.top + plotHeight / 2, class: "h18-scatter-axis-label", "text-anchor": "middle", transform: `rotate(-90 20 ${margin.top + plotHeight / 2})` });
    yLabel.textContent = labels.yAxis;
    svg.appendChild(yLabel);

    const meanAccuracy = tasks.reduce((sum, task) => sum + task.accuracy, 0) / tasks.length;
    const meanTv = tasks.reduce((sum, task) => sum + task.tv, 0) / tasks.length;
    const covariance = tasks.reduce((sum, task) => sum + (task.accuracy - meanAccuracy) * (task.tv - meanTv), 0);
    const accuracyVariance = tasks.reduce((sum, task) => sum + (task.accuracy - meanAccuracy) ** 2, 0);
    const tvVariance = tasks.reduce((sum, task) => sum + (task.tv - meanTv) ** 2, 0);
    const fitSlope = covariance / accuracyVariance;
    const fitIntercept = meanTv - fitSlope * meanAccuracy;
    const correlation = covariance / Math.sqrt(accuracyVariance * tvVariance);

    const fitLayer = svgElement("g", { class: "h18-scatter-fit" });
    fitLayer.appendChild(svgElement("line", {
      x1: x(0), y1: y(fitIntercept),
      x2: x(100), y2: y(fitIntercept + fitSlope * 100)
    }));
    const fitLabelX = 77;
    const fitLabel = svgElement("text", {
      x: x(fitLabelX),
      y: y(fitIntercept + fitSlope * fitLabelX + 0.55),
      "text-anchor": "middle"
    });
    fitLabel.textContent = `${labels.correlation} = ${correlation.toFixed(2)}`;
    fitLayer.appendChild(fitLabel);
    svg.appendChild(fitLayer);

    const tooltip = document.createElement("div");
    tooltip.className = "h18-scatter-tooltip";
    tooltip.setAttribute("role", "status");
    tooltip.setAttribute("aria-live", "polite");
    root.appendChild(tooltip);

    const points = [];

    function tooltipMarkup(task) {
      return `<div class="h18-scatter-tooltip-title">${task.label[lang]}</div>
        <div class="h18-scatter-tooltip-description">${task.description[lang]}</div>
        <div class="h18-scatter-tooltip-metrics">
          <div><span>${labels.success}</span><strong>${percent(task.accuracy)}</strong></div>
          <div><span>${labels.tv}</span><strong>${tvPercent(task.tv)}</strong></div>
        </div>`;
    }

    function positionTooltip(clientX, clientY) {
      const rootRect = root.getBoundingClientRect();
      const tooltipRect = tooltip.getBoundingClientRect();
      let left = clientX - rootRect.left + 14;
      let top = clientY - rootRect.top + 14;
      left = Math.max(10, Math.min(left, rootRect.width - tooltipRect.width - 10));
      if (top + tooltipRect.height > rootRect.height - 10) top = clientY - rootRect.top - tooltipRect.height - 14;
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${Math.max(10, top)}px`;
    }

    function highlight(index) {
      points.forEach((point, pointIndex) => {
        point.classList.toggle("is-active", pointIndex === index);
        point.classList.toggle("is-muted", pointIndex !== index);
      });
    }

    function showTooltip(task, index, clientX, clientY) {
      tooltip.innerHTML = tooltipMarkup(task);
      tooltip.dataset.visible = "true";
      highlight(index);
      positionTooltip(clientX, clientY);
    }

    function hideTooltip() {
      tooltip.dataset.visible = "false";
      points.forEach((point) => point.classList.remove("is-active", "is-muted"));
    }

    const pointLayer = svgElement("g", { class: "h18-scatter-points" });
    svg.appendChild(pointLayer);
    tasks.forEach((task, index) => {
      const point = svgElement("circle", {
        cx: x(task.accuracy), cy: y(task.tv), r: 7.5, fill: task.color,
        class: "h18-scatter-point", tabindex: "0", role: "button",
        "aria-label": `${task.label[lang]}: ${labels.success} ${percent(task.accuracy)}, ${labels.tv} ${tvPercent(task.tv)}`
      });
      const title = svgElement("title");
      title.textContent = `${task.label[lang]} — ${labels.success} ${percent(task.accuracy)}, ${labels.tv} ${tvPercent(task.tv)}`;
      point.appendChild(title);
      point.addEventListener("pointerenter", (event) => showTooltip(task, index, event.clientX, event.clientY));
      point.addEventListener("pointermove", (event) => positionTooltip(event.clientX, event.clientY));
      point.addEventListener("pointerleave", hideTooltip);
      point.addEventListener("focus", () => {
        const rect = root.getBoundingClientRect();
        showTooltip(task, index, rect.left + rect.width * 0.52, rect.top + 110);
      });
      point.addEventListener("blur", hideTooltip);
      pointLayer.appendChild(point);
      points.push(point);
    });
  }

  document.querySelectorAll(".h18-task-scatter").forEach(initialize);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyTaskColors, { once: true });
  } else {
    applyTaskColors();
  }
})();
