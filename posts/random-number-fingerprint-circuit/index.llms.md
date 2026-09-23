# From a Model’s Random-Number Fingerprint: A Study of LLM Reasoning Circuits

Author

Published

September 22, 2026

Discussions about GPT “getting worse” or being routed to a smaller model never really stop. A [LINUX DO post](https://linux.do/t/topic/2472419) proposed a simple test: repeatedly ask a model to choose a random number from a fixed range, estimate its output distribution, and use that distribution as a statistical fingerprint of the model¹.

Note 1

Since the release of Astra, a more popular test seems to be the pelican-riding-a-bicycle prompt; see this [LINUX DO post](https://linux.do/t/topic/2858863).

It is easy to see why this method works and why it is useful. But it also suggests a more interesting research question:

**How is this random-number fingerprint written or generated inside the model?**

To investigate this question, we began with prompts asking the model to “choose a random number from 1 to 300”. We located heads that strongly read the candidate-range boundary `300`. Further experiments gradually revealed their more general role: reading the candidate set in a reasoning task. When the answer has not yet been successfully inferred and the model must still choose among candidates, the information these heads write into the residual stream markedly changes the output.

When we manually interrupt the normal reasoning circuit—for example, by deleting a key reasoning chain or blocking retrieval heads—the causal effects of these candidate heads are likewise amplified. This suggests two competing circuits: when the model can reason normally, the reasoning circuit operates while the candidate-head circuit is suppressed;when that circuit is interrupted, the candidate circuit gains causal influence and begins to dominate behavior.

Based on this observation, we use the candidate heads’ behavior as a signal of whether the model’s reasoning is reliable. When it indicates unreliable reasoning, we trigger additional reasoning. The experiments show that this strategy can detect such cases and improve performance.

**If this question interests you, read on. The complete post will take about 30 minutes.**

## 1 Background: Non-random Outputs from LLMs

Suppose we give a language model the following prompt:

Choose a random number from 1 to 300. Output only the number.

The model does not literally draw a random number during its forward pass. More precisely, a language model’s forward pass is a deterministic function from the input text to the logits for the next token. The decoder then uses a temperature T to turn those logits into a probability distribution:

p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}.

With both the prompt and decoding configuration fixed, the model’s next-token probability distribution p is fixed as well².

Under sampling, a pseudorandom number generator selects a token from the distribution p. Individual outputs can vary, while the long-run statistical distribution should remain stable. This is why random-number choice can serve as a model fingerprint: different parameter sets assign different but stable probabilities to number tokens, and repeated sampling turns those preferences into a recognizable output distribution.

Note 2

Real inference services are not necessarily reproducible bit for bit. Batching and parallel reductions in backends such as vLLM and SGLang can change the order of floating-point operations and introduce numerical differences. See Thinking Machines Lab’s [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

The question we really care about is: **How is this probability distribution written and generated during the model’s forward pass?**

## 2 The Head That Writes the Random-Number Fingerprint

During an LLM forward pass, QK determines where attention reads from, while OV determines what is written into the residual stream. That information is read and processed by MoE layers before the output layer produces a logit distribution.

Under this picture, a reasonable hypothesis is that some deep-layer heads encode the range `1–300`, so downstream computation can read the candidate range. That information is written into the residual stream and shapes the output; the model’s selection preference is written along the way.

To test this hypothesis, we ran experiments on the MoE model Qwen3-235B-A22B. We constructed nine Chinese prompts similar to “请从1-300随机选择一个数，只输出这个数字”. All ask the model to select a random number from `1–300`, but use different semantic phrasings.

The exact probabilities vary across prompts, but their overall structure remains stable: almost all next-token probability mass is assigned to digits, and `1` and `2` consistently form the dominant competition³.

Note 3

I suspect that the high probability of `1` and `2` relative to other digits reflects the numerical distribution in the pretraining corpus. The pronounced preference between `1` and `2`, however, seems more likely to be encoded during post-training. This may explain why many models favor `1` and `2` while the exact probabilities form distinct fingerprints.

|  P(1)  |  P(2)  | P(3\text{–}9) | P(\text{digit}) |
|:------:|:------:|:-------------:|:---------------:|
| 49.39% | 49.74% |     0.73%     |     99.86%      |

We therefore use “which head most strongly disrupts this generation preference?” as the criterion for locating the key head.

### 2.1 Head localization

Qwen3-235B-A22B has 94 layers and 64 query heads per layer. We examined each head through ablation: immediately before the layer’s attention `o_proj`, we zeroed the target head’s 128-dimensional output slice at every input-token position while leaving all other computation unchanged.

We measured the change between the original and ablated full-vocabulary distributions with total variation distance:

\operatorname{TV}(h,l) =\operatorname{TV}\\\left(p,p^{(h,l)}\right) =\frac{1}{2}\sum\_{v\in\mathcal V} \left\|p(v)-p^{(h,l)}(v)\right\|.

Here, p is the original next-token distribution, while p^{(h,l)} is the distribution after ablating head (h,l). A larger TV means that the head has a stronger causal effect on the current output distribution.

We identified `L82.H18` as the most significant head. Its full-vocabulary TV reached **34.0%**, the strongest effect among all heads⁴.

Note 4

`L82.H18` also ranks first by centered full-vocabulary logit \operatorname{RMS}\_{c}(h,l), defined as:

\operatorname{RMS}\_{c}(h,l)= \sqrt{\frac{1}{\|\mathcal V\|}\sum\_{v\in\mathcal V} \left(\Delta z_v^{(h,l)}-\overline{\Delta z}^{(h,l)}\right)^2}.

Here, \Delta z_v^{(h,l)} is the change in token v’s logit after ablating head (h,l), and \overline{\Delta z}^{(h,l)} is its full-vocabulary mean. \operatorname{RMS}\_{c}(h,l) represents the overall effect of ablating that head on the full-vocabulary logit structure.

![Local heatmap of mean full-vocabulary total variation around L82.H18.](../../assets/figures/random-number/all_head_tv_heatmap.png)

Figure 1(a): Local TV landscape around `L82.H18`.

![Grouped bar chart comparing probabilities for tokens 1 and 2 in the full model and after ablating L82.H18.](../../assets/figures/random-number/l82_h18_probability_shift.png)

Figure 1(b): Changes in P(1) and P(2) after ablating `L82.H18`.

After ablating `L82.H18`, P(1) decreased for every prompt, with a mean change of **−33.80** percentage points; meanwhile, P(2) changed by an average of **+33.54** percentage points, while the remaining output probabilities were nearly unchanged.

This shows that the identified `L82.H18` does indeed write a strong preference within the set of numerical candidates.

### 2.2 Head attention distribution

We next examine the attention distribution of `L82.H18`. All prompts used here contain 28 tokens and share a similar structure: `300` is split across t7–t9, the model-side `assistant` token is at t22, and the final `\n\n` before the answer is at t27.

![Twenty-eight token prompt map arranged in two rows, highlighting 300 at t7 through t9, assistant at t22, think-mode tokens, and the final double-newline answer boundary at t27.](../../assets/figures/random-number/h18_prompt_token_map.png)

Figure 2: The 28-token random-number prompt after applying the chat template.

The most concentrated attention connection of `L82.H18` runs from token `\n\n` at t27 to token `300` at t7–t9, with an average attention mass of **99.56%**. The figure below shows the attention distributions for five representative prompts.

![L82.H18 attention across five representative prompts; each row is one prompt and 300 always occupies t7 through t9](../../assets/figures/random-number/h18_q27_attention_heatmap.png)

Figure 3: `L82.H18@q27` attention across five representative prompts. Each row represents one prompt, each column a token position, and token `300` always occupies t7–t9.

In other words, as the model prepares its answer, `L82.H18` reads almost exclusively from the range boundary `300` and writes the resulting vector at the start of the answer.

### 2.3 Head attention intervention

To determine whether the prominent attention edge actually dominates the output shift, we performed the following `L82.H18` attention interventions on the same prompts:

- **Full model**: no intervention; this is the baseline.
- **Remove `L82.H18`**: zero the head’s output at every token position.
- **Remove the `\n\n → 300` attention edges**: zero only the three edges from q27 to t7–t9.
- **Remove `L82.H18`, then restore the `\n\n → 300` write**: first remove `L82.H18`, then add back at t27 the original residual-write vector contributed predominantly by t7–t9; the other token positions remain ablated.

The first three experiments test whether the `\n\n → 300` attention edges dominate the behavior of `L82.H18`. The fourth restores the information these edges write into the residual stream, testing whether it recovers the effect of the complete head.

![Grouped vertical bar chart showing the probability of generating token 1, the probability of generating token 2, and full-vocabulary TV under four L82.H18 intervention conditions.](../../assets/figures/random-number/h18_q27_attention_intervention.png)

Figure 4: The probabilities of generating tokens `1` and `2`, and full-vocabulary TV, under four conditions: full model, remove L82.H18, remove \n\n → 300 attention edges, and remove L82.H18 then restore the \n\n → 300 write.

The results show that removing the `\n\n → 300` attention edges reaches **96.95%** of the effect of removing the complete `L82.H18` head. After removing `L82.H18`, restoring only the residual write dominated by these edges recovers **97.33%** of the original deletion effect.

This suggests that the main role of `L82.H18` is to transport candidate-range information to the answer position and thereby help form the model’s output fingerprint.

## 3 Head Generalization Experiments

So far, we have demonstrated the behavior of `L82.H18` when choosing a random number from 1 to 300. The next question is how well it generalizes: can it aggregate and transport candidate information to help form a model fingerprint in other tasks, and can it serve a more general function?

To answer this question, we varied candidate types and constraints, downstream tasks, and reasoning length to examine the head’s function across settings.

### 3.1 Candidate-Type Generalization Experiments

We began with the most direct generalization: if numerical candidates are replaced by another type of candidate, does `L82.H18` exhibit the same properties? We replaced the original `1–300` range with an alphabetic range:

Choose any letter from A to D and output only that letter.

The result closely mirrors the numerical-range experiment: `L82.H18` assigns **88.62%** attention to the letter `D`, and deleting this prominent attention edge produces **30.3%** TV.

We then expanded the candidates into explicit lists to test whether this property extends to general candidate sets. Every experiment used the same template:

Choose any item from the candidate list 【…】 and output only the selected item.

We constructed a range of candidate lists, including canonical and non-canonical lists.

- **Canonical lists**: the elements have a strong intrinsic order and are presented in that order, such as the heavenly-stem list 【甲, 乙, 丙, 丁, …】.
- **Non-canonical lists**: the elements have no stable intrinsic order and are arranged without a canonical sequence, such as the food list 【dumplings, noodles, rice, buns, …】.

| List type | Attention to final list item | Total candidate attention | TV after removing list-attention edges |
|----|---:|---:|---:|
| **Canonical lists** | 24.09% | 75.60% | 7.2% |
| **Non-canonical lists** | 6.56% | 41.89% | 2.7% |

The results show that `L82.H18` has a substantially stronger effect on canonical lists: it assigns more attention to them, and removing its attention edges to the list produces greater TV than for non-canonical lists⁶.

Note 6

The more salient or familiar a list’s ordering relation is, the stronger the effect of `L82.H18`. For example, the numerical list `[1, 2, 3, 4]` produces a stronger effect than the seasonal list `[spring, summer, autumn, winter]`.

We further investigated whether this behavior depends on the elements’ intrinsic ordering or requires them to be presented in order. We completely reversed each list and constructed several shuffled arrangements, then observed `L82.H18`.

| List type | Arrangement | Attention to final list item | Total candidate attention | TV after removing list-attention edges |
|----|----|----|----|----|
| **Canonical lists** | Reversed | 11.27% | 62.43% | 4.8% |
|  | Shuffled | 17.07% | 68.90% | 8.2% |
| **Non-canonical lists** | Reversed | 10.65% | 44.76% | 2.6% |
|  | Shuffled | 7.54% | 44.40% | 2.5% |

Overall, list order has little effect. For non-canonical lists, attention and TV remain essentially unchanged regardless of order. Among canonical lists, only complete reversal produces a noticeable decline in TV; attention and TV otherwise stay relatively high across arrangements. This suggests that `L82.H18` can recognize a candidate set’s strict intrinsic order rather than relying only on the order in which the list appears in the prompt.

### 3.2 Candidate-Constraint Generalization Experiments

Next, we examine generalization across candidate constraints: when we add constraints to the full candidate set and change the set of allowed choices, does `L82.H18` shift its attention to the constraints defining that set? We first run a simple experiment, retaining the original `1–300` range while additionally requiring the selected number to be less than `150`:

请从1-300中小于150的数里随机选择一个，只输出这个数字

In English: “Randomly choose a number less than 150 from the range 1–300. Output only that number.”

At the final answer position, `L82.H18` assigns **83.42%** attention to `150` in the added constraint “less than 150,” while attention to the original upper bound `300` falls to **9.07%**. Removing this head’s write at the final answer position produces **44.4%** TV. This shows that its primary reading location shifts to the new candidate boundary, while its write still substantially affects the output probability distribution.

We then extend the experiment to different types of canonical lists (see [Section 3.1](#candidate-type-generalization-experiments)), constructing eight active subsets for each list. We explicitly specify the set allowed for the current task using a fixed template, translated here:

The full candidate set is 【…】. The allowed set for this task is 【…】. Randomly select one item from the allowed set and output only the answer.

For example, given the letter list 【A, B, C, D, E, F, G, H】, we additionally specify 【A, C, E】 as the allowed set. We separately measure the attention that `L82.H18` assigns at the final answer position to the restated active set and the original full set, and the TV after jointly removing its attention edges to both sets:

[TABLE]

Across list types, `L82.H18` assigns more attention on average to the active set than to the original full set, with overall means of **57.06%** and **17.17%**, respectively. This shows that additional constraints do redirect its primary attention toward the candidates currently allowed. Jointly removing its attention edges to the original full set and the active set produces an overall mean TV of **8.1%**, showing that reading candidate information from these two locations jointly influences the output probability distribution.

We further construct more complex rule-based list-selection tasks. Instead of directly listing the active candidate set, we provide the full list and a filtering rule, asking the model to choose any candidate satisfying that rule. The rules include internal interval, midpoint neighborhood, distance shell, closer to the end, compound exclusion, and pair-sum membership. The bar chart below shows mean `L82.H18` attention to the list and rule, together with the corresponding edge-ablation TV, for each task type.

![Six paired bar groups, ordered as internal interval, midpoint neighborhood, distance shell, closer to the end, compound exclusion, and pair-sum membership; solid bars show joint list-and-rule attention, and hatched bars show joint-edge ablation TV.](../../assets/figures/random-number/h18_rule_attention_and_tv.png)

Figure 5: Mean attention and joint-edge ablation TV for `L82.H18` across six rule-based selection tasks. Solid bars show total attention to list and rule tokens (left axis); hatched bars show full-vocabulary TV after removing the corresponding attention edges (right axis).

The results are similar to the earlier experiments: `L82.H18` allocates most of its attention to the list and rule descriptions. Removing its attention edges to these positions and renormalizing attention over the remaining sources produces mean TV values of **3.2%–10.0%** across task types, showing that these reads have a substantial causal effect on the output probability distribution.

In addition, during these experiments, we also observed functional differentiation between heads. Although the preceding experiments focus on `L82.H18`, it is not the only head with similar properties. For another head with similar functionality in the same layer, `L82.H29`⁸, we find different reading emphases in constrained-choice tasks: `L82.H18` focuses more on the constraints defining the candidate range, whereas `L82.H29` focuses more on the final candidate in the original list.

Note 8

This head was localized using a full-head screening and independent-validation approach similar to [Section 2.1](#head-localization). On the original `1–300` random-number task, its mean attention to `300` ranks first among all 6,016 heads. Its centered full-vocabulary logit \operatorname{RMS}\_{c}(h,l) ranks second among all heads.

The following prompt provides an intuitive example. We retain the original Chinese text to match the token positions in the experiment: “给定有序字母列表【A，B，C，D，E，F，G】。B、E与F均不可选择；请在其余字母中任选一项。只输出所选字母，不要解释。答案：” The prompt gives the ordered list `A–G`, excludes `B/E/F`, and asks for any remaining letter without explanation. At the same final answer position, `q61`, `L82.H18` assigns **57.60%** total attention to the constraint tokens (such as `B/E/F`). In contrast, `L82.H29` assigns **45.33%** attention to the final list item `G`. Removing their corresponding high-attention edges produces TV values of **23.9%** and **5.0%**, respectively.

![Two-row, ten-column attention heatmap: H18 focuses on B, E, and F in the constraint, while H29 peaks at G in the original list. Each column labels the letter and its original token position; both heads share a 0%–50% color scale.](../../assets/figures/random-number/h18_h29_exclusion_attention.png)

Figure 6: Attention from the final answer position `q61` for `L82.H18` (top row) and `L82.H29` (bottom row) on the same prompt. Each column is an actual source token: `A–G` from the original list on the left, and the repeated `B/E/F` from the constraint on the right. Original token positions appear below the letters.

This example demonstrates differences between heads in their reading locations and causal contributions. It suggests that the model contains multiple computational circuits carrying different aspects of information, which may converge in subsequent deeper computation to jointly influence the downstream task.

### 3.3 Task-Type Generalization Experiments

We next ask whether the behavior of `L82.H18` generalizes to downstream tasks rather than being limited to random selection. We keep the input range fixed at `1–300`, construct a variety of downstream tasks, and observe the head’s behavior. Initial experiments suggest that the head’s effectiveness is related to the number of valid candidates allowed by the task. Specifically, we divide the tasks into two groups:

- **Non-unique tasks**: the task admits multiple valid answers, and outputting any one of those candidates satisfies the task requirements.
- **Unique tasks**: the task has a unique valid answer, and only that candidate satisfies the task requirements.

| Task type | Attention to `300` | TV after removing the `L82.H18` attention edges |
|----|---:|---:|
| **Non-unique tasks** | 92.231% | 13.7% |
| **Unique tasks** | 58.862% | 0.5% |

In both groups, `L82.H18` shows substantial attention to `300`. However, ablating the head produces a substantial TV change only in the non-unique tasks⁹.

Note 9

Not every non-unique task produces a substantial TV change. For example, in the task “choose a number greater than 250 from 1–300,” the TV effect is only 0.3%. One plausible explanation is that although the task admits multiple valid candidates, the first output token of every valid candidate must be `2`, so it is unaffected.

Therefore, `L82.H18` cannot simply be described as a head that operates only in random-selection tasks. It still attends to the candidate boundary in unique-answer tasks, indicating that it transports range information to the answer position. It appears, however, that downstream tasks use this information strongly only when the answer has not yet been fixed and the model must still distribute probability among valid candidates¹⁰.

Note 10

In fact, I did not recognize this distinction at first and mistakenly thought that the head became effective whenever the output required global information. Further experiments ruled out that hypothesis. The mistaken inference arose because many non-unique tasks require global information to determine the valid candidates.

We further test this claim with a straightforward intuition: if the substantial effect of `L82.H18` indeed arises when the model remains uncertain about the answer, then the head should also have a substantial effect on a sufficiently difficult task with a unique answer whenever the model cannot infer that answer effectively on its own.

We therefore ran a controlled comparison using two classes of unique-answer tasks that both require global information, with one class designed to be **simple** and the other **difficult**. We kept their wording and length as similar as possible to reduce confounding from prompt length and presentation. As in the preceding experiments, the model remained in non-thinking mode throughout. We compared accuracy and list attention on ordered and shuffled lists, together with the TV produced by removing the attention edges to the list elements.

| Task difficulty | Ordered |  |  | Shuffled |  |  |
|----|----|----|----|----|----|----|
|  | Accuracy | List attention | List-edge TV | Accuracy | List attention | List-edge TV |
| **Simple tasks** | 100.0% | 10.08% | 0.24% | 98.0% | 2.47% | 0.24% |
| **Difficult tasks** | 52.5% | 66.17% | 5.39% | 52.0% | 60.38% | 4.40% |

The results are broadly consistent with this prediction. Accuracy on the simple tasks reaches **100%**, indicating that the model has already fixed the answer before responding; attention to the list is low, and list-edge TV is also only about **0.24%**. In contrast, accuracy on the difficult tasks is only about **52%**, indicating that the model cannot reliably infer the unique answer; here, list attention rises to roughly **60%**, while list-edge TV reaches **5.39%**. Thus, when the answer has not been successfully inferred, `L82.H18` continues to read more information from the original list and has a stronger causal effect: the candidate information it transports substantially alters the output probability distribution.

To test whether this relationship holds across a broader range of difficulty, we ran additional tasks and summarize each task’s success rate and list-edge TV in the scatter plot below. Each point represents one task. As the plot shows, more difficult tasks do indeed exhibit higher TV. Their correlation coefficient is -0.84.

Figure 7: Task success rate and `L82.H18` list-edge TV. The x-axis shows task success rate; the y-axis shows full-vocabulary TV after removing the final-position `L82.H18` attention edges to all list elements.

At this point, our claim about the conditions under which `L82.H18` becomes effective appears to have preliminary support.

### 3.4 Generalization to Long-Form Reasoning

The experiments in [Section 3.3](#sec-task-type-generalization) all disabled thinking mode. To test whether the claim above about the conditions under which `L82.H18` becomes effective generalizes to long-form reasoning, we enabled thinking mode, let the model first generate a complete reasoning trace and then output the answer directly when the answer phase began, and finally measured the metrics of `L82.H18` at the final answer position after `</think>`. We selected several representative tasks spanning multiple accuracy levels; the comparison between non-thinking and thinking modes is shown below.

| Task | no-think |  |  | think |  |  |
|----|----|----|----|----|----|----|
|  | Accuracy | List attention | List-edge TV | Accuracy | List attention | List-edge TV |
| Closest to endpoint geometric mean | 2.08% | 81.41% | 8.61% | 95.83% | 5.38% | 0.015% |
| Closest odd-position rank mean | 18.75% | 68.54% | 8.06% | 100% | 7.30% | 0.008% |
| Closest to mean rank | 22.92% | 72.08% | 6.93% | 100% | 6.01% | 0.007% |
| Item after the largest rank gap | 29.17% | 71.92% | 5.79% | 100% | 5.28% | 0.011% |
| Earliest canonical even-position item | 43.75% | 55.38% | 3.97% | 97.92% | 4.10% | 0.002% |
| Canonical item 2 | 70.83% | 54.34% | 3.15% | 100% | 11.24% | 0.003% |
| Displayed item 5 | 93.75% | 40.14% | 1.24% | 100% | 4.46% | 0.003% |

With thinking enabled, task accuracy rises to nearly 100%, while list-edge TV falls to almost 0% across all tasks; at the same time, attention assigned to the original list drops sharply. This mirrors the earlier observation: once long-form reasoning has inferred the answer, the model’s response relies very little on `L82.H18` to transport information from the original list¹¹.

Note 11

Another point worth noting is that the higher a task’s success rate, the less attention `L82.H18` assigns to the list. It is currently unclear whether reduced attention directly causes the smaller TV, or whether attention and TV changing together with task difficulty reflects a deeper circuit mechanism. I personally favor the latter interpretation.

Thinking mode, however, inserts a long reasoning trace between the input list and the final answer. The reduction in TV might therefore be caused merely by the increased distance between the list and the answer position. To rule out this explanation, we ran a controlled experiment that held the input list and task fixed while constructing four strictly length-matched combinations: the Cartesian product of **no-think** versus **forced-think** and **correct reasoning** versus a **length-matched placeholder**. Here, **forced-think** means enabling model reasoning and filling the reasoning segment with prepared information; **correct reasoning** is a reasoning process that helps derive the answer, whereas the **length-matched placeholder** contains task-irrelevant content of exactly the same length. The final answer position has the same sequence length in every condition.

| Condition | Exact-answer accuracy | Original-list attention | List-edge TV |
|----|---:|---:|---:|
| **No-think** + **correct information** | 100% | 5.897% | 0.140% |
| **No-think** + **length-matched placeholder** | 25.0% | 57.133% | 8.197% |
| **Forced-think** + **correct reasoning** | 100% | 6.358% | 0.009% |
| **Forced-think** + **length-matched placeholder** | 23.6% | 61.551% | 5.266% |

The length of the prompt or reasoning does not disable `L82.H18`: list-edge TV remains above 5% in both length-matched placeholder conditions. Only when the context includes reasoning sufficient to determine the answer does accuracy rise to 100% while list attention and TV fall sharply together. The change is therefore driven neither by long-range distance nor by whether thinking is enabled, but by whether the answer has already been fixed by the context.

These results further strengthen the argument in [Section 3.3](#sec-task-type-generalization): whenever the answer has not been successfully inferred or otherwise determined, the candidate information transported by `L82.H18` substantially affects the output distribution.

### 3.5 Cross-Head Functional Generalization Experiments

The preceding experiments focus primarily on `L82.H18`. As [Section 3.1](#candidate-type-generalization-experiments) shows, however, these heads have a strong causal effect only when the list has a pronounced internal order, which limits the range of settings in which they operate.

A natural question is whether lists with weaker order relations still recruit a family of heads with a similar function. From the perspective of head specialization and Transformer circuits, such heads should exist¹². We therefore test whether comparable heads can be found on general lists.

Note 12

At a higher level, I prefer to view this as a form of distributed information, especially as models become larger and more capable. Multiple heads may perform similar functions while differing in the specific settings in which they become active.

Following the setup of [Section 2](#the-head-that-writes-the-random-number-fingerprint), we construct random unordered lists of Chinese single-token candidates and run both random-choice and deterministic-choice tasks. We then scan all 6,016 query heads, ablating each output slice immediately before the attention `o_proj`. Based on the preceding observations, we define the selection score as

S(h,l)=\max\\\left(\Delta \operatorname{TV}(h,l),0\right)\times A(h,l),

where \Delta \operatorname{TV}(h,l) is the head-ablation TV difference between random and deterministic choice, and A(h,l) is attention to the candidate list under random choice. The score requires both candidate-list reading and a causal effect concentrated in the unresolved condition, reducing interference from biases such as syntactic or ICL heads.

The six highest-scoring heads are `L80.H42`, `L82.H21`, `L85.H49`, `L89.H53`, `L87.H14`, and `L88.H13`. They are also the only heads with S(h,l)\>1 pp.

![A 94-layer by 64-query-head heatmap of selection scores. High scores concentrate in the late layers, and the six highest-scoring strictly revalidated heads are labeled.](../../assets/figures/random-number/unordered_candidate_head_selection_heatmap.png)

Figure 8: Full-model heatmap of S(h,l), with the six highest-scoring heads labeled.

The higher-scoring heads all lie in deep layers, while shallow and middle-layer heads are almost entirely white. This is consistent with the view that the ability to aggregate information streams and influence downstream computation appears mainly in deep layers. Beyond the selected heads, many late-layer heads also have weaker but visible S(h,l) responses, producing a dispersed functional pattern. This is another instance of the distributed information described in [Note 12](#note-head-information-dispersion).

Further experiments show that these heads have effects similar to `L82.H18`: while the answer remains unresolved, they focus on the candidate list and their ablation substantially changes the output distribution; once the answer is determined, their causal effect nearly disappears. To keep this post readable, we do not detail those experiments here. In the next section, we use these more general heads in intervention experiments to examine their mechanism and information circuit.

## 4 Causal Interventions on the Candidate-Head Circuit

The experiments in [Section 3](#sec-head-generalization) identified a family of heads with similar functions and suggested an initial account of their role:

**When the answer has not yet been successfully inferred and the model must still choose among multiple candidates, downstream computation relies strongly on the candidate information transported by these heads.**

This behavior generalizes across candidate lists, constraints on the candidate set, task types, and answers that involve long-form thinking.

We can also view this result through the lens of Transformer circuits. A model’s forward pass is not determined by one signal: multiple circuits write different signals into the residual stream, and those signals jointly shape the output. One useful way to understand their interaction is:

Ordinarily, the strongest signal dominates the model’s behavior. Other signals remain present but are suppressed in the competition. If the dominant circuit weakens or is interrupted, a weaker circuit may gain relative weight and compensate for the missing information. This competition may be especially apparent in larger models whose information representations are more distributed.

From this perspective, the heads we found participate in a broad candidate-competition circuit. When the model can reason normally, the signal that determines the answer dominates, and removing the candidate heads produces little TV. When that reasoning signal weakens or its circuit is interrupted, the candidate heads’ writes gain causal influence, and their ablation produces much larger TV.

The experiments in [Section 3.4](#sec-long-thinking-generalization) and [Section 3.5](#sec-cross-head-generalization) provide initial support for this account. We now test it with active interventions at three points in the reasoning pathway: the input evidence, the intermediate reasoning chain, and the retrieval heads that carry the answer to the output.

### 4.1 Removing Input Evidence

One direct way to disrupt normal reasoning is to remove the decisive evidence from the input, thereby damaging the start of the reasoning pathway.

We construct counterfactual question-answer cases from DROP, BrowseComp-Plus, 2WikiMultiHopQA, and MuSiQue. Each question is rewritten so that the correct answer and other candidates are replaced by randomly generated temporary codes. Neither the spelling nor the numbers in the codes reveal a pattern, preventing the model from answering through parametric memory.

Each case has a **complete-evidence** version and a **missing-evidence** version. The question, candidate list, and correct code are identical. The sole difference is that the missing-evidence version removes the key input facts needed to identify the correct candidate uniquely. At the point where thinking ends and the first final-answer token begins, we jointly ablate the writes of the heads identified in [Section 3.5](#sec-cross-head-generalization) and measure the full-vocabulary TV between the original and intervened distributions.

| Dataset | **Success rate(complete → missing evidence)** | **Head-ablation TV(complete → missing evidence)** |
|----|---:|---:|
| DROP | **100%**→**7.69%** | **0.639%**→**8.129%** |
| BrowseComp-Plus | **90%**→**3.33%** | **1.797%**→**6.153%** |
| 2WikiMultiHopQA | **90.48%**→**0%** | **2.006%**→**6.732%** |
| MuSiQue | **94.44%**→**0%** | **0.130%**→**7.466%** |

The pattern is consistent across all four datasets. With complete evidence, success is close to **100%** and head-ablation TV is small. Removing decisive evidence sharply lowers success, leaving the model almost unable to infer the answer, while TV rises substantially. This is consistent with the candidate information carried by these heads gaining weight when the input to the normal reasoning circuit is cut off.

### 4.2 Removing the Key Reasoning Chain

Next we intervene on the model’s reasoning itself. We delete the part of the reasoning trace that identifies the correct answer and ask whether the candidate heads become more causally important. This intervention damages the middle of the reasoning pathway.

We use the same counterfactual datasets and candidate heads as above. For each case, we construct a **complete-reasoning** version and a **missing-reasoning** version. The complete version inserts several segments of agent-style reasoning generated by the model with full evidence. In the missing version, we delete the continuous span from the first to the last mention of the correct code within that reasoning. Both versions end with “I have finished reasoning and know the answer,” after which the model produces its final answer. Again, we measure the heads’ effect at the first final-answer token.

| Dataset | **Success rate(complete → missing reasoning)** | **Head-ablation TV(complete → missing reasoning)** |
|----|---:|---:|
| DROP | **100%**→**0%** | **1.133%**→**8.428%** |
| BrowseComp-Plus | **100%**→**0%** | **1.098%**→**5.304%** |
| 2WikiMultiHopQA | **100%**→**0%** | **0.910%**→**7.956%** |
| MuSiQue | **100%**→**0%** | **1.454%**→**7.282%** |

After part of the reasoning chain is removed, successful reasoning turns into failure on every trajectory. The mean TV of candidate-head ablation rises from **1.133%** to **7.277%**. This again matches the prediction: interrupting the middle of the reasoning circuit increases the causal weight of candidate information.

### 4.3 Removing Retrieval Heads

The preceding interventions damage the input and the reasoning chain. We now cut off the end of the reasoning circuit: keep the question and evidence intact, but ablate the retrieval heads that bring the answer from context to the output¹³. Although the evidence remains available, the model loses a pathway for using it. We test whether the candidate heads’ TV rises under this condition.

Note 13

For the definition and role of retrieval heads, see [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574). Some of my recent experiments suggest that, at the end of a reasoning circuit, retrieval heads transport an answer already identified by the model from context to the decoding position.

Following the method in that paper, we identify the top 20 retrieval heads, about 0.3% of all heads:

`L81.H53`, `L81.H61`, `L79.H52`, `L81.H34`, `L85.H29`, `L79.H59`, `L81.H51`, `L83.H08`, `L61.H51`, `L61.H61`, `L81.H54`, `L76.H39`, `L81.H45`, `L06.H16`, `L79.H63`, `L59.H59`, `L61.H54`, `L82.H25`, `L83.H05`, and `L83.H10`.

Let R denote these retrieval heads and C the previously identified candidate-information heads. We compare four conditions: the intact model (`clean`), removal of only the retrieval circuit (`−R`), removal of only the candidate circuit (`−C`), and removal of both (`−R−C`).

On the same counterfactual datasets, we retain only trajectories answered correctly under `clean` but incorrectly under `−R`. For each trajectory, we examine three TV quantities:

- \operatorname{TV}(\mathit{clean},-R) measures the effect of removing retrieval heads. Because the answer changes from correct to incorrect, we expect this value to be very large.
- \operatorname{TV}(\mathit{clean},-C) measures the effect of removing candidate heads. With the reasoning circuit intact, we expect it to be small.
- \operatorname{TV}(-R,-R-C) measures the effect of removing candidate heads after disrupting the reasoning circuit. We expect it to be substantially larger than \operatorname{TV}(\mathit{clean},-C).

| Dataset | \operatorname{TV}(\mathit{clean},-R) | \operatorname{TV}(\mathit{clean},-C) | \operatorname{TV}(-R,-R-C) |
|----|---:|---:|---:|
| DROP | **65.507%** | **2.442%** | **7.459%** |
| BrowseComp-Plus | **41.133%** | **2.376%** | **17.158%** |
| 2WikiMultiHopQA | **54.411%** | **1.868%** | **10.483%** |
| MuSiQue | **34.428%** | **2.713%** | **5.042%** |

The results match our expectation. \operatorname{TV}(\mathit{clean},-R) is very large on all four datasets, confirming that the retrieval pathway has been effectively disrupted. Once retrieval heads are ablated, candidate-head TV rises substantially, by between **2.329%** and **14.782%**. The candidate information gains influence after the answer-transport function at the end of the reasoning circuit has been cut off.

Together, these experiments further support competition among circuits. The candidate circuit is suppressed when the normal reasoning circuit is strong, but has a greater effect on the final output when that circuit fails.

## 5 Improving Reasoning by Detecting Head TV

The preceding experiments clarify the mechanism of these heads: their causal effect on the output distribution becomes much stronger when an answer has not been inferred reliably. This suggests a practical question. Can we use that signal to improve the model’s reasoning performance?

**After an ordinary reasoning pass produces an answer, measure the candidate heads’ TV. If it exceeds a threshold, spend additional computation to continue reasoning.**

We test this on DROP, BrowseComp-Plus, 2WikiMultiHopQA, and MuSiQue. An agent-style workflow asks the model to reason from incomplete information returned by Search. Once it has reasoned and produced an answer, we ablate candidate-head writes at the position where the final response begins and measure the full-vocabulary TV before and after ablation¹⁴.

Note 14

Only the final `\n\n` position before the formal answer needs intervention. Earlier tokens can reuse the prefix cache, keeping detection inexpensive. In our implementation, `llm.collective_rpc` in vLLM switches the model into the intervention state.

Based on [Section 4](#sec-candidate-circuit-interventions), we set the TV threshold empirically to **6%**. When TV exceeds it, the agent continues its workflow: Search supplies more information, and a new user prompt tells the model that its previous answer may lack evidence and asks it to reason further and revise the answer.

![Four groups of bars for 2WikiMultiHopQA, BrowseComp-Plus, DROP, and MuSiQue, comparing baseline success, the frequency of TV exceeding the threshold, and success after additional reasoning.](../../assets/figures/random-number/head_tv_gated_reasoning_improvement.png)

Figure 9: Head-TV-gated reasoning enhancement, showing baseline success rate, frequency of TV exceeding the threshold, and success rate after additional reasoning on each dataset.

Across all four datasets, TV identifies trajectories that the model has not reasoned through reliably, and the additional reasoning strategy improves success. Candidate-head TV can thus serve as a signal of unreliable reasoning and trigger a partial repair.

The method has an important limit. It detects unreliable reasoning when the model itself has not formed a stable circuit. If the model reaches a wrong answer through a reasoning path it considers reliable, Head TV may remain low and the error will be missed.

Trajectory A **Detected**

**Dataset** · **ID**  
MuSiQue · `4hop2__567956_39078_8987_8974`

Question  
Country B was the only communist country to have an embassy where?

Ground truth  
Alfredo Stroessner's Paraguay

Model’s reasoning  
“Given the available evidence, the answer might be Confederate Gen. John Bell Hood.”

Head TV**9.551%**

**Outcome:** The model guessed incorrectly and its reasoning was uncertain. High TV triggered additional reasoning, which produced the correct answer.

Trajectory B **Missed**

**Dataset** · **ID**  
DROP · `2e82b6a6-8afa-4c30-8bf7-8bd101f9e16e`

Question  
Which team scored more touchdowns in the fourth quarter?

Ground truth  
Chargers

Model’s reasoning  
“The Panthers had two touchdowns, by Gamble and Rosario, so the answer should be the Panthers.”

Head TV**3.169%**

**Outcome:** The model confidently gave a wrong answer. TV did not reach the threshold, so no extra reasoning was triggered.

These two real trajectories illustrate the boundary. In A, the model’s reasoning already contained a tentative guess. Head TV reached `9.551%`, so the system detected it and successfully triggered additional reasoning. In B, the model formed a coherent but incorrect reasoning chain. Despite the wrong answer, Head TV stayed below the threshold and the error was missed.

Head TV can thus serve as a proxy for unreliable reasoning. Uncertainty may also appear in a model’s natural-language reasoning, but our experiments suggest that Head TV captures signals not expressed in words. Other measures, such as output-token entropy, can also respond to unreliable reasoning. Yet such token-level proxies are local and often become informative only later in the thinking process; Head TV can detect uncertainty in the reasoning process as a whole.

[TABLE]

We also measure computation cost: tokens and time spent on baseline and extra reasoning, and the extra cost as a percentage of the baseline. Across all results, an additional **11.1%** in tokens and **14.9%** in time yield a **5.9%** relative performance gain. Head-TV detection itself takes only about **1.9 s** per trajectory on average. The extra reasoning remains somewhat costly, but detection is lightweight.

This experiment turns Head TV from a retrospective mechanistic measure into a useful reasoning gate. It does not generate a better answer directly. Rather, it helps an agent identify when reasoning is unreliable and decide when to change its subsequent strategy to obtain a more trustworthy answer.

## 6 Conclusion and Outlook

Starting from the stable fingerprint in a model’s random-number outputs, we identified a family of heads that transport candidate information. Their causal effect becomes pronounced when the answer has not been uniquely determined and the model must still choose among candidates. This role generalizes across tasks and phrasings, and appears consistently under several interventions that interrupt normal reasoning circuits. Finally, we used Head TV as a signal of unreliable reasoning to trigger additional computation and improve the model’s reasoning performance.

This study does not fully explain how the random-number fingerprint itself is formed. When I first noticed the LINUX DO post, I hoped to trace that formation. But I realized that without examining the training data or training dynamics, the question would be difficult to answer. Even after pretraining, a very small amount of trajectory fine-tuning can greatly change a model’s preferences, further complicating the analysis.

I therefore shifted from the origin of the fingerprint to the model’s information flow, gradually arriving at the results presented here. The work unfolded intermittently over about a month and a half. The most important observations for my own understanding were perhaps the dispersion of information and competition among circuits. The final reasoning-enhancement method is somewhat akin to J-Space: begin with an understanding of the model, then devise a way to constrain its behavior.

Many questions remain. Why does ablating a head have such a large causal effect after the reasoning circuit is interrupted? Softmax is sensitive to logit perturbations: a difference of one logit unit can change the probability distribution substantially. From the perspective of a linear map, however, a one-unit difference in the output logit is only a small perturbation. This seemingly counterintuitive relationship suggests that normal reasoning may somehow stabilize a decision boundary that would otherwise be fragile. Once the reasoning circuit is damaged, that fragility may become visible again.

Understanding such effects is part of understanding the model. So is asking whether the circuits identified here persist under sparse attention or linear attention. These questions remain for future work.

Back to top
