"""Plot existing LongBench-v2 results in the random-number article's style.

With --results-root, refresh from the ten existing JSONL files and reapply the
recorded user corrections. Otherwise redraw from the checked-in data.
No inference code is loaded.
"""

from argparse import ArgumentParser
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/dsa-auto-compact"
STEM = "v4-longbench-subtask-success"
BACKGROUND, INK, MUTED, GRID = "#f8f8f3", "#191a20", "#696d78", "#d9dbe2"
CATEGORIES = [
    ("Single-Document QA", "单文档问答", "Single-document\nQA", 175),
    ("Multi-Document QA", "多文档问答", "Multi-document\nQA", 124),
    ("Long In-context Learning", "长上下文样本学习", "Many-shot\nlearning", 74),
    ("Long-dialogue History Understanding", "长对话历史理解", "Long-dialogue\nhistory", 39),
    ("Code Repository Understanding", "代码仓库理解", "Code repository\nunderstanding", 31),
    ("Long Structured Data Understanding", "长结构化数据理解", "Long structured\ndata", 30),
    ("TOTAL", "合计", "Overall", 473),
]
CELLS = [("baseline_100", 1.0), ("a_080", 0.8), ("a_060", 0.6),
         ("a_040", 0.4), ("a_020", 0.2)]
CAMPAIGNS = [
    ("deepseek_v4_longbench_v2_hardlong180_vllm_a_e1_v1", 180),
    ("deepseek_v4_longbench_v2_remaining293_vllm_baseline_a060_e1060_v1", 293),
]


def collect(results_root):
    sources, series, baseline = [], [], None
    for cell, ratio in CELLS:
        rows = []
        for campaign, expected in CAMPAIGNS:
            relative = Path("results") / campaign / cell / "results.jsonl"
            raw = (results_root / relative).read_bytes()
            part = [json.loads(line) for line in raw.splitlines() if line.strip()]
            assert len(part) == expected
            sources.append({"path": str(relative), "sha256": hashlib.sha256(raw).hexdigest()})
            rows.extend(part)
        by_id = {row["source_id"]: row for row in rows}
        assert len(rows) == len(by_id) == 473
        assert all(row["ratio"] == ratio and not row.get("error") for row in rows)
        assert all(row["method"] == ("baseline" if ratio == 1 else "A") for row in rows)
        if baseline is None:
            baseline = by_id
        else:
            assert by_id.keys() == baseline.keys()
            for source_id, row in by_id.items():
                for field in ("answer", "document_tokens", "prompt_tokens"):
                    assert row[field] == baseline[source_id][field]
                assert row["dataset_metadata"]["domain"] == baseline[source_id]["dataset_metadata"]["domain"]
        counts = Counter(row["dataset_metadata"]["domain"] for row in rows)
        correct = []
        for domain, _, _, expected in CATEGORIES:
            subset = rows if domain == "TOTAL" else [r for r in rows if r["dataset_metadata"]["domain"] == domain]
            assert len(subset) == expected
            if domain != "TOTAL":
                assert counts[domain] == expected
            assert all(isinstance(row["evaluation"]["final_answer_match"], bool) for row in subset)
            correct.append(sum(row["evaluation"]["final_answer_match"] for row in subset))
        series.append({"cell": cell, "target_retention": ratio, "correct": correct})
    return {
        "metric": "evaluation.final_answer_match",
        "categories": [{"id": d, "label": zh, "n": n} for d, zh, _, n in CATEGORIES],
        "series": series,
        "sources": sources,
    }


def apply_user_corrections(data):
    category_indices = {c["id"]: i for i, c in enumerate(data["categories"])}
    series_by_cell = {s["cell"]: s for s in data["series"]}
    for correction in data.get("user_corrections", []):
        index = category_indices[correction["category"]]
        assert index < 6, "Overall is derived from the six subtasks"
        series_by_cell[correction["cell"]]["correct"][index] = correction["correct"]
    for series in data["series"]:
        series["correct"][-1] = sum(series["correct"][:6])


def validate(data):
    assert [c["id"] for c in data["categories"]] == [c[0] for c in CATEGORIES]
    assert [c["n"] for c in data["categories"]] == [c[3] for c in CATEGORIES]
    assert [s["target_retention"] for s in data["series"]] == [r for _, r in CELLS]
    for series in data["series"]:
        assert len(series["correct"]) == 7
        assert all(isinstance(k, int) and 0 <= k <= c["n"] for k, c in zip(series["correct"], data["categories"]))
        assert sum(series["correct"][:6]) == series["correct"][6]


def draw(data):
    # Match Figure 4 (plot_h18_attention_intervention.py) in the
    # random-number article, including its per-element font sizes.
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND, "font.family": "DejaVu Sans",
        "font.size": 12.5, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "path", "svg.hashsalt": STEM,
    })
    figure, axis = plt.subplots(figsize=(9.2, 6.0))
    # Seven task labels need three legend rows rather than Figure 4's one.
    figure.subplots_adjust(left=0.105, right=0.965, bottom=0.18, top=0.70)
    x = list(range(5))
    # Filled markers and colors distinguish the six identically dashed tasks.
    # The pooled total uses a solid rose-colored line.
    styles = [
        ("#5363ee", "o"),
        ("#d66f4d", "s"),
        ("#249b83", "D"),
        ("#9472b6", "^"),
        ("#3b91b7", "v"),
        ("#7f8792", "P"),
        ("#b64b76", "X"),
    ]
    for index, (category, style) in enumerate(zip(data["categories"], styles)):
        color, marker = style
        y = [100 * series["correct"][index] / category["n"] for series in data["series"]]
        label = CATEGORIES[index][2].replace("\n", " ")
        overall = category["id"] == "TOTAL"
        line, = axis.plot(x, y, label=label, color=color,
                          linewidth=2.6 if overall else 1.9,
                          linestyle="-" if overall else (0, (5, 2)),
                          marker=marker, markersize=7.2,
                          markerfacecolor=color,
                          markeredgewidth=1.45,
                          zorder=3 + index)
        line.set_gid(f"longbench-task-{index + 1}")
    figure.suptitle("LongBench-v2 Success Rate vs. Retention Rate", y=0.94,
                   fontsize=18, fontweight="bold", color=INK)
    axis.set_ylabel("Success rate", labelpad=10, fontsize=12.5, fontweight="bold")
    axis.set_xlabel("Target retention rate", labelpad=10, fontsize=12.5, fontweight="bold")
    axis.set_xticks(x, ["100%\n(full cache)", "80%", "60%", "40%", "20%"])
    axis.set_xlim(-0.18, 4.18)
    axis.set_ylim(30, 75)
    axis.set_yticks([30, 40, 50, 60, 70])
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.tick_params(axis="x", length=0, pad=9, labelsize=11.5, colors=INK)
    axis.tick_params(axis="y", length=3, width=0.8, labelsize=11, colors=INK)
    axis.grid(axis="y", color=GRID, linewidth=0.85, alpha=0.78)
    axis.set_axisbelow(True)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK)
        axis.spines[side].set_linewidth(0.8)
    # Matplotlib fills legend columns first; reorder handles to read by rows.
    handles, labels = axis.get_legend_handles_labels()
    order = [i for column in range(3) for i in range(column, len(handles), 3)]
    axis.legend([handles[i] for i in order], [labels[i] for i in order],
                loc="lower left", bbox_to_anchor=(0.0, 1.025), ncols=3,
                frameon=False, fontsize=9.5, handlelength=2.3,
                columnspacing=1.05)
    return figure


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUTPUT / f"{STEM}.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.results_root:
        previous = json.loads(args.data.read_text()) if args.data.exists() else {}
        data = collect(args.results_root)
        if previous.get("user_corrections"):
            data["user_corrections"] = previous["user_corrections"]
        apply_user_corrections(data)
        validate(data)
        args.data.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    else:
        data = json.loads(args.data.read_text())
        apply_user_corrections(data)
        validate(data)
    figure = draw(data)
    description = "LongBench-v2: five retention rates on the x-axis; seven curves for six subtasks and the pooled 473-question total."
    figure.savefig(args.output_dir / f"{STEM}.png", dpi=240,
                   metadata={"Software": Path(__file__).name, "Description": description})
    figure.savefig(args.output_dir / f"{STEM}.svg",
                   metadata={"Creator": Path(__file__).name, "Date": None,
                             "Title": "LongBench-v2 Success Rate vs. Retention Rate", "Description": description})
    plt.close(figure)
    with (args.output_dir / f"{STEM}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_retention", "task", "task_zh", "correct", "total", "success_rate_pct"])
        for series in data["series"]:
            for category, correct in zip(data["categories"], series["correct"]):
                writer.writerow([series["target_retention"], category["id"], category["label"],
                                 correct, category["n"], 100 * correct / category["n"]])
    print(json.dumps({"figure": str(args.output_dir / f"{STEM}.svg"),
                      "series": len(data["categories"]), "points_per_series": 5,
                      "total_correct": [s["correct"][-1] for s in data["series"]]}))


if __name__ == "__main__":
    main()
