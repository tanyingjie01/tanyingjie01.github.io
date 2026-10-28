"""Recompute the Chinese article's query-free V4 results from local records.

Only saved JSON/CPU tensors are read; this script does not run inference.
Run in a PyTorch environment with --results-root pointing to Cache_compression.
"""
from argparse import ArgumentParser
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


SITE = Path(__file__).resolve().parents[1]
HEADS = [[20, 5], [20, 30], [20, 31], [26, 58]]
HEAD_NAMES = [f"MLA:L{layer}-H{head}" for layer, head in HEADS]
R3 = "deepseek_v4_longbench_v2_473_a60_single_prefill_mla128_top24_v1"
H5 = "deepseek_v4_a60_longbench_head_sweep_v2"
INFINITE = "deepseek_v4_infinitebench_701_baseline_a60_r3_1630_24_v1"
CASES = ["5b450986c9c3415abd09fbaa", "c3737bdaf751152684d697ef"]


def collect(root):
    sources = {}

    def source(relative):
        path = root / relative
        with path.open('rb') as handle:
            sources[relative] = hashlib.file_digest(handle, 'sha256').hexdigest()
        return path

    def read_json(relative):
        return json.loads(source(relative).read_text())

    def read_rows(relative):
        with source(relative).open() as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def combined(directories, cell):
        rows = []
        for directory in directories:
            rows += read_rows(f"results/{directory}/{cell}/results.jsonl")
        assert len(rows) == len({r['prompt_id'] for r in rows}) == 473
        return rows

    aware = ["deepseek_v4_longbench_v2_hardlong180_vllm_a_e1_v1",
             "deepseek_v4_longbench_v2_remaining293_vllm_baseline_a060_e1060_v1"]
    generic = ["deepseek_v4_longbench_v2_hardlong180_vllm_a_e1_generic_document_study_v1",
               "deepseek_v4_longbench_v2_remaining293_vllm_a_e1_generic_document_study_v1"]
    groups = {
        "full": combined(aware, "baseline_100"),
        "task_aware_60": combined(aware, "a_060"),
        "query_free_60": combined(generic, "a_060"),
        "query_free_recovery": combined([R3], "a_060_mla128_top24_additive"),
    }
    indexed = {name: {r['prompt_id']: r for r in rows} for name, rows in groups.items()}
    original = indexed['full']
    for name, rows in groups.items():
        assert indexed[name].keys() == original.keys()
        for row in rows:
            previous = original[row['prompt_id']]
            assert row['document_tokens'] == previous['document_tokens']
            assert row['evaluation']['expected_choice'] == previous['evaluation']['expected_choice']
            if name != 'full':
                ref = row['compression']['reference']
                assert ref['uses_literal_query'] == (name == 'task_aware_60')
                if name != 'task_aware_60':
                    assert ref['self_study_instruction_tokens'] == 93
                    assert ref['reference_suffix_tokens'] == 1630
            if name == 'query_free_recovery':
                verify_recovery(row)

    def summarize(rows, full=False):
        correct = sum(r['evaluation']['final_answer_match'] for r in rows)
        return dict(n=len(rows), correct=correct, accuracy=correct / len(rows),
                    mean_document_retention=1.0 if full else mean(
                        r['compression']['plan']['effective_raw_ratio'] for r in rows))

    longbench = {'cells': {name: summarize(rows, name == 'full') for name, rows in groups.items()}}
    longbench['domains'] = {}
    for domain in sorted({r['dataset_metadata']['domain'] for r in groups['full']}):
        longbench['domains'][domain] = {
            name: summarize([r for r in rows if r['dataset_metadata']['domain'] == domain], name == 'full')
            for name, rows in groups.items()}

    def paired(before, after):
        counts = Counter((indexed[before][pid]['evaluation']['final_answer_match'],
                          indexed[after][pid]['evaluation']['final_answer_match']) for pid in original)
        return {'right_to_wrong': counts[True, False], 'wrong_to_right': counts[False, True],
                'both_right': counts[True, True], 'both_wrong': counts[False, False]}

    longbench['full_to_query_free'] = paired('full', 'query_free_60')
    longbench['query_free_to_recovery'] = paired('query_free_60', 'query_free_recovery')
    assert [longbench['cells'][k]['correct'] for k in groups] == [246, 246, 244, 258]

    # Same-cohort sensitivity comparisons, including each setting's own S/s.
    sweep60 = 'deepseek_v4_longbench_v2_473_a60_reference_recovery_sweep_v1'
    sweep50 = 'deepseek_v4_longbench_v2_473_a50_mla128_top24_additive_suffix1024_2048_v1'
    sweep50short = 'deepseek_v4_longbench_v2_473_a50_mla128_top24_additive_suffix512_v1'
    settings = [
        (R3, 'a_060_mla128_top24_additive', .6, 1630, 24),
        (sweep60, 'a_060_mla128_top24_additive_ref1024', .6, 1024, 24),
        (sweep60, 'a_060_mla128_top16_additive_ref1024', .6, 1024, 16),
        (sweep60, 'a_060_mla128_top24_additive_ref512', .6, 512, 24),
        (sweep50short, 'a_050_mla128_top24_additive_ref512', .5, 512, 24),
        (sweep50, 'a_050_mla128_top24_additive_ref1024', .5, 1024, 24),
        (sweep50, 'a_050_mla128_top24_additive_ref2048', .5, 2048, 24),
    ]
    longbench['sensitivity'] = []
    for directory, cell, ratio, length, windows in settings:
        rows = groups['query_free_recovery'] if directory == R3 else read_rows(
            f'results/{directory}/{cell}/results.jsonl')
        assert len(rows) == len({r['prompt_id'] for r in rows}) == 473
        assert {r['prompt_id'] for r in rows} == original.keys()
        for row in rows:
            assert row['ratio'] == ratio
            assert row['document_tokens'] == original[row['prompt_id']]['document_tokens']
            assert row['evaluation']['expected_choice'] == original[row['prompt_id']]['evaluation']['expected_choice']
            ref = row['compression']['reference']
            assert ref['reference_suffix_tokens'] == length
            assert ref['self_study_instruction_tokens'] == 93 and ref['chat_tail_tokens'] == 2
            assert row['compression']['reference_query_budget'] == (1535 if length == 1630 else length)
            if length != 1630:
                assert ref['all_reference_positions_scored']
            verify_recovery(row, windows=windows)
        longbench['sensitivity'].append(dict(base_retention=ratio, reference_tokens=length,
            scored_queries=1535 if length == 1630 else length, recovery_windows=windows,
            window_tokens=128, **summarize(rows),
            mean_reference_scoring_seconds=mean(r['timings']['reference_pass_and_scoring_wall_seconds'] for r in rows),
            mean_pipeline_seconds=mean(r['timings']['pipeline_wall_seconds'] for r in rows)))

    manifest = read_json(f"data/{H5}/manifest.json")
    localization = dict(prompt_count=manifest['prompt_count'], dataset_counts=manifest['counts'],
                        heads=HEADS, measured_layers=manifest['experimental_settings']['measured_layers'])
    path = source(f"results/{H5}/analysis/ensemble_summary.csv")
    localization['evidence_window_metrics'] = [r for r in csv.DictReader(path.open())
        if r['policy'] == 'new_LB_top4:mean_z' and r['resolution'] == '128'
        and r['budget_tokens'] == '3072' and r['scope'] in
        ['primary:LongBench-v2', 'primary:BrowseComp-Plus-near128K']]

    # Audit the two examples against the actual frozen base and final native rows.
    import torch
    labels = read_json(f"data/{H5}/offline_evidence_annotations.json")['labels']
    prompts = {r['prompt_id']: r for r in read_rows(f"data/{H5}/prompts.jsonl")}
    examples = []
    for pid in CASES:
        result = indexed['query_free_recovery'][pid]
        path = Path(result['compression']['workers'][0]['selection_artifact'])
        source(str(path.relative_to(root)))
        saved = torch.load(path, map_location='cpu', weights_only=True)
        layer = saved['layers'][20]
        base = set(layer['base_selected_document_origins'].tolist())
        final = set(layer['final_selected_document_origins'].tolist())
        spans = []
        for evidence in labels[pid]['evidence']:
            start, end = evidence['token_span']
            native_rows = set(range(start // 4, (end + 3) // 4))
            spans.append(dict(label=evidence['label'], token_span=[start, end],
                              L20_C4_rows=len(native_rows), missing_base=len(native_rows - base),
                              missing_final=len(native_rows - final)))
        assert all(s['missing_final'] == 0 for s in spans)
        examples.append(dict(prompt_id=pid, query=prompts[pid]['query'], evidence=spans,
            correct_choice=original[pid]['evaluation']['expected_choice'],
            outputs={name: dict(choice=indexed[name][pid]['evaluation']['predicted_choice'],
                                completion=indexed[name][pid]['completion']) for name in groups},
            selected_token_spans=result['query_guided_recovery']['selection']['selected_token_spans']))

    config = read_json(f"results/{INFINITE}/manifest.json")
    expected_heads = [{'kind': 'mla', 'layer': l, 'head': h} for l, h in HEADS]
    assert config['runtime']['query_head_ensemble'] == expected_heads
    assert (config['reference_total_tokens'], config['recovery_windows'],
            config['recovery_window_tokens'], config['runtime']['max_new_tokens']) == (1630, 24, 128, 192)
    infinite = {'sample_count': 701, 'runtime': config['runtime'], 'tasks': {}, 'cells': {}}
    for setting in config['settings']:
        rows = read_rows(f"results/{INFINITE}/formal/{setting}/results.jsonl")
        assert len(rows) == len({r['prompt_id'] for r in rows}) == 701
        if setting != 'baseline':
            for row in rows:
                verify_recovery(row)
        else:
            original_infinite = {r['prompt_id']: r for r in rows}
        for row in rows:
            old = original_infinite[row['prompt_id']]
            assert row['input_row_sha256'] == old['input_row_sha256']
            assert row['input_prompt_sha256'] == old['input_prompt_sha256']
            assert row['document_tokens'] == old['document_tokens']
        infinite['cells'][setting] = dict(n=len(rows),
            mean_document_retention=1 if setting == 'baseline' else mean(
                r['compression']['plan']['effective_raw_ratio'] for r in rows),
            mean_pipeline_seconds=mean(r['timings']['pipeline_wall_seconds'] for r in rows),
            context_token_range=[min(r['context_tokens'] for r in rows), max(r['context_tokens'] for r in rows)])
        for task, count in config['task_counts'].items():
            subset = [r for r in rows if r['infinitebench_task'] == task]
            assert len(subset) == count
            binary = [r['evaluation']['final_answer_match'] for r in subset
                      if r['evaluation']['final_answer_match'] is not None]
            infinite['tasks'].setdefault(task, {})[setting] = dict(n=len(subset),
                metric=subset[0]['evaluation']['metric'], mean_score=mean(r['evaluation']['score'] for r in subset),
                binary_correct=sum(binary) if binary else None,
                generation_limit_count=sum(r['generation_limit_reached'] for r in subset))
    return dict(schema='dsa-query-free-recovery-evidence-v1', longbench=longbench,
                localization=localization, examples=examples, infinitebench=infinite,
                sources=[dict(path=path, sha256=digest) for path, digest in sorted(sources.items())])


def verify_recovery(row, windows=24):
    recovery = row['query_guided_recovery']
    assert recovery['policy'] == 'additive'
    assert not recovery['answer_or_gold_used_for_selection']
    assert recovery['logical_probe_only_saw_frozen_base_selection']
    assert recovery['full_cache_preserved_until_final_copy']
    selection = recovery['selection']
    assert selection['head_names'] == HEAD_NAMES
    assert selection['requested_windows'] == windows and selection['window_tokens'] == 128
    assert row['compression']['reference']['uses_literal_query'] is False


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output', type=Path,
        default=SITE / 'assets/figures/dsa-auto-compact/query-free-recovery-evidence.json')
    args = parser.parse_args()
    report = collect(args.results_root.resolve())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'longbench': report['longbench']['cells'],
                      'infinitebench': report['infinitebench']['cells']}, ensure_ascii=False))
