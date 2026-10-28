"""Recompute the Chinese article's GLM / V4.1 tables from local CPU records."""
from argparse import ArgumentParser
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

SITE = Path(__file__).resolve().parents[1]
CAMPAIGNS = {
    'glm_longbench192': 'glm53_longbench_v2_473_a60_r3_additive_v1',
    'glm_longbench512': 'glm53_longbench_v2_473_a60_r3_additive_max512_v1',
    'glm_infinitebench512': 'glm53_infinitebench_701_a60_r3_additive_max512_v1',
    'v41_longbench1630': 'deepseek_v41_longbench_v2_473_a60_r3_logical_additive_max192_v1',
    'v41_longbench_fixed': 'deepseek_v41_longbench_v2_473_a60_r3_ref1024_2048_optimized_max192_v1',
}


def collect(source):
    sources, campaigns, baselines = {}, {}, {}

    def track(path):
        sources[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read_json(path):
        return json.loads(track(path).read_text())

    for label, directory in CAMPAIGNS.items():
        root = source / 'results' / directory
        config, stored = read_json(root / 'config.json'), read_json(root / 'SUMMARY.json')
        assert stored['complete']
        cells, raw = {}, {}
        for cell in config['cells']:
            path = track(root / cell / 'results.jsonl')
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            assert len(rows) == len({r['prompt_id'] for r in rows}) == config['prompts_per_cell']
            assert all(r['complete'] and not r['preflight'] and not r['gold_used_online'] for r in rows)
            raw[cell] = {r['prompt_id']: r for r in rows}
            stats = {
                'n': len(rows),
                'mean_pipeline_seconds': mean(r['timings']['pipeline_wall_seconds'] for r in rows),
                'mean_generated_tokens': mean(r['generated_tokens'] for r in rows),
                'stopped_by_length_count': sum(r['finish_reason'] == 'length' for r in rows),
                'mean_stage_seconds': {
                    key: mean(r['timings'][key] for r in rows)
                    for key in rows[0]['timings']
                },
                'reference_tokens': sorted({r['reference_tokens'] for r in rows}),
                'reference_instruction_tokens': sorted({r['reference_instruction_tokens'] for r in rows}),
                'truncated_inputs': sum(bool(r['truncation']) for r in rows),
            }
            if label.startswith('glm_'):
                stats['mean_document_equivalent_retention'] = mean(
                    1 - (r['original_prefix_tokens'] - r['compacted_prefix_tokens']) /
                    (r['document_span'][1] - r['document_span'][0]) for r in rows)
                if cell != 'baseline':
                    assert all(r['page_release']['fixed_state_groups_preserved'] for r in rows)
                    stats['mean_growing_prefix_pages_saved'] = mean(
                        r['page_release']['growing_prefix_pages_saved'] for r in rows)
            else:
                assert all(not r['physical_compaction'] and
                           r['physical_prefix_tokens'] == r['original_prefix_tokens'] for r in rows)
                stats['physical_prefix_unchanged'] = True
            if 'infinitebench' in label:
                tasks = sorted({r['infinitebench_task'] for r in rows})
                stats['tasks'] = {}
                for task in tasks:
                    subset = [r for r in rows if r['infinitebench_task'] == task]
                    stats['tasks'][task] = {
                        'n': len(subset), 'metric': subset[0]['evaluation']['metric'],
                        'mean_score': mean(r['evaluation']['score'] for r in subset),
                        'correct': sum(bool(r['evaluation']['final_answer_match']) for r in subset),
                        'stopped_by_length_count': sum(r['finish_reason'] == 'length' for r in subset),
                    }
                stats['mixed_task_native_mean_score'] = mean(r['evaluation']['score'] for r in rows)
                published = stored['cells'][cell]['overall']
                assert abs(stats['mixed_task_native_mean_score'] - published['mean_score']) < 1e-10
            else:
                stats['correct'] = sum(bool(r['evaluation']['final_answer_match']) for r in rows)
                stats['accuracy'] = stats['correct'] / len(rows)
                stats['domains'] = {
                    domain: {'n': len(subset), 'correct': sum(
                        bool(r['evaluation']['final_answer_match']) for r in subset)}
                    for domain in sorted({r['dataset_metadata']['domain'] for r in rows})
                    for subset in [[r for r in rows if r['dataset_metadata']['domain'] == domain]]
                }
                published = stored['cells'][cell]['groups']['overall']
                assert stats['correct'] == published['correct']
            assert abs(stats['mean_pipeline_seconds'] - published['mean_pipeline_seconds']) < 1e-9
            cells[cell] = stats
        if 'baseline' in raw:
            baseline = raw['baseline']
            baselines[label] = baseline
        elif label == 'v41_longbench_fixed':
            baseline = baselines['v41_longbench1630']
        for cell, rows in raw.items():
            assert rows.keys() == baseline.keys()
            assert all(rows[pid]['input_ids_sha256'] == baseline[pid]['input_ids_sha256'] for pid in rows)
            if cell != 'baseline' and 'infinitebench' not in label:
                cells[cell]['paired_vs_full'] = dict(Counter(
                    f"{int(bool(baseline[pid]['evaluation']['final_answer_match']))}->"
                    f"{int(bool(row['evaluation']['final_answer_match']))}"
                    for pid, row in rows.items()))
        campaigns[label] = {'config': config, 'cells': cells}

    glm_inputs = baselines['glm_longbench512']
    for label in ['glm_longbench192', 'v41_longbench1630']:
        assert baselines[label].keys() == glm_inputs.keys()
    assert all(row['input_ids_sha256'] == glm_inputs[pid]['input_ids_sha256']
               for pid, row in baselines['glm_longbench192'].items())

    heads = {}
    for model, name in [('glm53', 'native_h5_glm53_v1'), ('v41', 'native_h5_v41_v1')]:
        root = source / 'results' / name
        summary = read_json(root / 'analysis/summary.json')
        manifest = read_json(root / 'manifest.json')
        assert summary['complete']
        with track(root / 'analysis/ensemble_per_prompt.csv').open() as handle:
            rows = [r for r in csv.DictReader(handle) if r['policy'] == 'new_LB_top4:mean_z'
                    and int(r['resolution']) == 128 and int(r['budget_tokens']) == 3072]
        assert len(rows) == len({r['prompt_id'] for r in rows}) == 152
        groups = {}
        for key, subset in [
            ('primary:LongBench-v2', [r for r in rows if r['dataset'] == 'LongBench-v2' and r['primary'] == 'True']),
            ('dataset:BrowseComp-Plus-near128K', [r for r in rows if r['dataset'] == 'BrowseComp-Plus-near128K']),
        ]:
            groups[key] = {'n': len(subset), 'hits': sum(float(r['hit']) for r in subset)}
            expected = summary['recommended_ensemble_metrics'][key]
            assert len(subset) == expected['n']
            assert abs(groups[key]['hits'] / len(subset) - expected['hit']) < 1e-10
        lb_ids = set(baselines['glm_longbench512'])
        overlap = sum(pid in lb_ids for pid in manifest['prompt_ids'])
        assert overlap == 68
        heads[model] = {'heads_zero_based': summary['recommended_heads'], 'groups': groups,
                        'discovery_inputs': summary['completed_gpu_prompts'],
                        'labelled_inputs': len(rows), 'longbench_overlap': overlap,
                        'protocol': manifest['protocol'],
                        'max_native_kernel_relative_error': summary['max_native_kernel_relative_error']}
    for name in ['glm_a_r3.py', 'glm_a_r3_engine.py', 'v41_a_r3.py', 'v41_a_r3_optimized.py', 'native_h5_worker.py']:
        track(source / 'experiments' / name)
    return {'campaigns': campaigns, 'heads': heads, 'source_sha256': sources,
            'retention_definition': 'Per-question document-equivalent ratio after physical GLM page alignment; excludes fixed KDA and pooling-tail states. V4.1 keeps the full physical prefix.',
            'length_stop_definition': 'finish_reason == length; not generated_tokens == max_new_tokens'}


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.source_root.resolve())
    output = SITE / 'assets/figures/dsa-auto-compact/generalization-evidence.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {output}: {len(result["campaigns"])} campaigns, {len(result["source_sha256"])} source checksums.')
