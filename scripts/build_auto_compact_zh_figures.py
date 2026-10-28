"""Audit local V3.2 and GLM records and render the Chinese article's figures.

Run with --results-root pointing to Cache_compression, in a matplotlib environment.
Only JSON records are read; models, tensors and inference services are untouched.
"""
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/figures/dsa-auto-compact'
BLUE, GREEN, GREY, ORANGE = '#5363ee', '#2f9d78', '#9397a8', '#d86f4d'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    docs = [r['document_span'][1] - r['document_span'][0] for r in rows]
    timing_keys = sorted(set().union(*(r['timings'] for r in rows)))
    result = {
        'n': len(rows),
        'correct': sum(bool(r['evaluation']['final_answer_match']) for r in rows),
        'mean_document_equivalent_retention': mean(
            1 - (r['original_prefix_tokens'] - r['compacted_prefix_tokens']) / n
            for r, n in zip(rows, docs)),
        'document_tokens_min': min(docs), 'document_tokens_max': max(docs),
        'document_tokens_mean': mean(docs),
        'truncated_input_count': sum(bool(r['truncation']) for r in rows),
        'generation_limit_count': sum(r['generated_tokens'] == r['max_new_tokens'] for r in rows),
        'mean_generated_tokens': mean(r['generated_tokens'] for r in rows),
        'mean_timings_seconds': {
            k: mean(r['timings'][k] for r in rows if k in r['timings']) for k in timing_keys},
        'reference_token_counts': sorted(set(r['reference_tokens'] for r in rows)),
        'domains': {},
    }
    result['accuracy'] = result['correct'] / len(rows)
    for domain in sorted(set(r['dataset_metadata'].get('domain', '') for r in rows)):
        subset = [r for r in rows if r['dataset_metadata'].get('domain', '') == domain]
        result['domains'][domain] = {
            'n': len(subset),
            'correct': sum(bool(r['evaluation']['final_answer_match']) for r in subset)}
    if 'infinitebench_task' in rows[0]:
        result['tasks'] = {}
        for task in sorted(set(r['infinitebench_task'] for r in rows)):
            subset = [r for r in rows if r['infinitebench_task'] == task]
            result['tasks'][task] = {
                'n': len(subset), 'metric': subset[0]['evaluation']['metric'],
                'mean_score': mean(r['evaluation']['score'] for r in subset),
            }
        result['mean_task_native_score'] = mean(r['evaluation']['score'] for r in rows)
        # The mixed-task average is a score, not an accuracy. Summaries have no
        # binary target; don't publish an overall binary metric for this cohort.
        del result['correct'], result['accuracy'], result['domains']
    return result


def collect_v32(source, add_source):
    path = source / 'results/deepseek_v32_browsecomp_plus_normal100_a_e1_v2_causal'
    config = json.loads((path / 'config.json').read_text())
    stored = json.loads((path / 'summary.json').read_text())
    audit = json.loads((path / 'chunk_answer_audit_summary.json').read_text())
    for name in ['config.json', 'summary.json', 'input_prompts.jsonl',
                 'results.jsonl', 'chunk_answer_audit_summary.json']:
        add_source(path / name)
    assert digest(path / 'input_prompts.jsonl') == config['prompts_sha256']
    inputs = [json.loads(line) for line in (path / 'input_prompts.jsonl').read_text().splitlines()]
    assert len(inputs) == len({r['source_id'] for r in inputs}) == 100
    chunks = [c for r in inputs for c in r['chunks']]
    assert len(chunks) == audit['source_document_chunks']['total']
    assert sum(c['metadata']['truncated'] for c in chunks) == audit['source_document_chunks']['truncated_total']
    records = [json.loads(line) for line in (path / 'results.jsonl').read_text().splitlines()]
    selected = [r for r in records if r['ablation'] in ('baseline', 'A')]
    baseline = {r['prompt_id']: r for r in selected if r['ablation'] == 'baseline'}
    assert len(baseline) == 100 and len(selected) == 500
    cells = {}
    for ratio in ['100', '080', '060', '040', '020']:
        rows = [r for r in selected if r['ratio_name'] == ratio]
        assert len(rows) == len({r['prompt_id'] for r in rows}) == 100
        assert {r['prompt_id'] for r in rows} == baseline.keys()
        for r in rows:
            original = baseline[r['prompt_id']]
            assert r['original_document_tokens'] == original['original_document_tokens']
            assert r['answer'] == original['answer']
            if ratio != '100':
                assert r['reference_query_sources']['uses_literal_query']
                assert r['reference_query_sources']['reference_query_mode'] == 'task_aware'
                assert r['compacted_layer_count'] == 61
        string_correct = sum(r['evaluation']['reference_string_match'] for r in rows)
        summary = next(g for g in stored['groups'] if g['ratio_name'] == ratio
                       and g['ablation'] == ('baseline' if ratio == '100' else 'A'))
        assert string_correct == summary['successes']
        cells[f'retention{ratio}'] = {
            'n': len(rows), 'target_retention': int(ratio)/100,
            'mean_document_retention': mean(r['compacted_document_tokens']/r['original_document_tokens'] for r in rows),
            'reference_string_correct': string_correct,
            'final_answer_correct': sum(r['evaluation']['final_answer_match'] for r in rows),
            'reference_queries_per_head': sorted(set(r['reference_queries_per_head'] for r in rows)),
            'mean_prefill_seconds': mean(r['prefill_seconds'] for r in rows),
            'mean_compaction_seconds': mean(r['compaction_seconds'] for r in rows),
            'mean_generation_seconds': mean(r['generation_seconds'] for r in rows),
            'mean_indexer_topk_recall': None if ratio == '100' else mean(r['mean_indexer_topk_recall'] for r in rows),
        }
        if ratio != '100':
            lengths = [r['reference_query_sources']['reference_suffix_tokens'] for r in rows]
            cells[f'retention{ratio}']['reference_suffix_tokens'] = {
                'min': min(lengths), 'max': max(lengths), 'mean': mean(lengths)}
    return {
        'model': 'DeepSeek-V3.2', 'dataset': 'BrowseComp-Plus/normal100',
        'reference_query_mode': config['algorithm']['reference_query_mode'],
        'generation': config['generation'], 'scheme': config['scheme'],
        'parallelism': 'Transformers TP8', 'compacted_layers': 61,
        'cache_chunk_size': config['algorithm']['cache_chunk_size'],
        'index_topk': config['algorithm']['dsa_index_topk'],
        'paired_inputs_verified': True, 'cells': cells,
        'document_tokens': audit['document_tokens'],
        'source_documents': audit['source_document_chunks'],
        'construction': 'All official gold/evidence documents plus official hard negatives; '
            'deterministic mixing; approximately 4096-token per-document clipping; '
            'positive clipping centered on the answer string when present.',
    }


def collect(source):
    snapshot = json.loads((OUT / 'evidence.json').read_text())
    deepseek = deepcopy(snapshot)
    deepseek['cells'] = {k: v for k, v in deepseek['cells'].items() if 'e1' not in k}
    deepseek['sources'] = [s for s in deepseek['sources']
                           if not any(p.startswith('e1_') for p in Path(s['path']).parts)]
    for entry in deepseek['sources']:
        if digest(source / entry['path']) != entry['sha256']:
            raise ValueError('DeepSeek source changed: ' + entry['path'])
    report = {
        'schema': 'dsa-auto-compact-zh-evidence-v2', 'snapshot_date': str(date.today()),
        'deepseek_v4': deepseek, 'glm53': {},
        'retention_note': 'GLM document equivalent retention = 1 - '
            '(original_prefix_tokens - compacted_prefix_tokens) / document_tokens. '
            'Includes scheduler padding; excludes fixed KDA and pooling-tail state bytes.',
        'sources': [],
    }

    def add_source(path):
        report['sources'].append({'path': str(path.relative_to(source)), 'sha256': digest(path)})

    report['deepseek_v32'] = collect_v32(source, add_source)

    campaigns = {
        'longbench192': ('glm53_longbench_v2_473_a60_r3_additive_v1', 473),
        'longbench512': ('glm53_longbench_v2_473_a60_r3_additive_max512_v1', 473),
        'infinitebench512': ('glm53_infinitebench_701_a60_r3_additive_max512_v1', 701),
    }
    retained_inputs = None
    for label, (directory, expected) in campaigns.items():
        path = source / 'results' / directory
        config = json.loads((path / 'config.json').read_text())
        stored = json.loads((path / 'SUMMARY.json').read_text())
        add_source(path / 'config.json')
        add_source(path / 'SUMMARY.json')
        groups = {}
        for cell in ['baseline', 'a60_r3_additive']:
            file = path / cell / 'results.jsonl'
            rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
            add_source(file)
            assert len(rows) == expected and len({r['prompt_id'] for r in rows}) == expected
            assert all(r['complete'] and not r['preflight'] for r in rows)
            assert all(r['max_new_tokens'] == config['max_new_tokens'] for r in rows)
            if cell != 'baseline':
                assert all(not r['gold_used_online'] for r in rows)
                assert all(r['original_document_prefill_count'] == 1 for r in rows)
                assert all(r['page_release']['fixed_state_groups_preserved'] for r in rows)
                for r in rows:
                    for phase in ['reference_workers', 'probe_workers']:
                        assert all(w['bitwise_state_restore'] for w in r['state_rollbacks'][phase])
            stats = summarize(rows)
            saved = stored['cells'][cell]
            assert saved['complete']
            saved_overall = saved.get('overall', saved.get('groups', {}).get('overall'))
            if label.startswith('longbench'):
                assert stats['correct'] == saved_overall['correct']
            else:
                assert abs(stats['mean_task_native_score'] - saved_overall['mean_score']) < 1e-12
                for task, values in stats['tasks'].items():
                    assert abs(values['mean_score'] - saved['tasks'][task]['mean_score']) < 1e-12
            groups[cell] = {r['prompt_id']: r for r in rows}
            report['glm53'].setdefault(label, {'config': config, 'cells': {}})['cells'][cell] = stats
        a, b = groups['baseline'], groups['a60_r3_additive']
        assert a.keys() == b.keys()
        for pid in a:
            for field in ['source_id', 'input_ids_sha256', 'document_span', 'max_new_tokens']:
                assert a[pid][field] == b[pid][field], (label, pid, field)
        current_inputs = {pid: r['input_ids_sha256'] for pid, r in a.items()}
        if label == 'longbench192':
            retained_inputs = current_inputs
        elif label == 'longbench512':
            assert current_inputs == retained_inputs
        counts = Counter((bool(a[k]['evaluation']['final_answer_match']),
                          bool(b[k]['evaluation']['final_answer_match'])) for k in a)
        report['glm53'][label]['paired_inputs_verified'] = True
        if label.startswith('longbench'):
            report['glm53'][label]['paired_outcomes'] = {
                'wrong_to_right': counts[(False, True)], 'right_to_wrong': counts[(True, False)],
                'both_right': counts[(True, True)], 'both_wrong': counts[(False, False)]}

    manifest = source / 'results/native_h5_glm53_v1/manifest.json'
    analysis = source / 'results/native_h5_glm53_v1/analysis/summary.json'
    add_source(manifest)
    add_source(analysis)
    discovery = json.loads(manifest.read_text())
    labels = json.loads(analysis.read_text())
    report['glm53']['head_discovery'] = {
        'scanned': len(discovery['prompt_ids']),
        'labelled': labels['analyzed_labelled_prompts'],
        'overlap_with_longbench473': len(set(discovery['prompt_ids']) & retained_inputs.keys()),
        'heads_zero_based': labels['recommended_heads'], 'independent_heldout_evaluation': False}
    (OUT / 'evidence-zh.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def finish(fig, name):
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', metadata={'Date': None})
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', dpi=170)
    plt.close(fig)


def figures(report):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#bfc2ce', 'text.color': '#191a20',
        'svg.fonttype': 'path', 'svg.hashsalt': 'dsa-auto-compact-zh',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    v32 = report['deepseek_v32']['cells']
    fig, (quality, routing) = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    ratios = [20, 40, 60, 80]
    for metric, label, color, marker in [
            ('reference_string_correct', 'Full-output answer match', BLUE, 'o'),
            ('final_answer_correct', 'Final Answer match', GREEN, 's')]:
        values = [v32[f'retention{r:03d}'][metric] for r in ratios]
        quality.plot(ratios, values, color=color, marker=marker, label=label)
        quality.axhline(v32['retention100'][metric], color=color, ls='--', alpha=.6)
        for x, y in zip(ratios, values):
            quality.annotate(str(y), (x, y), xytext=(0, 8 if marker == 'o' else -15),
                             textcoords='offset points', ha='center', color=color, fontsize=9)
    quality.set(xticks=ratios, xlim=(14, 86), ylim=(76, 92), xlabel='Target document retention (%)',
                ylabel='Answer match (%)', title='V3.2: BrowseComp-Plus normal100')
    quality.legend(frameon=False, loc='upper left', fontsize=9)
    overlaps = [100*v32[f'retention{r:03d}']['mean_indexer_topk_recall'] for r in ratios]
    routing.plot(ratios, overlaps, color=BLUE, marker='o')
    for x, y in zip(ratios, overlaps):
        routing.annotate(f'{y:.2f}%', (x, y), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
    routing.set(xticks=ratios, xlim=(14, 86), ylim=(55, 105), xlabel='Target document retention (%)',
                ylabel='Mean native Top-k overlap (%)', title='Retained routing candidates (reference pass)')
    for axis in (quality, routing):
        axis.grid(True, alpha=.16)
    finish(fig, 'v32-quality-zh')

    data = report['deepseek_v4']['cells']
    fig, (ax, curve) = plt.subplots(1, 2, figsize=(11, 4.1), layout='constrained')
    names = ['baseline', 'generic_a_060', 'recovery_060_ref1630_top24']
    bars = ax.bar(range(3), [100 * data[n]['accuracy'] for n in names], color=[GREY, BLUE, GREEN], width=.6)
    for bar, name in zip(bars, names):
        c = data[name]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .8,
                f"{100*c['accuracy']:.2f}%\n{c['correct']}/473", ha='center', fontsize=9)
    ax.set(xticks=range(3), xticklabels=['Full cache', 'Proposed\n60% target', 'Proposed +\nrecovery'],
           ylim=(0, 67), ylabel='Answer accuracy (%)', title='Generic memory: query initially unknown')
    ratios = [20, 40, 60, 80]
    values = [100 * data[f'task_a_{r:03d}']['accuracy'] for r in ratios]
    curve.plot(ratios, values, marker='o', color=BLUE, label='Proposed method')
    for x, y in zip(ratios, values):
        curve.annotate(f'{y:.2f}', (x, y), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
    curve.axhline(100*data['baseline']['accuracy'], color=GREY, ls='--', label='Full cache')
    curve.set(xlim=(14, 86), ylim=(48.5, 55), xticks=ratios, xlabel='Target document retention (%)',
              ylabel='Answer accuracy (%)', title='Task-aware reference: query already known')
    curve.legend(frameon=False, loc='upper left', fontsize=9)
    for axis in (ax, curve):
        axis.yaxis.grid(True, alpha=.16)
        axis.set_axisbelow(True)
    finish(fig, 'quality-zh')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, ratio, refs, color in [(axes[0], '060', [512, 1024, 1630], BLUE),
                                  (axes[1], '050', [512, 1024, 2048], GREEN)]:
        values = [100*data[f'recovery_{ratio}_ref{r}_top24']['accuracy'] for r in refs]
        ax.plot(refs, values, marker='o', color=color, label='24 recovery windows')
        for x, y in zip(refs, values):
            keep = 100*data[f'recovery_{ratio}_ref{x}_top24']['mean_document_retention']
            ax.annotate(f'{y:.2f}%\nkeep {keep:.2f}%', (x, y), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9)
        ax.set(xticks=refs, xlim=(min(refs)-150, max(refs)+180), ylim=(49, 56.2),
               xlabel='Reference suffix length (tokens)', ylabel='Answer accuracy (%)',
               title=f'Base retention: {int(ratio)}% + recovery')
        ax.grid(True, alpha=.16)
        ax.axhline(100*data['baseline']['accuracy'], color=GREY, ls='--', label='Full cache')
    point = 100*data['recovery_060_ref1024_top16']['accuracy']
    axes[0].scatter([1024], [point], color=ORANGE, marker='s', zorder=5, label='16 recovery windows')
    axes[0].annotate(f'{point:.2f}%\nkeep 62.60%', (1024, point), xytext=(30, -28),
                     textcoords='offset points', fontsize=9, arrowprops={'arrowstyle': '-', 'color': ORANGE})
    axes[0].legend(loc='upper left', frameon=False, fontsize=8)
    finish(fig, 'budget-zh')

    fig, (ax, tasks) = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for offset, cell, color, label in [(-.18, 'baseline', GREY, 'Full cache'),
                                     (.18, 'a60_r3_additive', GREEN, 'Proposed + recovery')]:
        values = [100*report['glm53'][f'longbench{cap}']['cells'][cell]['accuracy'] for cap in [192, 512]]
        bars = ax.bar([i+offset for i in range(2)], values, width=.34, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+.8, f'{value:.2f}', ha='center', fontsize=9)
    ax.set(xticks=[0, 1], xticklabels=['192-token cap', '512-token cap'], ylim=(0, 66),
           ylabel='Answer accuracy (%)', title='GLM-5.3-Flash: LongBench-v2 (473)')
    ax.legend(frameon=False, loc='upper left', fontsize=8)
    order = ['longbook_qa_chn', 'longbook_qa_eng', 'longbook_choice_eng', 'longbook_sum_eng', 'code_debug']
    for offset, cell, color in [(-.18, 'baseline', GREY), (.18, 'a60_r3_additive', GREEN)]:
        scores = [100*report['glm53']['infinitebench512']['cells'][cell]['tasks'][t]['mean_score'] for t in order]
        bars = tasks.bar([i+offset for i in range(5)], scores, width=.34, color=color)
        for bar, score in zip(bars, scores):
            tasks.text(bar.get_x()+bar.get_width()/2, score+1, f'{score:.1f}', ha='center', fontsize=7)
    tasks.set(xticks=range(5), xticklabels=['Chinese QA\nChar F1', 'English QA\nWord F1', 'Book MC\nAccuracy',
                                         'Summary\nROUGE-Lsum', 'Code debug\nAccuracy'],
              ylim=(0, 103), ylabel='Task-native score (%)', title='InfiniteBench (701), 512-token cap')
    tasks.tick_params(axis='x', labelsize=8)
    for axis in (ax, tasks):
        axis.yaxis.grid(True, alpha=.16)
        axis.set_axisbelow(True)
    finish(fig, 'glm-quality-zh')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    args = parser.parse_args()
    evidence = collect(args.results_root.resolve())
    figures(evidence)
    print('Audited V3.2 and GLM paired records, scores, states, and source checksums; rendered 4 figures.')
