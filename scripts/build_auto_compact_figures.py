"""Render blog figures from the checked, aggregate-only evidence snapshot.

Run with a Python environment containing matplotlib. No model is loaded.
"""
from pathlib import Path
import json
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/figures/dsa-auto-compact'
DATA = json.loads((OUT / 'evidence.json').read_text())['cells']
BLUE, GREEN, ORANGE, INK = '#5363ee', '#2f9d78', '#d86f4d', '#191a20'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': INK, 'text.color': INK, 'axes.edgecolor': '#bfc2ce',
    'svg.fonttype': 'path', 'svg.hashsalt': 'dsa-auto-compact',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def finish(fig, name):
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', metadata={'Date': None})
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', dpi=170)
    plt.close(fig)


fig, (ax, curve) = plt.subplots(1, 2, figsize=(11, 4.1), layout='constrained')
names = ['baseline', 'generic_a_060', 'generic_e1_060', 'recovery_060_ref1630_top24']
labels = ['Full\ncache', 'A60\ngeneric', 'E1-60\ngeneric', 'A60 +\nrecovery']
bars = ax.bar(range(4), [100 * DATA[n]['accuracy'] for n in names],
              color=['#9397a8', BLUE, ORANGE, GREEN], width=.63)
for bar, name in zip(bars, names):
    c = DATA[name]
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{100*c['accuracy']:.2f}%\n{c['correct']}/473", ha='center', va='bottom', fontsize=9)
ax.set(xticks=range(4), xticklabels=labels, ylim=(0, 67), ylabel='Answer accuracy (%)',
       title='A  Query-independent compression and recovery')
ax.yaxis.grid(True, alpha=.16)
ax.set_axisbelow(True)
ratios = [20, 40, 60, 80]
for method, color, marker in [('a', BLUE, 'o'), ('e1', ORANGE, 's')]:
    values = [100*DATA[f'task_{method}_{r:03d}']['accuracy'] for r in ratios]
    curve.plot(ratios, values, marker=marker, color=color, label=method.upper(), linewidth=1.8)
    for x, y in zip(ratios, values):
        curve.annotate(f'{y:.2f}', (x, y), xytext=(0, 8 if method=='a' else -15),
                       textcoords='offset points', ha='center', fontsize=8)
curve.axhline(100*DATA['baseline']['accuracy'], color='#9397a8', linestyle='--', label='Full cache')
curve.set(xlim=(14,86), ylim=(48.5,56), xticks=ratios,
          xlabel='Target document-row retention (%)', ylabel='Answer accuracy (%)',
          title='B  Task-aware reference (query already known)')
curve.grid(True, alpha=.16)
curve.legend(frameon=False, loc='upper left', ncols=3, fontsize=8)
finish(fig, 'quality')

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
for ax, ratio, refs, color in [(axes[0], '060', [512,1024,1630], BLUE),
                              (axes[1], '050', [512,1024,2048], GREEN)]:
    values=[100*DATA[f'recovery_{ratio}_ref{r}_top24']['accuracy'] for r in refs]
    ax.plot(refs,values,marker='o',color=color,lw=1.8,label='24 recovery windows')
    for x,y in zip(refs,values):
        keep=100*DATA[f'recovery_{ratio}_ref{x}_top24']['mean_document_retention']
        ax.annotate(f'{y:.2f}%\nkeep {keep:.2f}%',(x,y),xytext=(0,10),
                    textcoords='offset points',ha='center',fontsize=9)
    ax.set(xticks=refs, xlim=(min(refs)-150,max(refs)+180), ylim=(49,56.2),
           xlabel='Reference suffix length (tokens)', ylabel='Answer accuracy (%)',
           title=f"{'A' if ratio=='060' else 'B'}  Base A{int(ratio)} + additive recovery")
    ax.grid(True,alpha=.16)
    ax.axhline(100*DATA['baseline']['accuracy'],color='#9397a8',ls='--',lw=1,label='Full cache')
point=100*DATA['recovery_060_ref1024_top16']['accuracy']
axes[0].scatter([1024],[point],color=ORANGE,marker='s',zorder=5,label='16 recovery windows')
axes[0].annotate(f'{point:.2f}%\nkeep 62.60%',(1024,point),xytext=(30,-28),
                 textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','color':ORANGE})
axes[0].legend(loc='upper left',frameon=False,fontsize=8)
finish(fig,'budget')


def pipeline(language):
    if language=='zh':
        title='一次长前缀前向，分阶段确定最终记忆'
        boxes=[('01  通用 Reference', ['完整缓存上评分', '冻结每层的 S 与 s']),
               ('02  受限 Query Probe', ['真实 query 只读取 S', '选出 24 个 128-token 窗口']),
               ('03  最终物理压实', ['从完整缓存补入缺失行', '保留 S，得到 S ∪ R'])]
        bottom='完整缓存保留到最终复制；临时 Reference / Probe 状态在正式回答前回退。'
        tail='在最终记忆上重放短 query，继续生成'
    else:
        title='One long-prefix prefill, followed by staged memory selection'
        boxes=[('01  Generic reference', ['Score the full cache', 'Freeze per-layer S and s']),
               ('02  Restricted query probe', ['The query reads only S', 'Select 24 windows of 128 tokens']),
               ('03  Final physical rewrite', ['Restore missing rows from full cache', 'Preserve S to form S ∪ R'])]
        bottom='Keep the full cache until the final copy; rewind temporary reference and probe states.'
        tail='Replay the short query on final memory, then continue decoding'
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="330" viewBox="0 0 1000 330" role="img">',
         f'<title>{html.escape(title)}</title>',
         '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="#9397a8"/></marker></defs>',
         '<rect width="1000" height="330" rx="12" fill="#fcfcfb"/>',
         '<g font-family="Inter, Noto Sans CJK SC, Microsoft YaHei, sans-serif" fill="#191a20">',
         f'<text x="30" y="38" font-size="22" font-weight="600">{html.escape(title)}</text>']
    for i,(heading,lines) in enumerate(boxes):
        x=30+i*325
        out.extend([f'<rect x="{x}" y="66" width="290" height="134" rx="9" fill="white" stroke="#dcdcd5"/>',
                    f'<text x="{x+17}" y="98" fill="{BLUE}" font-size="18" font-weight="600">{html.escape(heading)}</text>'])
        for j,line in enumerate(lines):
            out.append(f'<text x="{x+17}" y="{133+j*29}" font-size="16">{html.escape(line)}</text>')
        if i<2:
            out.append(f'<path d="M{x+295} 132H{x+320}" fill="none" stroke="#9397a8" stroke-width="2" marker-end="url(#arrow)"/>')
    out.extend([f'<text x="500" y="242" text-anchor="middle" font-size="17" fill="#70737d">{html.escape(bottom)}</text>',
                f'<text x="500" y="289" text-anchor="middle" font-size="19" fill="{GREEN}" font-weight="600">{html.escape(tail)}</text>',
                '</g></svg>'])
    (OUT/f'pipeline-{language}.svg').write_text('\n'.join(out)+'\n')

pipeline('zh')
pipeline('en')
print('Rendered quality, budget, and bilingual pipeline figures.')
