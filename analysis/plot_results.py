#!/usr/bin/env python3
"""Plot only saved aggregate TruthfulQA results; no models or data downloads.

From the checkout: python -m pip install -e '.[figures]'
                  python analysis/plot_results.py
The three generated files are deterministic for the pinned plotting environment.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('Base', 'SFT', 'F2-Lite', 'DPO')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/figures')
    args = parser.parse_args()
    source = args.root / 'results/matched_results.json'
    score_csv = args.root / 'results/scores.csv'
    result = json.loads(source.read_text())
    with score_csv.open(newline='') as stream:
        rows = {r['model']: r for r in csv.DictReader(stream)}
    n = result['integrity']['questions']
    if set(rows) != set(ARMS) or n != 817:
        raise ValueError('Expected exactly four recipes and 817 matched questions')
    for arm in ARMS:
        if int(rows[arm]['n']) != n:
            raise ValueError('Mismatched question count')
        for metric in ('mc1', 'mc2'):
            saved = result['arms'][arm][metric]
            if float(rows[arm][metric + '_mean']) != saved['mean']:
                raise ValueError('CSV/JSON means disagree')
            if json.loads(rows[arm][metric + '_bootstrap95']) != saved['bootstrap95']:
                raise ValueError('CSV/JSON intervals disagree')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.edgecolor': '#cbd5e1',
        'text.color': '#182c43', 'axes.labelcolor': '#475569',
        'xtick.color': '#64748b', 'ytick.color': '#182c43',
        'svg.fonttype': 'none', 'svg.hashsalt': 'truthfulness-lora-figure-v1',
    })
    fig, ax = plt.subplots(figsize=(11.6, 4.9))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.075, right=.98, top=.79, bottom=.24)
    fig.text(.075, .95, 'Matched four-model comparison', size=17, weight='bold', va='top')
    fig.text(.075, .865, 'TruthfulQA  |  817 questions  |  native-chat likelihood scoring', size=10, color='#64748b')
    plot_data = {}
    x = np.arange(len(ARMS))
    width = .31
    for metric, offset, color, label in zip(
        ('mc1', 'mc2'), (-width/2, width/2), ('#78a8e8', '#8acb9c'), ('MC1', 'MC2'),
    ):
        means = np.array([result['arms'][a][metric]['mean'] for a in ARMS]) * 100
        intervals = np.array([result['arms'][a][metric]['bootstrap95'] for a in ARMS]) * 100
        if not np.all((0 <= intervals[:, 0]) & (intervals[:, 0] <= means) &
                      (means <= intervals[:, 1]) & (intervals[:, 1] <= 47)):
            raise ValueError('Invalid metric/interval or insufficient fixed plot headroom')
        ax.bar(x+offset, means, width=width, color=color, label=label, zorder=2)
        ax.errorbar(x+offset, means, yerr=np.vstack((means-intervals[:, 0], intervals[:, 1]-means)),
                    fmt='none', ecolor='#3a4b5f', capsize=3, elinewidth=1, capthick=1, zorder=3)
        for i, value in enumerate(means):
            ax.text(x[i]+offset, intervals[i, 1]+1.1, f'{value:.2f}',
                    ha='center', va='bottom', size=9, weight='bold')
        plot_data[metric] = {
            arm: {'mean_percent': float(means[i]), 'nominal_bootstrap95_percent': intervals[i].tolist()}
            for i, arm in enumerate(ARMS)
        }
    ax.set(xticks=x, xticklabels=ARMS, ylim=(0, 50), yticks=np.arange(0, 51, 10))
    ax.set_yticklabels([f'{v}%' for v in range(0, 51, 10)])
    ax.tick_params(axis='both', length=0, pad=9)
    ax.set_ylabel('Metric score (%)', size=10, labelpad=10)
    ax.grid(axis='y', color='#e5eaf0', linewidth=.7, linestyle=(0, (4, 4)), zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2, frameon=False,
              handlelength=1.2, columnspacing=2)
    fig.text(.075, .035, 'Whiskers: nominal 95% item-bootstrap intervals. One training seed per adapted recipe.',
             size=8.5, color='#64748b')
    args.output.mkdir(parents=True, exist_ok=True)
    png, svg = (args.output / ('truthfulqa_comparison.' + x) for x in ('png', 'svg'))
    fig.savefig(png, dpi=200, facecolor='white', metadata={'Software': 'Matplotlib ' + matplotlib.__version__})
    fig.savefig(svg, facecolor='white', metadata={'Date': None, 'Creator': 'analysis/plot_results.py'})
    plt.close(fig)
    provenance = {
        'source_files_sha256': {str(p.relative_to(args.root)): digest(p) for p in (source, score_csv)},
        'generated_files_sha256': {p.name: digest(p) for p in (png, svg)},
        'generator': 'analysis/plot_results.py', 'matplotlib': matplotlib.__version__,
        'question_count': n, 'y_axis_percent_limits': [0, 50], 'recipes_in_order': list(ARMS),
        'values': plot_data,
        'caption': 'Saved historical means and nominal 95% item-bootstrap intervals; no cross-seed uncertainty or multiplicity adjustment. MC1 is ranking credit; MC2 is normalized true-choice likelihood mass. Neither is free-generation factual accuracy.',
        'raw_item_data_used': False,
    }
    (args.output / 'truthfulqa_comparison.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
    print('Wrote aggregate comparison PNG, SVG and provenance JSON to', args.output)


if __name__ == '__main__':
    main()
