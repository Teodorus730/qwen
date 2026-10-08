"""Reproduce three historical scientific figures from recorded result JSONs.

Only CPU plotting; no models are loaded or measurements rerun.
Usage: bundled python create_historic.py [--workspace PATH] [--output PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=Path, default=HERE / 'data')
parser.add_argument('--output', type=Path, default=HERE)
args = parser.parse_args()
ROOT, OUT = args.workspace.resolve(), args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT / '.mpl_config')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 14, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 150,
})
BLUE, TEAL, GRAY, ORANGE = '#2765A6', '#247F7F', '#8A929B', '#C47128'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(rel):
    p = ROOT / rel
    return json.loads(p.read_text(encoding='utf-8-sig')), {'path': rel, 'sha256': sha(p)}

manifest = {'status': 'complete', 'measurement_rerun': False, 'figures': []}

def save(fig, stem, numbers, sources, anchor, caption):
    image = OUT / (stem + '.png')
    fig.savefig(image, dpi=150, facecolor='white')
    plt.close(fig)
    payload = {
        'figure': image.name, 'anchor': anchor, 'caption_ru': caption,
        'sources': sources, 'plotted_data': numbers,
        'runtime': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
    }
    data = OUT / (stem + '.json')
    data.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest['figures'].append({
        'file': image.name, 'sha256': sha(image), 'numbers': data.name,
        'numbers_sha256': sha(data), 'anchor': anchor, 'sources': sources,
    })

summary, historic_src = read('october_benchmark/results/analysis/summary.json')

# 1. Exactly 11 ordinary text predictors; token/PLS/acquisition excluded.
methods = [
    'global_mean', 'subject_mean', 'length_only', 'ridge', 'bayesian_ridge',
    'knn', 'krr', 'extra_trees', 'gp_rbf', 'gp_matern32', 'gp_matern52',
]
labels = [
    'Общее среднее', 'Среднее домена', 'Только длина', 'Ridge', 'Bayesian Ridge',
    'kNN', 'KRR', 'Extra Trees', 'GP RBF', 'GP Matérn 3/2', 'GP Matérn 5/2',
]
datasets = ['controlled', 'natural_fold0', 'natural_fold1', 'natural_fold2']
rows = [r for r in summary['summary']
        if r['folder'] == 'predictors' and r['part'] == 'test' and r['budget'] == 192]
assert len(rows) == 44
assert {r['kind'] for r in rows} == set(methods)
assert all(r['runs'] == 5 for r in rows)
index = {(r['dataset'], r['kind']): r for r in rows}
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
plotted = []
for ax, dataset in zip(axes.flat, datasets):
    items = [index[dataset, m] for m in methods]
    means = [r['mae'] for r in items]
    sd = [r['mae_sd'] for r in items]
    y = np.arange(len(methods))
    ax.barh(y, means, xerr=sd, height=.67,
            color=[BLUE if m.startswith('gp_') else TEAL if m == 'krr' else GRAY for m in methods],
            error_kw={'ecolor': '#30343B', 'elinewidth': 1, 'capsize': 2}, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, .0155)
    ax.set_xticks([0, .005, .010, .015], ['0', '0,005', '0,010', '0,015'])
    ax.grid(axis='x', color='#DCE1E6', linewidth=.7, zorder=0)
    ax.set_title(dataset)
    ax.set_xlabel('Test MAE JSD: меньше — лучше')
    plotted.extend({'dataset': dataset, 'method': m, 'mae': r['mae'],
                    'mae_sd': r['mae_sd'], 'runs': r['runs']}
                   for m, r in zip(methods, items))
fig.suptitle('Одиннадцать предикторов при одинаковых 192 измеренных текстах', fontsize=18, y=.975)
fig.subplots_adjust(left=.13, right=.98, top=.91, bottom=.10, hspace=.33, wspace=.43)
fig.text(.5, .035, 'Среднее ± SD по 5 seeds каждого пула. GP выделены синим, KRR — зелёным.\n'
         'Все методы получают одинаковые метки; лучшего метода для всех четырёх пулов нет.',
         ha='center', va='center', fontsize=11)
save(fig, 'predictors_mae', plotted, [historic_src], 'section-21',
     'Test MAE одиннадцати предикторов при 192 метках; среднее ± SD пяти seeds. '
     'График не включает token- и PLS-предикторы.')

# 2. Controlled acquisition uses 96, not 192, measurements.
policies = ['random', 'stratified_random', 'sigma', 'ucb5']
policy_labels = ['Random', 'Random\nпо доменам', 'Максимум σ', 'UCB: μ + 5σ']
controlled = {r['policy']: r for r in summary['acquisition_summary'] if r['dataset'] == 'controlled'}
assert set(controlled) == set(policies)
assert all(r['runs'] == 5 for r in controlled.values())
fig, axes = plt.subplots(1, 2, figsize=(14, 6.7))
x = np.arange(4)
colors = [GRAY, '#BBC1C8', TEAL, BLUE]
chart_specs = [
    ('recall10', 'recall10_sd', 100, 'Найдено настоящих top-10 train, %', 'Поиск больших JSD', (0, 75)),
    ('theta_macro_error_pp', 'theta_macro_error_pp_sd', 1, 'Средняя ошибка карты θ, п.п.', 'Оценка карты', (0, 19)),
]
for ax, (metric, sd_key, scale, ylabel, title, limits) in zip(axes, chart_specs):
    values = np.array([controlled[p][metric] * scale for p in policies])
    errors = [controlled[p][sd_key] * scale for p in policies]
    ax.bar(x, values, yerr=errors, width=.65, color=colors, zorder=3,
           error_kw={'ecolor': '#30343B', 'elinewidth': 1.3, 'capsize': 4})
    ax.set_xticks(x, policy_labels)
    ax.set_ylim(*limits)
    ax.set_ylabel(ylabel)
    ax.set_title(title + (' — больше лучше' if metric == 'recall10' else ' — меньше лучше'))
    ax.grid(axis='y', color='#DCE1E6', linewidth=.7, zorder=0)
    for i, value in enumerate(values):
        ax.text(i, .6 if metric == 'recall10' else .15,
                (f'{value:.0f}%' if metric == 'recall10' else f'{value:.2f}').replace('.', ','),
                ha='center', va='bottom', color='white' if i != 1 else '#30343B', fontweight='bold')
fig.suptitle('Controlled: правило выбора измерений меняет поиск и карту', fontsize=18, y=.96)
fig.subplots_adjust(left=.075, right=.98, top=.85, bottom=.23, wspace=.23)
fig.text(.5, .105, 'Бюджет 96: общий случайный старт 24 + шесть пакетов по 12. Среднее ± SD по 5 seeds.\n'
         'σ ищет неопределённые прогнозы; UCB ищет высокие JSD. Условия отличаются от fresh-проверки 8 октября.',
         ha='center', va='center', fontsize=11)
save(fig, 'acquisition_controlled',
     {'dataset': 'controlled', 'budget_documents': 96, 'start_documents': 24,
      'batch_documents': 12, 'policies': [controlled[p] for p in policies],
      'recall_definition': 'Measured true top10 of train among 96 selected, including initial24',
      'theta_definition': 'Test macro domain theta error via GP CDF'},
     [historic_src], 'section-22',
     'Controlled acquisition: при 96 метках разные правила выбора дают разные результаты '
     'по поиску train top-10 и по test-карте θ. Среднее ± SD пяти seeds.')

# 3. Two predeclared fresh confirmation pools; keep B's positive primary ranking result.
a, src_a = read('confirmation_aligned_20261007/engine/results/data_sealed/analysis/summary.json')
b, src_b = read('gp_task_confirmation_20261007/engine/results/data_sealed/task_analysis/summary.json')
assert a['pool_size'] == b['pool_size'] == 600
assert a['n_seeds'] == b['seeds'] == 20
assert a['budget_each_arm'] == b['budget_each'] == 192
diff_a = a['primary_improvement_srs_minus_gp']
diff_b = b['theta_improvement_srs_minus_gp']
rank_b = b['ranking_improvement_gp_minus_srs']
numbers = {
    'A': {'primary': 'theta_macro_mae', 'gp_mae_pp': a['means']['gp']['mae_pp'],
          'srs_mae_pp': a['means']['direct_srs']['mae_pp'],
          'srs_minus_gp_mae_pp': diff_a['mean_pp'],
          'conditional_95_pp': diff_a['conditional_percentile_95']},
    'B': {'primary': 'domain_spearman', 'gp_mae_pp': b['gp_theta_mae_pp'],
          'srs_mae_pp': b['srs_theta_mae_pp'],
          'srs_minus_gp_mae_pp': diff_b['mean'], 'conditional_95_pp': diff_b['conditional_95'],
          'gp_spearman': b['gp_domain_spearman'], 'srs_spearman': b['srs_domain_spearman'],
          'gp_minus_srs_spearman': rank_b['mean'], 'ranking_conditional_95': rank_b['conditional_95']},
    'budget_documents_each': 192, 'pool_documents_each': 600, 'selection_plans_each_pool': 20,
}
fig, axes = plt.subplots(1, 3, figsize=(18, 7.0), gridspec_kw={'width_ratios': [1, 1.05, 1.1]})
left, middle, right = axes
x = np.arange(2)
width = .33
gp = [numbers[p]['gp_mae_pp'] for p in ['A', 'B']]
srs = [numbers[p]['srs_mae_pp'] for p in ['A', 'B']]
left.bar(x - width/2, gp, width, color=BLUE, label='GP', zorder=3)
left.bar(x + width/2, srs, width, color=GRAY, label='SRS', zorder=3)
left.set_xticks(x, ['Пул A\nцель: ошибка θ', 'Пул B\nцель: порядок доменов'])
left.set_ylim(0, 6)
left.set_title('Ошибка карты θ')
left.set_ylabel('Средняя абсолютная ошибка, п.п.')
left.legend(frameon=False, loc='upper right')
left.grid(axis='y', color='#DCE1E6', linewidth=.7, zorder=0)
for i, (g, s) in enumerate(zip(gp, srs)):
    left.text(i-width/2, g+.10, f'{g:.3f}'.replace('.', ','), ha='center', fontsize=10)
    left.text(i+width/2, s+.10, f'{s:.3f}'.replace('.', ','), ha='center', fontsize=10)

for yy, p in [(1, 'A'), (0, 'B')]:
    n = numbers[p]
    value, ci = n['srs_minus_gp_mae_pp'], n['conditional_95_pp']
    middle.errorbar(value, yy, xerr=[[value-ci[0]], [ci[1]-value]], fmt='o',
                    markersize=8, color=BLUE, ecolor=BLUE, capsize=5, elinewidth=2, zorder=3)
    middle.text(value, yy+.25,
                f'{value:+.3f} [{ci[0]:+.3f}; {ci[1]:+.3f}]'.replace('.', ','),
                ha='center', fontsize=10)
middle.axvline(0, color='#505A64', linestyle='--', linewidth=1)
middle.set_yticks([1, 0], ['Пул A', 'Пул B'])
middle.set_ylim(-.55, 1.55)
middle.set_xlim(-1.25, 1.25)
middle.set_title('Парная разница ошибки θ')
middle.set_xlabel('SRS − GP, п.п.\nположительная разница — в пользу GP')
middle.grid(axis='x', color='#DCE1E6', linewidth=.7, zorder=0)

rv, rci = rank_b['mean'], rank_b['conditional_95']
right.errorbar(rv, 0, xerr=[[rv-rci[0]], [rci[1]-rv]], fmt='o', color=TEAL,
               markersize=9, capsize=5, elinewidth=2)
right.axvline(0, color='#505A64', linestyle='--', linewidth=1)
right.set_ylim(-.55, .65)
right.set_xlim(-.015, .065)
right.set_yticks([0], ['Пул B'])
right.set_title('Основная цель пула B:\nпорядок доменов')
right.set_xlabel('Spearman GP − Spearman SRS\nположительная разница — в пользу GP')
right.grid(axis='x', color='#DCE1E6', linewidth=.7, zorder=0)
right.text(rv, .27, f'+{rv:.5f}\n[{rci[0]:.5f}; {rci[1]:.5f}]'.replace('.', ','),
           ha='center', color=TEAL, fontsize=12)
right.text(.025, -.40, 'GP: 0,97143; SRS: 0,94337', ha='center', fontsize=10)
fig.suptitle('GP и SRS: две подтверждающие проверки на новых пулах 7 октября', fontsize=18, y=.96)
fig.subplots_adjust(left=.06, right=.98, top=.79, bottom=.29, wspace=.37)
fig.text(.5, .085, 'По 600 текстов и 20 пар планов каждого пула; равный бюджет 192. Интервалы — условный 95% bootstrap.\n'
         'По ошибке θ оба интервала пересекают ноль. В пуле B основной тест порядка доменов поддерживает GP.\n'
         'Планы одного пула не являются независимыми новыми корпусами; проверяются разные основные цели.',
         ha='center', va='center', fontsize=11)
save(fig, 'map_confirmations', numbers, [src_a, src_b], 'section-56',
     'Два подтверждающих новых пула 7 октября: средняя θ-MAE и условные 95% интервалы парной разности '
     'SRS−GP. Справа отдельно показан положительный основной результат ранжирования доменов пула B; '
     'он не заменяет проверки θ-MAE.')

manifest['script'] = {'file': Path(__file__).name, 'sha256': sha(Path(__file__))}
(OUT / 'historic_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'complete', 'output': str(OUT),
                  'figures': [r['file'] for r in manifest['figures']]}, ensure_ascii=False))
