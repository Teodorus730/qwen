"""Figures from completed, audited experiments; no model fits or new data."""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE / 'data'
os.environ['MPLCONFIGDIR'] = str(HERE / '.mpl_config')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = HERE
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 150, 'savefig.dpi': 150})
COLORS = dict(gp='#2563eb', krr='#15803d', bayesian_ridge='#7c3aed',
              cluster_mean='#c2410c', random='#64748b', domain_mean='#a16207')
NAMES = dict(gp='GP', krr='KRR', bayesian_ridge='Bayesian Ridge',
             cluster_mean='Кластерные средние', random='Random', domain_mean='Среднее домена')
SOURCE_PATHS = [
    'hybrid_srs_search_20261008/results/task_B/summary.json',
    'hybrid_srs_search_20261008/results/confirmation_A/summary.json',
    'hybrid_fresh_confirmation_20261008/received_run/results/summary.json']
SOURCES = {name: json.loads((WORKSPACE/name).read_text(encoding='utf-8')) for name in SOURCE_PATHS}
FRESH = SOURCES[SOURCE_PATHS[2]]
PLOTTED = {}


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def search_three_pools():
    methods = ['gp', 'krr', 'bayesian_ridge', 'cluster_mean', 'random']
    pools = [SOURCES[SOURCE_PATHS[1]], SOURCES[SOURCE_PATHS[0]], FRESH]
    values = np.array([[100*p['methods'][m]['next_top10_recall'] for p in pools] for m in methods])
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    positions = np.arange(3)
    width = .145
    for index, method in enumerate(methods):
        bars = ax.bar(positions + (index-2)*width, values[index], width,
                      label=NAMES[method], color=COLORS[method])
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in values[index]], padding=3, fontsize=9)
    ax.axhline(100*10/68, color='#a16207', ls='--', lw=1.4,
               label='Точное ожидание random: 14,71%')
    ax.axvline(1.5, color='#94a3b8', ls=':', lw=1)
    ax.set_xticks(positions, ['Пул A: открытые данные', 'Пул B: открытые данные', 'Пул C: новое подтверждение'])
    ax.set_ylabel('Найденная доля настоящих top-10 остатка, %')
    ax.set_ylim(0, 41)
    ax.grid(axis='y', alpha=.18)
    fig.suptitle('Одинаковые 192 стартовые метки + 60 следующих текстов', fontsize=14, y=.99)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc='upper center',
               bbox_to_anchor=(.5, .93), ncol=3, fontsize=10)
    fig.text(.5, .015, 'По 20 планов на каждом пуле. A/B — development; только C использован как новое подтверждение.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .82))
    name = 'search_three_pools.png'
    save(fig, name)
    PLOTTED[name] = dict(methods=methods, pools=['A', 'B', 'C'], recall_percent=values.tolist(),
                        exact_random_percent=100*10/68, sources=SOURCE_PATHS)


def fresh_search_and_gate():
    methods = ['krr', 'gp', 'bayesian_ridge', 'cluster_mean', 'domain_mean', 'random']
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), gridspec_kw={'width_ratios': [1.2, 1]})
    means, intervals = [], []
    for number, method in enumerate(methods):
        result = FRESH['methods'][method]
        mean = 100*result['next_top10_recall']
        low, high = np.array(result['conditional_intervals']['next_top10_recall']['conditional_95'])*100
        axes[0].plot([low, high], [number, number], color=COLORS[method], lw=2.5)
        axes[0].plot(mean, number, 'o', color=COLORS[method], ms=7)
        axes[0].text(high+.65, number, f'{mean:.2f}%', va='center', fontsize=10)
        means.append(mean)
        intervals.append([float(low), float(high)])
    axes[0].set_yticks(range(6), [NAMES[m] for m in methods])
    axes[0].invert_yaxis()
    axes[0].axvline(100*10/68, color='#a16207', ls='--', lw=1.3)
    axes[0].set_xlim(7, 42)
    axes[0].set_xlabel('Recall top-10 остатка, %')
    axes[0].set_title('Качество поиска')
    axes[0].grid(axis='x', alpha=.18)
    gate = FRESH['secondary_noninferiority']
    mean = 100*gate['mean']
    low, high = np.array(gate['conditional_95'])*100
    margin = -100*FRESH['declared_noninferiority_margin']
    axes[1].axvspan(margin, 3, color='#dcfce7', alpha=.65)
    axes[1].axvline(margin, color='#a16207', ls='--', lw=1.5)
    axes[1].axvline(0, color='#94a3b8', lw=1)
    axes[1].plot([low, high], [0, 0], color=COLORS['krr'], lw=3)
    axes[1].plot(mean, 0, 'o', color=COLORS['krr'], ms=8)
    axes[1].text(mean, .15, f'{mean:+.2f} п.п. [{low:+.2f}; {high:+.2f}]', ha='center', fontsize=11)
    axes[1].text(margin, -.35, 'Допуск: −1,67 п.п.', ha='center', color='#92400e', fontsize=10)
    axes[1].text(.2, -.53, 'Нижняя граница −1,00 выше порога', ha='center', fontsize=10)
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-.7, .65)
    axes[1].set_yticks([])
    axes[1].set_xlabel('Парная разница recall: KRR − GP, п.п.')
    axes[1].set_title('Заранее заданная проверка замены')
    axes[1].grid(axis='x', alpha=.18)
    fig.suptitle('Новый пул: KRR заменяет GP в пределах допуска среднего качества', fontsize=14, y=.99)
    fig.text(.5, .015, '95% bootstrap условен на один корпус и 20 планов. Допуск не означает равенство методов или гарантию каждого запуска.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .93))
    name = 'fresh_search_and_gate.png'
    save(fig, name)
    PLOTTED[name] = dict(methods=methods, recall_percent=means, conditional95_percent=intervals,
                        krr_minus_gp_percent=dict(mean=mean, low=float(low), high=float(high), margin=margin),
                        source=SOURCE_PATHS[2])


def fresh_domains():
    names = {'computer_security': 'Безопасность\nкомпьютеров',
             'conceptual_physics': 'Концептуальная\nфизика',
             'econometrics': 'Эконометрика', 'formal_logic': 'Формальная\nлогика',
             'machine_learning': 'Машинное\nобучение', 'public_relations': 'Связи с\nобщественностью'}
    domains = FRESH['domains']
    methods = ['gp', 'krr', 'random']
    values = np.array([[100*FRESH['methods'][m]['by_domain'][d]['remaining_top10_recall']
                        for d in domains] for m in methods])
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    positions = np.arange(6)
    for number, method in enumerate(methods):
        bars = ax.bar(positions+(number-1)*.23, values[number], .23,
                      label=NAMES[method], color=COLORS[method])
        ax.bar_label(bars, labels=[f'{v:.1f}' for v in values[number]], padding=3, fontsize=9)
    ax.axhline(100*10/68, color='#a16207', ls='--', lw=1.3, label='Ожидание random: 14,71%')
    ax.set_xticks(positions, [names[d] for d in domains], fontsize=10)
    ax.set_ylabel('Recall top-10 остатка, %')
    ax.set_ylim(0, 55)
    ax.grid(axis='y', alpha=.18)
    fig.suptitle('По три категории с более высоким средним recall у GP и KRR', fontsize=14, y=.99)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc='upper center',
               bbox_to_anchor=(.5, .93), ncol=4, fontsize=10)
    fig.text(.5, .015, 'Средние 20 планов. Доменные сравнения описательные: это не правило назначения GP/KRR новым категориям.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .83))
    name = 'fresh_domain_search.png'
    save(fig, name)
    PLOTTED[name] = dict(domains=domains, methods=methods, recall_percent=values.tolist(), source=SOURCE_PATHS[2])


def fresh_map_tradeoff():
    parts = ['initial192', 'random252']
    values = [FRESH['theta'][part]['error_pp']['mean'] for part in parts]
    intervals = [FRESH['theta'][part]['error_pp']['conditional_95'] for part in parts]
    errors = np.array([[v-lo, hi-v] for v, (lo, hi) in zip(values, intervals)]).T
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar([0, 1], values, width=.47, color=['#15803d', '#64748b'],
                  yerr=errors, capsize=6)
    ax.bar_label(bars, labels=[f'{v:.2f} п.п.' for v in values], padding=36, fontsize=12)
    ax.set_xticks([0, 1], ['Карта по 192 случайным\n+ 60 адресных для поиска',
                          'Карта по всем\n252 случайным'])
    ax.set_ylabel('Средняя абсолютная ошибка θ, п.п. — меньше лучше')
    ax.set_ylim(0, 8)
    ax.grid(axis='y', alpha=.18)
    ax.set_title('Одинаковый общий бюджет: адресный поиск уменьшает число меток для карты', fontsize=12)
    fig.text(.5, .015, 'θ — доля текстов с JSD ≤ 0,05. Адресные 60 не входят в простую случайную долю. Интервалы — условный95% bootstrap.',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .085, 1, .97))
    name = 'fresh_map_tradeoff.png'
    save(fig, name)
    PLOTTED[name] = dict(parts=parts, theta_mae_pp=values, conditional95_pp=intervals, source=SOURCE_PATHS[2])


def fresh_costs():
    methods = ['gp', 'krr', 'random']
    cpu = [FRESH['methods'][m]['predictor_cpu_recorded_seconds'] for m in methods]
    qwen = np.array([FRESH['methods'][m]['qwen_virtual252_recorded_seconds'] for m in methods])
    bge = np.array([FRESH['methods'][m]['bge_fixed600_seconds'] for m in methods])
    # Unknown random selection time is excluded, not claimed as measured zero.
    cpu_recorded = np.array([cpu[0], cpu[1], 0.])
    sums = qwen+bge+cpu_recorded
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={'width_ratios': [.8, 1.25]})
    bars = axes[0].bar([0, 1], cpu[:2], width=.5, color=[COLORS['gp'], COLORS['krr']])
    axes[0].bar_label(bars, labels=[f'{v:.3f} с' for v in cpu[:2]], padding=4)
    axes[0].set_xticks([0, 1], ['GP', 'KRR'])
    axes[0].set_ylim(0, .48)
    axes[0].set_ylabel('Измеренное время CPU, с')
    axes[0].set_title('Fit + прогноз + выбор')
    axes[0].text(.5, .44, f'KRR: −{100*(1-cpu[1]/cpu[0]):.1f}% времени CPU', ha='center', fontsize=10)
    axes[0].grid(axis='y', alpha=.18)
    positions = np.arange(3)
    axes[1].bar(positions, qwen, color='#475569', width=.53, label='Qwen252: суммы времени документов')
    axes[1].bar(positions, bge, bottom=qwen, color='#f59e0b', width=.53, label='BGE600: загрузка + векторы')
    axes[1].bar(positions, cpu_recorded, bottom=qwen+bge, color='#10b981', width=.53, label='Измеренные стадии предиктора')
    for p, value in zip(positions, sums):
        axes[1].text(p, value+2, f'{value:.2f} с'+(' *' if p == 2 else ''), ha='center', fontsize=11)
    axes[1].set_xticks(positions, ['GP', 'KRR', 'Random'])
    axes[1].set_ylim(0, 165)
    axes[1].set_ylabel('Сумма записанных компонентов, с')
    axes[1].set_title('Основной расход — Qwen и BGE')
    axes[1].legend(frameon=False, loc='upper left', fontsize=8.5)
    axes[1].grid(axis='y', alpha=.18)
    fig.suptitle('Замена GP на KRR: заметная доля CPU, малая доля полной суммы', fontsize=14, y=.99)
    fig.text(.5, .035, '* Время выбора Random отдельно не измерялось. Общие загрузки Qwen и подготовка вне этой суммы.',
             ha='center', fontsize=9)
    fig.text(.5, .005, 'Это расчёт из одного прогона, не три end-to-end запуска. GP−KRR ≈ 0,18 с, или 0,15%. Память не сравнивалась.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, .93))
    name = 'fresh_cost_components.png'
    save(fig, name)
    PLOTTED[name] = dict(methods=methods, predictor_cpu_seconds=cpu,
                        qwen252_seconds=qwen.tolist(), bge600_seconds=bge.tolist(),
                        component_sum_seconds=sums.tolist(), random_selection_time='not measured',
                        source=SOURCE_PATHS[2])


if __name__ == '__main__':
    search_three_pools()
    fresh_search_and_gate()
    fresh_domains()
    fresh_map_tradeoff()
    fresh_costs()
    (OUT/'fresh_plotted_values.json').write_text(json.dumps(PLOTTED, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = dict(source_sha256={name: hashlib.sha256((WORKSPACE/name).read_bytes()).hexdigest()
                                   for name in SOURCE_PATHS},
                    plots={name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in PLOTTED},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    scope='Figures from existing audited results; no new experiments or model fits')
    (OUT/'current_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('CREATED_CURRENT_FIGURES', len(PLOTTED), flush=True)
