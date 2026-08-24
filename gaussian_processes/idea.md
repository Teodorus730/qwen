# 0. Пролог

**Задача:** сравнение двух LLM без прогона через всё множество данных. Проверка функциональной эквивалентности (FE) на выборке с количественной оценкой уверенности.

**Инструмент:** гауссовские процессы (GP) как суррогатная модель поведения LLM в латентном пространстве контекстов.

**Объект сравнения:** локальные next-token распределения pretrained LLM при фиксированных префиксах. Не утверждаем полную sequence-level эквивалентность.

**Два пути:** **(1) практический** — GP как суррогат для скалярного расхождения, active search нарушений; **(2) академический** — GP над low-dimensional discrepancy field, сравнение posterior processes.

**Задел:** если метод заработает, естественное продолжение — итеративная дистилляция: FE-проверка между student и expert находит контексты-нарушения, те идут в training data student, повторяем до достижения эквивалентности. Применяя последовательно к разным доменам (coder → biologist → ...), можно собирать универсальную модель, где каждый домен закрыт целенаправленно отобранными данными, а не случайной выборкой.

---

# 1. Теория

## 1.1 LLM как conditional predictor

LLM — не статическая функция $f: \mathbb{R}^n \to \mathbb{R}^m$, а условное распределение:

$$p_\theta(y_t \mid c)$$

где $c$ — контекст.

## 1.2 Скалярное расхождение

Для контекста $c$ и двух моделей $A$, $B$ с общим токенизатором:

$$d(c) = JS\bigl(\tilde p_A(\cdot \mid c),\; \tilde p_B(\cdot \mid c)\bigr)$$

## 1.3 Эквивалентность на домене

$$\mathbb{P}_{c \sim \mathcal{D}}\bigl(d(c) \leq \varepsilon\bigr) \geq 1 - \delta$$

**Оценка через posterior GP.** GP строится отдельно внутри каждого домена на примерах из этого домена. После обучения на $\{z_i, d_i\}_{i=1}^N$ GP даёт предсказание с неопределённостью:

$$d(z) \mid \mathcal{D} \sim \mathcal{N}\bigl(\mu(z),\, \sigma^2(z)\bigr)$$

Для каждой точки вероятность $d(z) \leq \varepsilon$:

$$\mathbb{P}(d(z) \leq \varepsilon \mid z, \mathcal{D}) = \Phi\!\left(\frac{\varepsilon - \mu(z)}{\sigma(z)}\right)$$

Усредняем по тестовому множеству из домена:

$$\hat\theta = \frac{1}{M_{\text{test}}} \sum_{j=1}^{M_{\text{test}}} \Phi\!\left(\frac{\varepsilon - \mu(z_j^{\text{test}})}{\sigma(z_j^{\text{test}})}\right)$$

**Выбор $\varepsilon$ и тестового множества:** открытые вопросы.

## 1.4 Входное пространство

Контекст $c$ кодируется через независимый encoder. GP работает **для ранжирования реальных текстов из пула** $\mathcal{U} = \{c_1, \dots, c_M\}$, не для генерации произвольных векторов. Это снимает проблему интерполяции вне текстового многообразия.

---

# 2. Путь 1: Практический

## 2.1 Постановка

GP над скалярной мерой расхождения:

$$d(z) \sim \text{GP}\bigl(\mu(z),\; k(z, z')\bigr)$$

Цели:

1. Оценить $\Pr(d(c) \leq \varepsilon)$ по доменам
2. Найти контексты с максимальным расхождением

## 2.2 Pipeline

**Этап 1 — Пул кандидатов и покрытие (отдельно по доменам):**

- Для каждого домена свой пул текстов из корпуса / бенчмарков
- Кластеризация по эмбедингам внутри домена
- данные для обучения GP: top-k точки к центроидам кластеров

**Этап 2 — GP:**

- строим GP на ближайших к центроиду точках
- возьмем какую-нибудь acquisition function и будем искать данные, на которых модельки расходятся больше всего

**Этап 3 — Оценка эквивалентности:**

- $\hat\theta = \Pr(d(c) \leq \varepsilon)$ по доверительному интервалу

## 2.3 Что это даёт

Метод позволяет найти тексты и целые домены, где две модели расходятся больше всего, не прогоняя через них все данные. Для каждого домена строится отдельная оценка эквивалентности $\hat\theta$ — получается карта: вот тут модели почти одинаковы, а вот тут — нет. Дальше это можно использовать по-разному: целенаправленно собирать данные для дистилляции в слабых доменах, сертифицировать модель для конкретного применения, отслеживать регрессию при обновлениях.

## 2.4 Риски

- GP концептуально не обязателен (может быть заменён другим surrogate)
- Нужно доказывать преимущество gp над random search для поиска расхождений

---

# 3. Путь 2: Академический

## 3.1 Постановка

GP над самими выходами моделей (low-dimensional discrepancy field), а не над скалярным агрегатом.

**Ключевое отличие от Пути 1:** GP описывает функцию $\Delta(z) = f_A(z) - f_B(z)$, а не свёрнутую меру $d(z) = D(p_A, p_B)$. Сравнение — через posterior processes, а не через scalar threshold.

## 3.2 Проблема полных выходов

Next-token distribution живёт на симплексе: $p_t \geq 0$, $\sum_t p_t = 1$. Обычный GP даёт гауссовский выход — может предсказывать отрицательные вероятности.

Размерность: $|V| \sim 10^5$ — multi-output GP невозможен напрямую.

Решение: **low-rank representation разности логитов.**

## 3.3 Low-rank discrepancy field

**Шаг 1.** Для каждого контекста получаем log-probability difference:

$$r(z) = \log p_A(\cdot \mid z) - \log p_B(\cdot \mid z) \in \mathbb{R}^{|V|}$$

**Шаг 2.** PCA/SVD на calibration set:

$$r(z) \approx U h(z), \quad U \in \mathbb{R}^{|V| \times L}, \quad h(z) \in \mathbb{R}^L$$

$L = 16, 32, 64, 128 \ll |V|$.

**Шаг 3.** Multi-output GP (или independent GP на каждую компоненту):

$$h(z) \sim \text{GP}(0, K)$$

**Шаг 4.** Эквивалентность:

$$\|h(z)\\_2 \leq \varepsilon \quad \Longleftrightarrow \quad \Delta(z) \approx 0$$

Или через восстановленную метрику:

$$\hat{d}(z) = JS\bigl(\text{softmax}(\log p_B + U h(z)),\; \text{softmax}(\log p_B)\bigr)$$

## 3.4 Сравнение posterior processes

На тестовом множестве $Z_* = \{z_1^*, \dots, z_T^*\}$:

$$h^* \mid \mathcal{D} \sim \mathcal{N}(m^*, C^*)$$

Эквивалентность:

$$\Pr\bigl(\|h(z)\|_2 \leq \varepsilon \;\forall\, z \in Z_*\bigr)$$

Или Expected Violation:

$$\alpha(z) = \mathbb{E}[\|h(z)\|] + \beta\,\sigma_{\|h(z)\|}$$

**Альтернатива — paired difference GP:** вместо двух отдельных GP строим один GP на $\Delta(z) = f_A(z) - f_B(z)$. Нулевая гипотеза $\Delta = 0$ соответствует эквивалентности. Чище, чем сравнение двух независимых GP (которые могут различаться из-за разной покрытости данных, а не из-за разницы LLM).

## 3.5 Сравнение с Путём 1

| Критерий        | Путь 1 (scalar)                 | Путь 2 (functional)                                |
| --------------- | ------------------------------- | -------------------------------------------------- |
| Объект GP       | $z \mapsto d(z) \in \mathbb{R}$ | $z \mapsto h(z) \in \mathbb{R}^L$                  |
| Роль GP         | Surrogate для active sampling   | Описание функции модели                            |
| Эквивалентность | $\Pr(d \leq \varepsilon)$       | $\Pr(\|h\| \leq \varepsilon)$                      |
| Новизна         | Ниже                            | Выше                                               |
| Сложность       | Средняя                         | Высокая                                            |
| Риски           | GP может быть не нужен          | Low-rank approximation, kernel choice, calibration |

## 3.6 Плюсы и минусы

| +                                                     | -                                                                                       |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------- |
| GP концептуально в центре метода                      | Сложнее в реализации                                                                    |
| Ближе к исходной идее «описать LLM как функцию»       | Нужно доказывать адекватность low-rank представления                                    |
| Больше новизны                                        | Больше рисков при рецензировании                                                        |
| Сохраняет структуру различий (не схлопывает в скаляр) | Детерминированность LLM → GP uncertainty отражает выбор точек, не стохастичность модели |

---

# 4. Инженерные трюки

## 4.1 Union top-$k$ + REST: детали реализации

```python
def reduced_dist(p, top_k_indices, union_set):
    p_tilde = np.zeros(len(union_set) + 1)  # +1 for REST
    for i, t in enumerate(union_set):
        p_tilde[i] = p[t] if t in top_k_indices else 0.0
    p_tilde[-1] = max(0.0, 1.0 - p_tilde[:-1].sum())  # REST
    return p_tilde
```

Численная стабильность: если REST $< 0$ из-за floating point — обнулить и перенормировать.

## 4.2 Kernel choice в high-dimensional space

RBF в $\mathbb{R}^{768}$ страдает от curse of dimensionality — расстояния плохо различимы.

- PCA projection на $\mathbb{R}^{50\text{--}100}$ перед ядром
- Learned kernel (deep kernel learning)
- Composite kernel: $k(z, z') = k_{\text{sem}}(z, z') \cdot k_{\text{domain}}(z, z')$ где второй фактор по domain label
- Ablation обязательна

## 4.3 Efficient GP

- **SVGP** с inducing points — основа
- **Online GP** для streaming оценки (добавляем точки по мере вычисления)
- **GPyTorch** с GPU-ускорением
- **Preconditioning** для стабильности при $N > 10^4$
- Аппроксимация GAIA-style: обновлять GP не на каждой итерации, а по батчам

## 4.4 Acquisition с cluster quotas

```python
def batch_acquisition(gp, pool, clusters, b, epsilon):
    # Глобальный UCB
    ucb = gp.mean + beta * gp.std

    # Per-cluster лучшие кандидаты
    cluster_best = {}
    for c_id in clusters:
        mask = (cluster_labels == c_id) & (~evaluated)
        if mask.any():
            cluster_best[c_id] = pool[mask][ucb[mask].argmax()]

    # Batch: b_exploit + b_explore + b_coverage
    batch = []
    batch.extend(top_k(ucb[~evaluated], b_exploit))
    batch.extend(sample_by_uncertainty(gp, pool, b_explore))
    batch.extend(least_covered_clusters(cluster_best, b_coverage))
    return batch
```

## 4.5 Handling разных токенизаторов

```python
ndef cross_tokenizer_js(pA, tokA, pB, tokB, k=50):
    # Вариант 1: canonical surface forms
    tokens_A = {tokA.decode([i]): pA[i] for i in top_k(pA, k)}
    tokens_B = {tokB.decode([i]): pB[i] for i in top_k(pB, k)}
    # Align по surface strings, JS по aligned distribution

    # Вариант 2: OT с семантической стоимостью
    # C_ij = 1 - cos(embed_A[i], embed_B[j])
    # OT = sinkhorn(pA_topk, pB_topk, C)
    pass
```

## 4.6 Calibration $\varepsilon$

```python
def calibrate_epsilon(model, corpus, n_samples=1000):
    # Noise floor
    d_self = [js(model(c, seed=0), model(c, seed=1)) for c in sample(corpus, n_samples)]
    noise_floor = np.percentile(d_self, 95)

    # Paraphrase invariance
    d_para = [js(model(c), model(paraphrase(c))) for c in sample(corpus, n_samples)]
    para_ceiling = np.percentile(d_para, 95)

    return noise_floor, para_ceiling
```

---

# 5. Future Vision

## 5.1 Dataset selection для универсальных моделей

Идея: использовать FE-проверку для итеративного отбора данных при дистилляции.

**Pipeline:**

1. Есть expert-модель (например, coder) и student
2. Проверяем FE: $\Pr(d(c) \leq \varepsilon) \geq 1 - \delta$?
3. Если нет — находим контексты-нарушения через active search
4. Добавляем эти контексты в training data student
5. Переобучаем student
6. Повторяем до достижения FE

**Симбиоз:** последовательное применение к разным доменам:

$$\text{Coder Expert} \xrightarrow{\text{FE check + data select}} \text{Student}_1 \xrightarrow{\text{FE check + data select}} \text{Biologist Expert} \xrightarrow{} \text{Student}_2 \xrightarrow{} \cdots$$

На каждом шаге: GP-суррогат находит, где student расходится с expert, данные для закрытия gap отбираются автоматически.

**Требование:** dataset selection должен быть целенаправленным — не случайная выборка, а именно те контексты, где FE нарушается. GP с acquisition function даёт именно это.

## 5.2 Domain-aware equivalence maps

Карта эквивалентности по доменам:

| Домен        | $\hat\theta$ | $\varepsilon$ | $N$ оценено | Стабильность |
| ------------ | ------------ | ------------- | ----------- | ------------ |
| General text | 0.93         | 0.05          | 500         | ±0.02        |
| Code         | 0.71         | 0.05          | 300         | ±0.04        |
| Math         | 0.85         | 0.05          | 200         | ±0.03        |
| Safety       | 0.62         | 0.05          | 400         | ±0.05        |

Позволяет: (a) сертифицировать модели для конкретных доменов, (b) целенаправленно улучшать слабые домены.

## 5.3 Continuous equivalence monitoring

При обновлении модели (fine-tuning, quantization, pruning) — автоматически пересчитывать FE-карту и детектировать регрессию. GP позволяет делать это на подмножестве данных, переиспользуя структуру из предыдущих оценок.
