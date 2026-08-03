# Scaling laws

Так как основная идея проекта это сократить размер датасета, это основная тема

Современные адаптации закона Чинчиллы делятся на следующие категории

## 0. Классический Chinchilla Scaling Law

Идея: Базовый закон Compute-Optimal Training. Для минимизации Loss при заданном бюджете вычислений $C \propto N \cdot D$ параметры ($N$) и токены ($D$) должны расти равномерно в пропорции $\approx 1:20$ ($D \approx 20N$).

Формула:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

где $E$ — неустранимый шум (irreducible loss), $A, B, \alpha, \beta$ — эмпирические константы ($\alpha \approx 0.34, \beta \approx 0.28$).

Статья: Hoffmann et al., 2022 — «Training Compute-Optimal Large Language Models»

В оригинальной статье DeepMind (Hoffmann et al., 2022) подгонкой под результаты 400+ экспериментов были получены следующие значения:
$A = 406.4$ $B = 410.7$ $\alpha = 0.34$ $\beta = 0.28$

## 1. Data Mixing Laws (Смешивание данных)

Идея: Данные поступают из разных доменов (код, математика, веб). Вместо скаляра $D$ учитывается вектор доменов, а оптимизация ищет вектор пропорций $p_k = D_k / D_{\text{total}}$, дающий наименьший Loss.

Формула:

$$L(N, D) = L_0 + \sum_{k=1}^K \frac{A_k}{N^{\alpha_k}} + \sum_{k=1}^K \frac{B_k}{D_k^{\beta_k}}, \quad p^* = \arg\min_p L(N, p \cdot D_{\text{total}})$$

Статья: Zhan et al., 2024 — «Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance»

## 2. Vocabulary Scaling (Масштабирование словаря)

Идея: Размер словаря $V$ влияет на распределение параметров. Оптимальный размер словаря $V^*$ растёт сублинейно относительно размера модели $N$.

Формула:

$$L(N, D, V) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + \frac{C}{V^\gamma} + L_0, \quad V^* \propto N^\delta \quad (\delta \approx 0.4 \text{--} 0.5)$$

Статья: Chen et al., 2024 — «Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies»

## 3. Epoch Scaling Laws (Повторение данных)

Идея: При дефиците уникальных данных $D_{\text{unique}}$ повторные проходы (эпохи $E$) дают угасающий полезный эффект за счёт сублинейного роста эффективного объема $D_{\text{eff}}$.

Формула:

$$D_{\text{eff}}(E) = D_{\text{unique}} \cdot E^\gamma \implies L(N, D_{\text{unique}}, E) = \frac{A}{N^\alpha} + \frac{B}{(D_{\text{unique}} \cdot E^\gamma)^\beta} + L_0 \quad (\gamma \approx 0.5 \text{--} 0.7)$$

Статьи:

Sardana et al., 2024 — «Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws»

Muennighoff et al., 2023 — «Scaling Data-Constrained Language Models»

## 4. Data Quality Multiplier (Качество данных)

Идея: Токены имеют разную информационную плотность. Вводится множитель качества $Q$, переводящий сырые токены в «эталонный» эквивалент $D_{\text{eff}} = Q \cdot D$.

Формула:

$$L(N, D, Q) = \frac{A}{N^\alpha} + \frac{B}{(Q \cdot D)^\beta} + L_0$$

Статьи:

Microsoft Research (Phi series) — Phi-1.5, 

Phi-3Penedo et al., 2024 — «The FineWeb Datasets»

## 5. Inference-Aware Scaling (Учёт инференса)

Идея: Оптимизируется не только Compute на обучение $C_{\text{train}}$, а полная стоимость владения (TCO). Для нагруженного инференса выгоднее переобучать меньшую модель далеко за пределы оптимума Чинчиллы.

Формула:

$$ \text{TCO}(N, D) = C_{\text{train}}(N, D) + C_{\text{infer}}(N, Q_{\text{total}}), \quad (N_0, D_0) = \arg\min_{N,D} \text{TCO}(N, D) $$

Статья: Sardana et al., 2024 — «Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws»

## 6. Multilingual Scaling (Многоязычность)

Идея: Учитывается плотность информации $\rho_k$ для каждого языка $k$, так как морфология языков по-разному расходует токены.

Формула:

$$L(N, D) = L_0 + \sum_{k=1}^K \frac{B_k}{(\rho_k \cdot D_k)^\beta}$$

Статьи:
Wei et al., 2024 — «Scaling Laws for Downstream Task Performance»
BigScience, 2022 — «BLOOM 176B»
