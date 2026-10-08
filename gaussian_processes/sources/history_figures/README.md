# Графики полной истории GP-ветки

Восемь графиков результатов экспериментов и скрипты их построения по сохранённым JSON.

## Воспроизведение

```bash
python -m pip install numpy==2.1.3 matplotlib==3.11.2
python create_historic.py
python create_current.py
```

Скрипты читают исходные данные из `data/` и записывают PNG в эту папку.

## Файлы

| Файл | Содержание |
|---|---|
| `predictors_mae.png` | Test MAE одиннадцати предикторов на четырёх пулах |
| `acquisition_controlled.png` | Поиск top-10 и ошибка карты при четырёх правилах выбора |
| `map_confirmations.png` | GP/SRS: ошибка θ и порядок доменов на новых пулах 7 октября |
| `search_three_pools.png` | Поиск top-10 на трёх пулах |
| `fresh_search_and_gate.png` | Поиск на новом пуле и допуск KRR относительно GP |
| `fresh_domain_search.png` | Поиск KRR, GP и random по шести доменам |
| `fresh_map_tradeoff.png` | Карта по случайным 192 и 252 измерениям |
| `fresh_cost_components.png` | Компоненты времени Qwen, BGE и предикторов |

Исходные JSON находятся в `data/`, использованные числа — в трёх одноимённых JSON исторических графиков и `fresh_plotted_values.json`, SHA256 источников, PNG и скриптов — в `historic_manifest.json` и `current_manifest.json`.
