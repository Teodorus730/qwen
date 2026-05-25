# Аудит 01: проблемы структуры и организации репозитория

Дата: 2026-05-26

Область аудита: только статический осмотр репозитория. Pipeline-скрипты, HF streaming, model loading, network access, commit, push и cleanup не запускались. Существующие generated samples и старые sweep outputs только просматривались по именам, путям и git status.

## Краткий вывод

Stage2 технически найти можно, но организационно он выглядит как подпроект, который рос через много быстрых рабочих итераций. Новый участник команды, скорее всего, разберется, где активный pipeline, но только после чтения нескольких README, планов, отчетов и списков generated outputs.

Главная проблема: активная часть по corpus preparation и domain classification спрятана в `external_samples/stage2_chunking_sample_clean`, а корень репозитория все еще выглядит как черновой файлообменник. Внутри stage2 рядом лежат код, fixtures, tiny real samples, generated outputs, отчеты, планы, future NLL-материалы и остатки sweep-запусков. Единой авторитетной карты пока нет.

## Шкала severity

- High: может ввести нового участника в заблуждение или привести к работе не в том файле/пути.
- Medium: создает трение, лишнее чтение и неопределенность, но вряд ли сразу ломает работу.
- Low: проблема полировки, naming convention или локального удобства.

## Краткая карта структуры

### Корень репозитория

- `README.MD`: очень короткая заметка, что репозиторий черновой и похож на файлообменник.
- `docs/`: две общие заметки про chunking/classification и taxonomies.
- `external_samples/stage2_chunking_sample_clean/`: фактически активный stage2-подпроект для chunking, metadata, rule-based/lexical labels, tiny samples и будущего embedding baseline.
- `research/`: исследования context window/NLL/next-token, notebooks, reports, PDF и графики.
- `data-filtering/`: более ранние notebooks, text sample и короткая markdown-заметка.
- `datasets/`: заметки про datasets от разных участников.
- `distribution/`: заметки и графики про distribution/tokenization.
- `results of the meetings/`: meeting notes.
- Корневые файлы `fasttext.md`, `hypothesis.md`, `metrics.md`, `PROJECT_STATE_REPORT.*`, `qwen-distributions*.ipynb`, `Passport draft.md`: смешанные отчеты, notebooks, drafts и артефакты.
- `scripts/`: содержит zero-byte `sample_fineweb_chunks.py`, хотя активные скрипты находятся внутри stage2.
- `data_samples/`: корневая sample-папка только с `README.md`; игнорируется через `.gitignore`.

### Stage2-подпроект

- `README.md`, `CHECK_HOW_TO.md`, `MESSAGE_TO_TEAM.txt`: входные human-readable файлы, но они отражают разные моменты развития проекта.
- `scripts/`: активные utilities и pipeline scripts:
  - chunking: `sample_fineweb_chunks.py`;
  - validation/inspection: `validate_chunks.py`, `inspect_chunks.py`;
  - classifiers: rule-based, lexical, optional embedding baseline;
  - evaluation/comparison/review scripts;
  - planning scripts для real samples и probability-profile manifests.
- `examples/`: локальные input JSONL fixtures.
- `config/dataset_sources.json`: registry для local и planned HF sources.
- `taxonomy/simple_domain_labels.json`: список domain/field/subfield labels.
- `data_samples/`: generated и/или committed local benchmark outputs, edge-case outputs, run stats, probability manifest и `README.md`.
- `data_samples/real_samples/`: tiny FineWeb-Edu, FineMath, local real-like outputs, labels, stats и NLL pilot candidate manifest.
- `docs/`: около 30 markdown-файлов: schemas, reports, audits, plans, checklists, runbooks, local env notes, real sample review, NLL notes, taxonomy notes.

## Найденные организационные проблемы

### 1. Активный stage2 спрятан в пути `external_samples`

Severity: High

Наблюдения:

- Активный код находится в `external_samples/stage2_chunking_sample_clean/scripts/`.
- Активные docs и outputs находятся в `external_samples/stage2_chunking_sample_clean/docs/` и `data_samples/`.
- Название `external_samples/stage2_chunking_sample_clean` звучит как временный sample/export, а не как основной модуль текущей работы.

Почему мешает команде:

Новый участник сначала пойдет в root `scripts/`, root `docs/` или `data-filtering/`, и только потом обнаружит настоящий stage2. Путь также смешивает ощущение source code и external sample: кажется, будто все внутри является внешним примером, хотя это текущая рабочая зона.

Что исправить позже:

Immediate scope correction: сейчас не стоит переносить stage2 наверх и не стоит переписывать root repo. Изоляция в `external_samples/stage2_chunking_sample_clean/` полезна для этой ветки, потому что снижает риск задеть чужие части проекта.

Ближайшее исправление должно быть stage2-local: сделать подпроект самодостаточным внутри текущей папки через README, docs index, pipeline map, source status и data policy. Переименование, root-level pointer или перенос в более явное место вроде `stage2_chunking/`, `corpus_preparation/` или `data_domain_classification/` стоит обсуждать позже с командой и только после explicit approval.

### 2. Корень репозитория все еще выглядит как черновой файлообменник

Severity: High

Наблюдения:

- `README.MD` почти не объясняет проект и прямо говорит, что репозиторий черновой.
- В корне вместе лежат docs, notebooks, PDF, meeting notes, dataset notes, research reports и project-state artifacts.

Почему мешает команде:

Корень не отвечает на первые вопросы нового участника: что актуально, что архивное, кто за что отвечает, что можно запускать, что нельзя трогать без approval. Важный контекст живет вне репозитория, а не в его структуре.

Что исправить позже:

Заменить root README на компактную карту:

- current active areas;
- inactive/archive areas;
- ownership/responsibility notes;
- предупреждения про HF/network/model tasks;
- ссылка на stage2 pipeline map.

### 3. Есть конфликт между root `scripts/` и stage2 `scripts/`

Severity: High

Наблюдения:

- В root `scripts/` лежит zero-byte `sample_fineweb_chunks.py`.
- Реальная реализация находится в `external_samples/stage2_chunking_sample_clean/scripts/sample_fineweb_chunks.py`.

Почему мешает команде:

Это ловушка неверной точки входа. Участник может открыть или запустить root script path, решить, что implementation потерян, или начать создавать duplicate script в неправильном месте.

Что исправить позже:

Выбрать один canonical code location. Root stub позже лучше удалить, заменить pointer-файлом или перенести активные stage2 scripts в top-level package/module. Это нужно делать отдельной cleanup-задачей, не в рамках аудита.

### 4. Inputs, fixtures, generated outputs и review artifacts лежат слишком близко

Severity: High

Наблюдения:

- Local inputs лежат в `examples/`.
- Generated local benchmark outputs лежат в `data_samples/`.
- Tiny real samples лежат в `data_samples/real_samples/`.
- Run stats, labeled outputs, probability manifests и NLL candidate manifests лежат рядом с sample data.
- `git status` показывает untracked generated variants:
  - `classifier_benchmark_chunks_maxdocs20.jsonl`;
  - `classifier_benchmark_chunks_maxdocs40.jsonl`;
  - `classifier_benchmark_chunks_target120.jsonl`;
  - `classifier_benchmark_labeled_target120.jsonl`;
  - matching `run_stats_*` files.

Почему мешает команде:

Сложно понять, какие файлы являются stable fixtures, какие являются reproducible outputs, какие являются старыми sweeps, а какие актуальным evidence. Это повышает риск смотреть stale files, случайно закоммитить generated leftovers или перезаписать полезные samples.

Что исправить позже:

Разделить stage2 data по ролям, например:

- `fixtures/` или `examples/inputs/` для hand-written local inputs;
- `outputs/generated/` для reproducible generated outputs;
- `outputs/real_samples/` для tiny real samples;
- `outputs/archive/sweeps/` для старых parameter sweeps;
- `reports/` или `docs/reports/` для human summaries.

Добавить короткую data policy: что коммитится, что игнорируется, что review-only, что можно регенерировать.

### 5. Stage2 docs плоские и многочисленные, но без индекса авторитетности

Severity: Medium

Наблюдения:

- В `docs/` около 30 markdown-файлов.
- В одной плоской папке лежат schemas, reports, audits, checklists, runbooks, future plans, environment checks и long review notes.
- Актуальный статус размазан между `next_steps_checklist.md`, `real_sample_next_plan.md`, `dataset_source_registry.md`, `fineweb_edu_tiny_sample_report.md`, `finemath_tiny_sample_report.md` и `real_sample_disagreement_review.md`.

Почему мешает команде:

Новый участник не видит, какой документ является canonical current state. Порядок чтения важен, но директория его не задает. Старые планы и completed reports выглядят одинаково авторитетно.

Что исправить позже:

Добавить `docs/README.md` как index:

- start here;
- current pipeline and schema;
- current results;
- review docs;
- future/later docs;
- archived notes.

Опционально разложить docs по подпапкам `schema/`, `reports/`, `plans/`, `reviews/`, `archive/`.

### 6. Current scope размывается future NLL/probability material

Severity: Medium

Наблюдения:

- В stage2 есть `scripts/prepare_probability_profile_manifest.py`.
- В docs есть `probability_profile_schema.md`, `nll_pilot_candidate_notes.md`, `nll_scoring_next_steps.md`.
- В `data_samples/real_samples/` tracked файл `nll_pilot_candidates.jsonl`.
- При этом текущая зона ответственности: corpus preparation и domain classification, не NLL/logprob/effective context window.

Почему мешает команде:

Новый участник может решить, что NLL work входит в активный stage2 deliverable, и начать двигать не ту часть проекта. Это также размывает следующий понятный шаг: embedding-based nearest-label classifier на MiniLM/Sentence-Transformers.

Что исправить позже:

Перенести NLL/probability материалы в явно названную зону `future_nll/` или `later_probability_profiling/`, либо пометить их в docs index как future only, not current MVP responsibility. В current pipeline map оставить короткую заметку, что это downstream consumer, а не текущая задача.

### 7. Dataset registry/status сигналы частично противоречат друг другу

Severity: Medium

Наблюдения:

- `config/dataset_sources.json` все еще помечает FineMath как `needs_verification: true`.
- `docs/dataset_source_registry.md` говорит, что FineWeb-Edu и FineMath tiny samples уже completed.
- В одном месте registry doc перечисляет FineWeb-Edu среди entries, которым нужна verification, хотя в JSON у FineWeb-Edu стоит `needs_verification: false`.
- OpenWebMath есть в registry рядом с активными источниками, хотя должен читаться как optional/later.

Почему мешает команде:

Команда теряет уверенность, источник planned, verified, sampled, reviewed или out of scope. Для current MVP важно, чтобы FineMath читался как единственный активный math dataset, а OpenWebMath визуально оставался optional/later comparison.

Что исправить позже:

Ввести единую source-status модель:

- `planned`;
- `verified`;
- `sampled`;
- `labeled_rule_based`;
- `labeled_lexical`;
- `reviewed`;
- `optional_later`.

Поддерживать одну таблицу статусов на основе `dataset_sources.json` и sample reports. OpenWebMath отдельно пометить как optional/later.

### 8. Pipeline не представлен одной canonical картой

Severity: Medium

Наблюдения:

- Команды pipeline разбросаны по `README.md`, `MESSAGE_TO_TEAM.txt`, `next_steps_checklist.md`, `real_sample_next_plan.md` и runbook/planning docs.
- Фактический flow понятен:
  `raw/local/HF datasets -> logical chunks -> dataset/source_type metadata -> domain labels -> rule/lexical baselines -> real tiny samples -> disagreement review -> embedding baseline`.
- Но этот flow не закреплен в одном canonical документе с inputs и outputs.

Почему мешает команде:

Можно запустить валидные команды в неправильном порядке, сравнить outputs от разных generations или не понять, какие files являются expected outputs каждого stage.

Что исправить позже:

Добавить `docs/pipeline_map.md` или `PIPELINE.md` с таблицей:

- stage;
- script;
- input path;
- output path;
- status;
- safe to run locally;
- requires HF/network/model.

В этом документе явно указать, что следующий current step: embedding nearest-label classification, без model download без approval.

### 9. Naming conventions непоследовательны

Severity: Low

Наблюдения:

- Смешаны extensions/casing: `README.MD`, `research.MD`, `datasets.MD`, обычные `.md`.
- Есть paths with spaces: `results of the meetings/`, `Passport draft.md`.
- Видны исторические typo/names, например `ppl-with-trreshhold-0.2.png`.
- Hyphen/underscore styles смешаны между директориями и файлами.

Почему мешает команде:

Это не блокер, но повышает friction при поиске, делает ссылки менее переносимыми и создает ощущение случайной структуры.

Что исправить позже:

Принять простую naming convention для новых файлов. Старые файлы переименовывать только в отдельной cleanup-задаче, чтобы не сломать ссылки.

### 10. Generated samples committed без достаточно видимой reproducibility/commit policy

Severity: Medium

Наблюдения:

- Stage2 tracks small generated JSONL и stats files в `data_samples/` и `data_samples/real_samples/`.
- `real_samples_output_structure.md` содержит полезную commit-policy, но это не первый документ, который увидит новый участник.
- Root `.gitignore` игнорирует root `data_samples/*`, но stage2 `data_samples/` в основном tracked, кроме specific HF cache path и текущих untracked sweep outputs.

Почему мешает команде:

Tiny generated files могут быть полезными fixtures, но без явной policy они выглядят случайно. Новый участник может удалить useful evidence или, наоборот, закоммитить слишком много generated data.

Что исправить позже:

Создать заметный `DATA_POLICY.md` или расширить `data_samples/README.md`, где каждая группа samples классифицирована:

- canonical fixture;
- current generated benchmark output;
- tiny real sample for review;
- old sweep/archive;
- local-only cache.

Также расширить ignore rules для regenerated outputs, если команда решит держать их локально.

### 11. Local environment clutter виден в working tree

Severity: Low

Наблюдения:

- `.idea/`, `.venv/` и stage2 HF cache test directory игнорируются в status output.
- Локальный `.gitignore` явно игнорирует `.idea/` и stage2 HF cache test, но не `.venv/`; `.venv/` похоже игнорируется другим ignore layer.

Почему мешает команде:

Это не блокер, но reliance on global ignore state хрупок. У разных участников может быть разный git status noise.

Что исправить позже:

Добавить стандартные local-environment patterns в repo `.gitignore`: `.venv/`, `.env`, cache folders, notebook checkpoints, local model/cache directories.

### 12. UTF-8/Russian docs могут выглядеть сломанными в default Windows shell output

Severity: Low

Наблюдения:

- При чтении русских markdown-файлов default PowerShell output показал mojibake.
- При явном UTF-8 те же файлы отображались корректно.

Почему мешает команде:

Файлы usable, но быстрый terminal inspection может создать ложное впечатление, что docs corrupted. Для onboarding это неприятное трение.

Что исправить позже:

Добавить `.editorconfig` с UTF-8 и/или короткую Windows note в root README про ожидаемую encoding setup.

## Рекомендуемый порядок cleanup позже

1. Добавить stage2-local `docs/README.md` или `PIPELINE.md` как canonical current map.
2. Обновить stage2 README/current-status docs так, чтобы подпроект был понятен без root-level изменений.
3. Задокументировать неоднозначность root `scripts/` vs stage2 `scripts/`; сами root files не менять без отдельного approval.
4. Разделить fixtures, generated outputs, real tiny samples и archived sweeps.
5. Пометить future NLL/probability material как downstream/later.
6. Нормализовать source status, особенно FineMath current MVP и OpenWebMath optional/later.
7. Усилить `.gitignore` и правила для local environment/cache.
8. Нормализовать naming только после появления структурной карты.

Root README pointer, перенос stage2 или root-level restructuring нужно считать postponed/later actions, а не immediate cleanup. Их стоит делать только после отдельного согласования с командой.

## Current MVP interpretation, которую важно сохранить

- FineMath: единственный активный math dataset в current MVP.
- OpenWebMath: только optional/later comparison.
- Текущая зона ответственности: corpus preparation и domain classification.
- NLL/logprob/effective context window: downstream, не immediate task.
- Следующий current stage: embedding-based nearest-label classification на MiniLM/Sentence-Transformers, только после явного approval на dependencies/model availability.
