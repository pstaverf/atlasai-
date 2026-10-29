# Atlas AI

Atlas AI — локальная агентная система с **собственным LLM-бэкендом**: свой BPE-токенизатор, Transformer, обучение и инференс. Ollama и внешние LLM API не используются.

Реализованы **этапы 1–3**: каркас проекта, строгий TOML-конфиг, CLI, byte-level BPE, датасет, собственный causal Transformer и обучение маленькой модели на малых данных. LLM-адаптер, агенты, инструменты и память появятся на следующих этапах. Заглушек, выдающих будущую логику за готовую, нет.

## Ограничения и решения

- Целевая среда: Linux, CPU, небольшой VPS; ориентир по RAM — около 1 ГБ, точные характеристики пока не зафиксированы.
- Python: **3.12.x**. Обязательных runtime-зависимостей у каркаса нет.
- PyTorch **2.15.0** — опциональная зависимость `model`, только для этапов модели и обучения.
- Маленькая модель будет экспериментальной. На таком VPS не следует ожидать качества большой pretrained LLM.
- Сначала локальный вызов модели внутри Python-процесса, без отдельного HTTP-сервиса.
- Rust/C++ не добавляются до профилирования.
- Инструменты этапа 6 должны давать реальный веб-доступ: поиск, загрузка страниц по URL и извлечение текста, а не имитация результатов.

## Что уже работает

Этап 1:

- Установка пакета `atlasai` и консольной команды `atlasai`.
- Загрузка TOML через стандартный `tomllib`, без PyYAML/Pydantic.
- Проверка типов, диапазонов, неизвестных ключей, UTF-8 и размера конфигурации.
- Разрешение относительных путей от каталога конфигурационного файла.
- CLI `config`: нормализованный конфиг в JSON.
- CLI `doctor`: Python, платформа, видимые CPU/RAM, наличие optional torch и файлов модели.
- `doctor` не загружает веса и сообщает `model_loaded: false`; наличие файла не означает, что модель проверена. Для проверки чекпоинта есть `model evaluate`.

Этап 2:

- Byte-level BPE с нуля: 256 UTF-8 байтов + `<pad>`, `<bos>`, `<eos>`, без `UNK`.
- Детерминированное обучение merge-правил по частоте, сохранение/загрузка JSON.
- Encode/decode для русского текста, английского текста, пунктуации и emoji.
- Датасет из UTF-8 `.txt`: EOS между документами, окна `input_ids / target_ids` со сдвигом на один токен.
- Воспроизводимый train/val split и CLI-статистика датасета.
- Маленький пример корпуса: `examples/corpus.txt`.

Этап 3:

- Causal Transformer decoder с нуля: token/position embeddings, multi-head attention, pre-norm residual blocks, GELU MLP, language-model head.
- Обучение AdamW на CPU с фиксированным seed, одним потоком по умолчанию и gradient clipping.
- Оценка validation loss/perplexity, сохранение **лучшего**, а не последнего чекпоинта.
- Безопасная загрузка `weights_only=True`, проверка архитектуры и SHA-256 токенизатора.
- CLI `model train` и `model evaluate`, JSON-метрики и проверка повторной загрузки весов.

Загрузка конфига **не создаёт каталоги, не импортирует torch, не загружает веса и не обращается к сети**. Команды этапов 1–2 работают без PyTorch. `seed` и `num_threads` применяются при обучении и оценке модели.

## Структура

Уже создано:

```text
.
├── pyproject.toml
├── README.md
├── .gitignore
├── configs/
│   ├── atlas.toml
│   └── smoke.toml
├── examples/
│   └── corpus.txt
├── src/atlasai/
│   ├── __init__.py
│   ├── __main__.py
│   ├── py.typed
│   ├── config.py
│   ├── cli.py
│   ├── core/tokenizer/bpe.py
│   ├── core/model/transformer.py
│   └── training/
│       ├── dataset.py
│       ├── trainer.py
│       ├── evaluation.py
│       ├── checkpoint.py
│       └── errors.py
└── tests/
    ├── test_config.py
    ├── test_cli.py
    ├── test_bpe.py
    ├── test_dataset.py
    ├── test_cli_stage2.py
    ├── test_transformer.py
    ├── test_training.py
    ├── test_checkpoint.py
    └── test_cli_stage3.py
```

План следующих этапов, пока без фиктивных реализаций:

```text
src/atlasai/
├── core/model/          # 4: интерфейс LLM, AtlasLocalModel; Transformer уже есть
├── agents/              # 5 и 7: базовый агент, оркестратор, подагенты
├── tools/               # 6: реестр, web search, чтение страниц, безопасность
└── memory/              # 8: краткосрочная и долгосрочная память
```

## Установка для разработки

Нужен установленный Python 3.12 с поддержкой venv. Из корня репозитория:

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Без PyTorch доступны этапы 1–2; ML-тесты в таком окружении явно пропускаются. Для полного набора тестов и этапа 3 нужен extra `model`.

### PyTorch для CPU

Обычный Linux-wheel PyTorch тянет CUDA-зависимости. Для лёгкого CPU-варианта рекомендуемый способ — `uv` с явным выбором backend:

```bash
python -m pip install uv==0.13.6
uv pip install --torch-backend cpu -e ".[dev,model]"
```

Эта команда обращается к индексу PyTorch. В текущей песочнице `download.pytorch.org` недоступен по сетевым правилам, поэтому лёгкий `+cpu` wheel здесь **не проверялся**. Вместо него обучение проверено на CPU со стандартным `torch 2.15.0+cu132`: `cuda.is_available() == False`, вычисления на GPU не выполнялись. Стандартный пакет потребовал CUDA-библиотеки и больше диска; не считай его лёгкой установкой для VPS.

Перед установкой на VPS с 1 ГБ проверь `free -h` и `df -h`. Измеренный peak RSS smoke-прогона со стандартным пакетом — около **686 MiB**; это не измерение отдельного CPU-wheel. Если свободной RAM меньше 1 GiB, `doctor` выдаст предупреждение.

## Запуск

Команды выполняются из корня репозитория после установки:

```bash
atlasai --version
atlasai config --config configs/atlas.toml
atlasai doctor --config configs/atlas.toml
```

Эквивалент без console script:

```bash
python -m atlasai config --config configs/atlas.toml
python -m atlasai doctor --config configs/atlas.toml
```

Пути внутри TOML разрешаются относительно **самого TOML-файла**. Путь `--config` по умолчанию разрешается относительно текущего каталога shell. Поэтому для запуска из другого каталога передай абсолютный путь к конфигу.

Коды завершения:

- `0`: команда выполнена; для `doctor` — окружение пригодно для реализованных этапов, возможны информационные сообщения или предупреждения.
- `1`: ошибка конфига, данных, токенизатора или критическая ошибка окружения.
- `2`: ошибка аргументов CLI.

Отсутствие torch и чекпоинта модели — нормальное состояние до этапов 3–4, а не признак готовой модели. Секреты и `.env` сейчас не нужны; обязательного внешнего API нет.

### Токенизатор и датасет

Обучение BPE на примере корпуса и сохранение в путь из конфига:

```bash
atlasai tokenizer train --config configs/atlas.toml
```

Свой корпус и параметры:

```bash
atlasai tokenizer train \
  --config configs/atlas.toml \
  --input path/to/first.txt path/to/second.txt \
  --output checkpoints/tokenizer.json \
  --vocab-size 1024 \
  --min-frequency 2
```

Encode/decode:

```bash
atlasai tokenizer encode --config configs/atlas.toml --text "Привет, Atlas! 🙂" --bos --eos
atlasai tokenizer decode --config configs/atlas.toml --ids "1,211,162,2"
```

Статистика датасета:

```bash
atlasai dataset inspect \
  --config configs/atlas.toml \
  --context-length 64 \
  --stride 64 \
  --val-fraction 0.2
```

Ограничения этапа 2: корпус и токены загружаются в память, поэтому начинай с небольших `.txt`; BPE использует упрощённую whitespace-претокенизацию.

### Обучение и оценка Transformer

После установки PyTorch и обучения токенизатора:

```bash
atlasai tokenizer train --config configs/smoke.toml
atlasai model train --config configs/smoke.toml
atlasai model evaluate --config configs/smoke.toml
```

Smoke-конфиг: контекст 64, 2 слоя, 4 головы, ширина 64, 100 шагов, batch size 8. Артефакты:

- `checkpoints/smoke.pt` — лучшие по validation веса и метаданные;
- `checkpoints/smoke.metrics.json` — история train/validation, настройки и fingerprint токенизатора.

Можно переопределить часть параметров:

```bash
atlasai model train --config configs/smoke.toml --max-steps 20 --batch-size 4
```

Важные ограничения:

- Для обучения требуется `stride == context_length`: train/validation окна не должны содержать общие токены.
- Validation — отдельные окна того же маленького корпуса, не независимый документный benchmark. BPE в smoke-примере обучен на всём корпусе; для строгой оценки качества его нужно обучать только на train-части.
- При оценке используй тот же корпус, `seed` и `val_fraction`, что при обучении. Архитектура читается из чекпоинта.
- Сохраняется лучший чекпоинт; продолжение прерванного обучения с состоянием optimizer пока не реализовано.
- Загрузка не включает небезопасный pickle: `weights_only=True`, проверка формата, архитектуры, конечности весов и токенизатора.
- LLM-интерфейс и генерация для агентов — этап 4; сейчас проверяется обучение и next-token loss.

## Проверки

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Тесты покрывают конфиг, CLI, BPE, датасет, causal mask, градиенты, снижение loss, воспроизводимость, чекпоинты, отказ от unsafe pickle и работу базового CLI без PyTorch. LLM-адаптер и агентная генерация пока не тестируются, потому что относятся к следующим этапам.

### Фактическая проверка этапа 1

В песочнице выполнен запуск на **Python 3.12.12**:

- `python -m pytest -q` — **37 passed**.
- `python -m ruff check .` — без замечаний.
- `python -m ruff format --check .` — форматирование корректно.
- `python -m mypy` — без ошибок типизации.
- `python -m pip check` — конфликтов зависимостей нет.
- `atlasai config` и `atlasai doctor` — команды выполняются; `doctor` сообщает `model_loaded: false`.

PyTorch и CPU-wheel на этапе 1 не устанавливались. Проверка собственного инференса появится только после его реализации.

### Фактическая проверка этапа 2

В песочнице выполнен запуск на **Python 3.12.12**:

- `python -m pytest -q` — **88 passed**.
- `python -m ruff check .` и `python -m ruff format --check .` — без замечаний.
- `python -m mypy` — без ошибок типизации.
- `atlasai tokenizer train` на `examples/corpus.txt` — словарь **512**, **253** merge-правила.
- `atlasai dataset inspect` с context/stride 64 — **1349** токенов, **17** train-блоков и **4** val-блока.

### Фактическая проверка этапа 3

В песочнице выполнен запуск на **Python 3.12.12**, PyTorch **2.15.0+cu132**, вычисления **только на CPU**:

- `python -m pytest -q` — **136 passed**. Одно предупреждение PyTorch об отсутствии optional NumPy; Atlas его не использует.
- Отдельный прогон с заблокированным импортом torch — **101 passed, 5 skipped**: базовый CLI и конфиг работают без PyTorch.
- Ruff check/format, strict mypy и `pip check` — без ошибок.
- Smoke-модель: **136 960 параметров**, 100 optimizer steps.
- Validation loss: **6.2496 → 5.4067** на лучшем чекпоинте, снижение примерно **13.5%**.
- Повторная безопасная загрузка и оценка дали тот же loss: **5.4066681862**.
- Размер чекпоинта: **689 165 байт**; время цикла обучения без импорта PyTorch: около **1.4 с**.
- Peak RSS процесса со стандартным Linux-wheel: около **686 MiB**. Отдельный лёгкий CPU-wheel не измерялся.

Это подтверждает работу forward/backward, optimizer, evaluation и checkpoint reload. Качество рассуждений и обобщение на новых данных этим экспериментом не подтверждены.

## Этапы

| Этап | Содержание | Статус |
| --- | --- | --- |
| 1 | Структура, зависимости, конфиги, CLI, тесты | Реализован |
| 2 | BPE-токенизатор и датасет | Реализован |
| 3 | Transformer с нуля и обучение на малых данных | Реализован |
| 4 | Интерфейс LLM и собственный локальный бэкенд | Запланирован |
| 5 | Цикл агента «думай → действуй → наблюдай» | Запланирован |
| 6 | Инструменты, безопасность, реальный веб-поиск и чтение сайтов | Запланирован |
| 7 | Оркестратор и подагенты | Запланирован |
| 8 | Память, логирование, оценка | Запланирован |

После каждого этапа работа останавливается до отдельного подтверждения. Каждый завершённый этап фиксируется отдельным git-коммитом после проверок.

### Требование к веб-инструментам

На этапе 6 нужны настоящие инструменты, а не заглушки:

- `search`: запрос к поисковому провайдеру, результаты с заголовками, URL и сниппетами.
- `fetch`: HTTP-загрузка страницы по URL, извлечение читаемого текста, таймауты и лимиты размера.
- Агент должен видеть текст страницы и URL источника, чтобы опираться на факты, а не только на сниппеты.
- Ошибки сети, недоступные страницы и опасные действия обрабатываются явно; ничего не скрывается за пустыми `except`.
