# Atlas AI

Atlas AI — локальная агентная система с **собственным LLM-бэкендом**: свой BPE-токенизатор, Transformer, обучение и инференс. Ollama и внешние LLM API не используются.

Реализованы **этапы 1–4**: каркас проекта, строгий TOML-конфиг, CLI, byte-level BPE, датасет, собственный causal Transformer, обучение и локальный интерфейс генерации. Модель по умолчанию содержит **10 868 736 параметров**. Агенты, инструменты и память появятся на следующих этапах. Заглушек, выдающих будущую логику за готовую, нет.

## Ограничения и решения

- Целевая среда: Linux, CPU, небольшой VPS; ориентир по RAM — около 1 ГБ, точные характеристики пока не зафиксированы.
- Python: **3.12.x**. Обязательных runtime-зависимостей у каркаса нет.
- PyTorch **2.15.0** — опциональная зависимость `model`, только для команд модели и LLM.
- Модель экспериментальная и очень маленькая по меркам LLM. На таком VPS не следует ожидать качества большой pretrained LLM.
- Сначала локальный вызов модели внутри Python-процесса, без отдельного HTTP-сервиса.
- Генерация без KV-cache: каждый новый токен прогоняет текущий контекст целиком. Это просто, но медленнее оптимизированного инференса.
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

Этап 4:

- Torch-free интерфейс `LLM`, `GenerationResult`, `LLMError`: агентный слой сможет зависеть от контракта, а не от конкретной модели.
- `AtlasLocalModel`: загрузка токенизатора и чекпоинта, greedy и sampling, остановка по EOS, лимит новых токенов, усечение длинного prompt до контекста.
- PAD/BOS не могут быть сгенерированы как обычные токены. Невалидный UTF-8 в выводе заменяется на `U+FFFD`, а строгий roundtrip токенизатора остаётся строгим.
- CLI `llm info` и `llm generate`.
- Конфигурация по умолчанию: `d_model=384`, `layers=6`, `heads=8`, vocab 512, context 64, tied embeddings → **10 868 736 параметров**.
- Учебный конфиг `foundation.toml`: vocab 2048, context 256 → **11 532 288 параметров**, отдельные train/validation документы.

Загрузка конфига **не создаёт каталоги, не импортирует torch, не загружает веса и не обращается к сети**. Команды этапов 1–2 работают без PyTorch. `seed` и `num_threads` применяются при обучении, оценке и генерации.

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
│   ├── corpus.txt
│   └── foundation/
│       ├── train/       # 10 учебных документов
│       └── val/         # 10 отдельных validation-документов
├── src/atlasai/
│   ├── __init__.py
│   ├── __main__.py
│   ├── py.typed
│   ├── config.py
│   ├── cli.py
│   ├── core/tokenizer/bpe.py
│   ├── core/model/base.py
│   ├── core/model/local.py
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
    ├── test_cli_stage3.py
    ├── test_llm.py
    ├── test_cli_stage4.py
    ├── test_model_scale.py
    └── test_foundation.py
```

План следующих этапов, пока без фиктивных реализаций:

```text
src/atlasai/
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

Без PyTorch доступны этапы 1–2; ML-тесты в таком окружении явно пропускаются. Для полного набора тестов и этапов 3–4 нужен extra `model`.

### PyTorch для CPU

Обычный Linux-wheel PyTorch тянет CUDA-зависимости. Для лёгкого CPU-варианта рекомендуемый способ — `uv` с явным выбором backend:

```bash
python -m pip install uv==0.13.6
uv pip install --torch-backend cpu -e ".[dev,model]"
```

Эта команда обращается к индексу PyTorch. В текущей песочнице `download.pytorch.org` недоступен по сетевым правилам, поэтому лёгкий `+cpu` wheel здесь **не проверялся**. Вместо него обучение проверено на CPU со стандартным `torch 2.15.0+cu132`: `cuda.is_available() == False`, вычисления на GPU не выполнялись. Стандартный пакет потребовал CUDA-библиотеки и больше диска; не считай его лёгкой установкой для VPS.

### Память и требование 10M параметров

Измеренный smoke-прогон модели **10 868 736 параметров** со стандартным пакетом:

- обучение 200 шагов, batch size 1: peak RSS около **934 MiB**;
- генерация 8 токенов: peak RSS около **600 MiB**;
- чекпоинт: около **44 MB**.

На VPS с 1 ГБ обучение такой модели будет очень тесным: ОС и файловый кэш тоже требуют память. Практический вариант для 1 ГБ — использовать готовый чекпоинт для инференса, добавить RAM/swap или обучать на более мощной машине. Лёгкий CPU-wheel может снизить накладные расходы, но это здесь не измерялось. `doctor` предупреждает, если доступно меньше 1.5 GiB.

Перед установкой на VPS проверь `free -h` и `df -h`.

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
- `1`: ошибка конфига, данных, токенизатора, модели или критическая ошибка окружения.
- `2`: ошибка аргументов CLI.

Отсутствие torch и чекпоинта модели — нормальное состояние до обучения, а не признак готовой модели. Секреты и `.env` сейчас не нужны; обязательного внешнего API нет.

## Токенизатор и датасет

Минимальный пример корпуса уже лежит в репозитории:

```bash
atlasai tokenizer train --config configs/atlas.toml
atlasai dataset stats --config configs/atlas.toml
```

Ожидаемые результаты для примера: vocab size 512, 253 merge-правила, 1349 токенов, 17 train-блоков и 4 validation-блока. Это проверка конвейера, а не полноценный обучающий корпус.

Свой корпус:

1. Создай каталог с UTF-8 `.txt` файлами.
2. Укажи `corpus_dir` в `[data]`.
3. При необходимости измени `vocab_size`, `context_length`, `stride`, `validation_fraction`.
4. Запусти `tokenizer train`, затем `dataset stats`.

Для обучения Transformer `stride` должен быть равен `context_length`, чтобы окна не пересекались. Команда обучения проверяет это явно. `tokenizer train` не перезаписывает существующий токенизатор; для другого словаря укажи другой `output`.

## Обучение и оценка модели

После обучения токенизатора:

```bash
atlasai model train --config configs/smoke.toml
atlasai model evaluate --config configs/smoke.toml
```

`model train` пишет JSON-метрики в `training.metrics_output`, прогресс — в stderr. Чекпоинт содержит веса, конфигурацию модели, шаг, validation loss и SHA-256 токенизатора. Сохраняется лучший чекпоинт по validation loss, после чего он перезагружается и loss сверяется с точностью до `1e-6`.

Фактический smoke-результат этапа 4 в этой песочнице:

- параметров: **10 868 736**;
- шагов: 200;
- initial validation loss: **6.301777**;
- лучший validation loss: **5.829174** на шаге 160;
- время training loop: около **26.2 s**;
- peak RSS: около **934 MiB**.

Корпус крошечный, поэтому снижение loss проверяет работоспособность обучения, а не качество LLM. Генерация после smoke-обучения может быть бессвязной — это ожидаемо и не скрывается.

Полезные override без изменения TOML:

```bash
atlasai model train --config configs/atlas.toml --max-steps 10 --batch-size 2
atlasai model evaluate --config configs/atlas.toml --batch-size 2
```

## Учебный прогон Foundation

Кроме крошечного smoke-корпуса есть авторский учебный корпус `examples/foundation/`: русский и английский диалоги, Python, JavaScript/TypeScript и фреймворки, веб/HTTP, терминал Linux, Git, SQL/данные, обзор Go/Rust/Java/C#/C++ и протокол агента. Train и validation лежат в разных файлах; validation не используется для обучения весов или BPE-правил.

```bash
atlasai tokenizer train --config configs/foundation.toml
atlasai dataset inspect --config configs/foundation.toml
atlasai model train --config configs/foundation.toml
atlasai model evaluate --config configs/foundation.toml
atlasai llm generate --config configs/foundation.toml --prompt $'Задача: Что делает git status?\nОтвет:' --max-new-tokens 96
```

Фактический прогон в этой песочнице:

- модель: **11 532 288 параметров**, vocab 2048, context 256;
- корпус: 20 документов, 9 989 токенов, 29 train-блоков и 9 validation-блоков;
- 600 шагов, batch size 1, learning rate 0.0003, dropout 0.2;
- initial validation loss **7.732805**, лучший validation loss **6.103735** на шаге 500;
- время training loop около **160 s**, peak RSS около **980 MiB**;
- чекпоинт около **49 MB**.

Это настоящее обучение, но **не полноценное знание языков и фреймворков**. Корпус слишком мал: модель начинает запоминать тренировочные фрагменты, а ответы на validation и новые вопросы остаются ненадёжными и часто бессвязными. Прогон нужен как честная база для следующих этапов, а не как готовая универсальная LLM. На 1 ГБ RAM обучение ещё теснее, чем smoke; для практики лучше инференс готового чекпоинта или больше RAM.

## Локальная генерация

После обучения чекпоинта:

```bash
atlasai llm info --config configs/smoke.toml
atlasai llm generate --config configs/smoke.toml --prompt "Atlas AI —" --max-new-tokens 16 --temperature 0
```

`temperature=0` — greedy decoding. При `temperature>0` используется sampling с `top_k` и seed из конфига. Длинный prompt усекается до последних `context_length` токенов; количество токенов до усечения видно в `prompt_tokens`. Генерация останавливается по EOS или лимиту новых токенов.

Из Python:

```python
from atlasai.config import load_config
from atlasai.core.model.local import AtlasLocalModel

config = load_config("configs/smoke.toml")
backend = AtlasLocalModel(
    config.model.checkpoint_path,
    config.model.tokenizer_path,
    max_new_tokens=config.model.max_new_tokens,
    temperature=config.model.temperature,
    num_threads=config.runtime.num_threads,
    seed=config.runtime.seed,
)
result = backend.generate("Atlas AI —", max_new_tokens=16)
print(result.text)
print(backend.info())
```

Агентный слой должен зависеть от `LLM` из `atlasai.core.model.base`, а не напрямую от PyTorch. Отдельного HTTP-сервиса пока нет.

## Проверки

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pip check
atlasai doctor --config configs/atlas.toml
```

В полном окружении с PyTorch: **163 passed**. Дополнительно проверено окружение с заблокированным импортом torch: команды каркаса, токенизатора и датасета работают, ML-тесты явно пропускаются. Проверялись также сборка wheel, установка console script, строгая типизация и реальные CLI-команды обучения/оценки/генерации.

## Дорожная карта

1. **Готово:** структура проекта, конфиги, CLI, диагностика.
2. **Готово:** BPE-токенизатор и подготовка датасета.
3. **Готово:** Transformer и обучение маленькой модели.
4. **Готово:** интерфейс LLM и локальный бэкенд генерации, модель ≥10M параметров.
5. **Следующий:** базовый агент с циклом think → act → observe.
6. Инструменты и безопасность, включая **настоящий веб-поиск и чтение сайтов**.
7. Оркестратор, декомпозиция задач и подагенты.
8. Память, логирование и оценка качества.
