# Учебный корпус Atlas Foundation

Это **авторский маленький корпус**, а не выгрузка интернета и не полноценный pretraining-датасет. Цель — проверить, что собственная 10M+ модель может обучаться на основах общения, кода и терминала, и честно измерить пределы такого обучения.

## Покрытые темы

- `russian`, `english` — запросы, ответы, перевод, уточнения, границы знания.
- `python` — синтаксис, типы, функции, файлы, ошибки, venv, pytest, asyncio.
- `javascript` — JavaScript/TypeScript, npm, React, Vue, Express, Next.js, async.
- `web` — HTML/CSS, HTTP, REST, JSON, curl, CORS, безопасность.
- `terminal` — Linux-команды, потоки, права, процессы, коды завершения, безопасность.
- `git` — status/diff/add/commit/push, PR, ветки, конфликты, артефакты.
- `data_sql` — SQL, JOIN, индексы, транзакции, параметры, JSON/CSV.
- `systems` — Go, Rust, Java/JVM, C#/.NET, C++, выбор языка.
- `agent_protocol` — один JSON-объект, действия `think`/`respond`, отказ от выдуманного выполнения.

## Разделение

`train/` используется для обучения токенизатора и модели. `val/` содержит **другие формулировки и примеры** и не используется для обучения весов или merge-правил. При явных validation-путях train и val образуют отдельные потоки токенов, поэтому validation-токены не попадают в обучение.

Это лучше, чем случайное перемешивание окон одного текста, но всё равно не является независимым industry benchmark: темы в train и val пересекаются намеренно.

## Формат

Обычный UTF-8 `.txt`. Примеры имеют вид:

```text
Задача: ...
Ответ: {"action":"respond","text":"..."}
```

Для короткой заметки агента:

```text
Задача: ...
Ответ: {"action":"think","text":"..."}
Наблюдение: Заметка записана. Инструменты не вызывались.
Ответ: {"action":"respond","text":"..."}
```

Команды в корпусе — учебные объяснения. Они не выполняются при загрузке датасета.

## Запуск

```bash
atlasai tokenizer train --config configs/foundation.toml
atlasai dataset inspect --config configs/foundation.toml
atlasai model train --config configs/foundation.toml
atlasai model evaluate --config configs/foundation.toml
atlasai llm generate --config configs/foundation.toml --prompt $'Задача: Что делает git status?\nОтвет:' --max-new-tokens 96
```

Артефакты пишутся в ignored `checkpoints/`: `foundation-tokenizer.json`, `foundation.pt`, `foundation.metrics.json`.
