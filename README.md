# Ассистент с долговременной памятью предпочтений

FastAPI-сервис передаёт сообщения LLM-агенту, который самостоятельно читает и обновляет
предпочтения пользователя. Данные хранятся в SQLite и сохраняются между перезапусками.

## Стек

- Python 3.11+, FastAPI, Pydantic
- LangChain `create_agent` и LangGraph
- Ollama с локальной моделью `qwen3:14b`
- SQLite
- pytest, HTTPX и Ruff

## Архитектура

```text
POST /chat
    │  user_id + message
    ▼
FastAPI ── context(user_id, storage) ──► LangGraph agent
                                             │
                                LLM выбирает │ инструмент
                          ┌──────────────────┴──────────────────┐
                          ▼                                     ▼
                 get_preferences                        save_preference
                          └──────────────────┬──────────────────┘
                                             ▼
                                  SQLite preferences.db
                               PK (user_id, category)
```

`create_agent` выполняет цикл «модель → инструмент → модель» до финального ответа.
`get_preferences` читает все предпочтения текущего пользователя, а `save_preference`
атомарно создаёт или обновляет одну категорию. `user_id` передаётся через серверный
`ToolRuntime` и отсутствует в доступной модели схеме аргументов инструментов.

SQLite использует первичный ключ `(user_id, category)` и `UPSERT`: новое значение `drink`
заменяет старое без дубля, а независимая категория `color` сохраняется. Семантически
равнозначную категорию выбирает LLM после чтения существующей памяти.

## Установка и запуск в Windows PowerShell

Требуются Python 3.11+ и установленный [Ollama](https://ollama.com/download/windows).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
ollama pull qwen3:14b
```

Если активация запрещена политикой PowerShell, запускайте команды через
`.\.venv\Scripts\python.exe`.

Настройки `.env`:

```dotenv
OLLAMA_MODEL=qwen3:14b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT_SECONDS=120
SQLITE_PATH=data/preferences.db
AGENT_RECURSION_LIMIT=12
```

Для Qwen3 используется `reasoning=False`: скрытые рассуждения не возвращаются через API,
при этом вызовы инструментов остаются доступны. Запуск сервиса:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Примеры API

Сохранение предпочтения:

```powershell
$body = @{ user_id = "alice"; message = "Я предпочитаю чай" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" -Body $body
```

Чтение или обновление выполняется через тот же endpoint:

```powershell
$body = @{ user_id = "alice"; message = "Теперь предпочитаю кофе" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" -Body $body

$body = @{ user_id = "alice"; message = "Какой напиток я предпочитаю?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" -Body $body
```

Формат ответа: `{"response": "Вы предпочитаете кофе."}`

## Тесты

Автоматические тесты используют детерминированную подставную `BaseChatModel` и не требуют
Ollama, интернета или реальной LLM. Настоящий LangGraph-цикл выполняет tool calls, а тесты
проверяют HTTP-ответы, `ToolMessage`, изоляцию пользователей и состояние SQLite.

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

## Решения и ограничения

- Память содержит только предпочтения; история диалога и checkpointer не хранятся.
- Категории ограничены `snake_case`, но их семантическое сопоставление зависит от модели.
- SQLite работает в WAL-режиме; запросы параметризованы, для операции создаётся соединение.
- Ошибки SQLite, Ollama и инструментов преобразуются в безопасные HTTP-ответы без деталей.
- `user_id` логически разделяет данные, но не заменяет аутентификацию.
- SQLite рассчитана на один небольшой сервис, а запросы не имеют идемпотентного ключа.
- Docker, авторизация, RAG, Redis, мониторинг и CI/CD намеренно не добавлены.

## Фактическая проверка

- `pytest -q`: **31 passed**.
- Ruff lint и format check: **пройдены**.
- Ручная интеграция с Ollama 0.40.18 и `qwen3:14b` прошла 9 сценариев, включая обновление,
  независимые категории, изоляцию пользователей и чтение SQLite после нового процесса.
- Фактически потраченное время: ______ (заполняется кандидатом).
