# photo-restore

Сервис реставрации семейного фотоархива: сервер (`server/`, FastAPI в Docker) + агент на ПК
(`agent/`, с M2). Полное ТЗ — `docs/SPEC.md`, этапы — §11. ТЗ есть только локально (в
`.gitignore`): в нём детали домашней сети.

Репозиторий публичный: в коммитах не должно быть хостов, адресов, путей и имён устройств
домашней сети — только плейсхолдеры и переменные (`RESTORE_HOST` и т. п.).

## Команды

```sh
uv sync                                   # зависимости (workspace: корень + server/)
uv run pytest -q                          # тесты
uv run ruff format server tests && uv run ruff check server tests
uv run mypy                               # типы (server/app)
# локальный сервер: скопируйте .env.example в .env
uv run uvicorn --factory app.main:create_app --reload --app-dir server
# новая миграция после правки models.py:
DB_PATH=var/gen.db uv run alembic -c server/alembic.ini upgrade head
DB_PATH=var/gen.db uv run alembic -c server/alembic.ini revision --autogenerate -m "..."
```

Сборка образа: `docker build -f deploy/Dockerfile .`; пример стека — `deploy/compose.yaml`.

## Версии и CI

- `.github/workflows/ci.yml`: каждый push — ruff, mypy, pytest; прошедший push публикует образ
  `ghcr.io/semenvav/photo-restore` (теги `latest` = ветка по умолчанию, `<ветка>`, `sha-…`).
- Номер сборки: серия из файла `VERSION` = номер этапа (`0.1` — M1, `0.2` — M2…). Первая сборка
  ветки по умолчанию в серии получает номер `0.1`, следующие — `0.1.1`, `0.1.2`… Номер ставится
  git-тегом `v…` после публикации образа и передаётся в образ как `APP_VERSION`
  (виден в шапке и в `/healthz`).
- **Начиная новый этап, первым делом поменять `VERSION`** (M2 → `0.2`).

## Инварианты

- `original.*` никогда не открывается на запись после приёма (файл в `store/` с правами 0444).
- Статус фото меняется только через `states.transition()`; события истории — `states.log_event()`.
- Агент не получает задач без активного `run`.
- Все пути внутри `store/` строятся из ID (`storage.Store`), имена от пользователя в пути не попадают.

## Решения, которых нет в ТЗ

- Файлы во «Входящих» после приёма остаются на месте; таблица `inbox_file` помнит
  (путь, размер, mtime_ns) принятых файлов. Новый файл по тому же пути — новое фото.
- `photo.sha256` уникален частичным индексом `WHERE duplicate_of IS NULL`: у дубля тот же хеш.
- Приём копирует файл в `store/.staging/`, считая sha256 по записанным байтам, проверяет его
  Pillow и только потом переносит в `store/R-xxxx/original.<ext>`. Если каталог для нового ID
  уже существует (сбой до commit, БД из бэкапа), он переносится в `store/.orphans/`.
- `photo.width/height` — размеры после EXIF-транспонирования.
- Кроп: `crop` задаётся как `cropper.getData()` (cropper.js 1.x) — в пикселях оригинала внутри
  описанного прямоугольника изображения, повёрнутого на `rotate90·90° + angle` по часовой.
  Сервер повторяет эту геометрию одним аффинным преобразованием (`imaging.prepare`),
  углы вне изображения заливаются белым. «Без обрезки» = `crop: null`, поворот сохраняется.
- Время в БД — UTC; показывается в `DISPLAY_TZ`.
- Форматы: jpg, png, webp, tif, avif (`storage.ALLOWED_EXT`). AVIF читает Pillow (libavif в
  wheel); его irot/imir Pillow отдаёт как EXIF-ориентацию. HEIC не поддерживается.
- Удаление фото (`app/deletion.py`, по просьбе владельца, вопреки «не удаляется» из ТЗ — но
  без потери файлов): из БД уходят фото, варианты, задачи и история, в общей истории остаётся
  событие `deleted`; каталог `store/R-xxxx` переносится в `store/.deleted/`, чистится вручную.
  `inbox_file` остаётся без `photo_id`, поэтому файл во «Входящих» повторно не принимается.
  Фото в обработке (`EDITING`, `UPSCALING`, `GEMINI`) не удаляются. Дубли удалённого
  основного остаются: основным становится первый, остальные ссылаются на него. ID не
  переиспользуются (`sqlite_autoincrement`). В UI удаление — только через диалог подтверждения.

## Устройство сервера

- `app/main.py` — `create_app()`, lifespan: миграции, каталоги, настройки по умолчанию, цикл приёма.
- `app/intake.py` — опрос `inbox/`; `app/imaging.py` — Pillow; `app/prep.py` — подготовка.
- `app/api/ui.py` — Jinja2 + htmx + Alpine; JS-библиотеки лежат в `app/static/vendor/` (без CDN).
- Тесты: `tests/`; картинки генерируются в `tests/imgutil.py`.
