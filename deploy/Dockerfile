# syntax=docker/dockerfile:1
# Сборка из корня репозитория: docker build -f deploy/Dockerfile .
# Готовые образы собирает CI (.github/workflows/ci.yml) и публикует в ghcr.io.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /srv

# Зависимости — отдельным слоем, чтобы правки кода не пересобирали их
COPY pyproject.toml uv.lock ./
COPY server/pyproject.toml server/pyproject.toml
# агент в образ не ставится, но его pyproject нужен uv, чтобы разобрать workspace
COPY agent/pyproject.toml agent/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package photo-restore-server --no-install-workspace

COPY server/ server/
COPY workflows/ workflows/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package photo-restore-server

# По умолчанию процесс работает как 1000:1000; владельца файлов на датасете задаёт compose
# (`user: PUID:PGID`). /db открыт на запись любому UID: именованный том при создании берёт
# права из образа, так что база создастся под любым пользователем.
RUN groupadd -g 1000 restore \
    && useradd -u 1000 -g 1000 -M -d /srv -s /usr/sbin/nologin restore \
    && mkdir -p /data /db \
    && chmod 1777 /db

USER 1000:1000
WORKDIR /srv/server
ENV DATA_DIR=/data \
    DB_PATH=/db/restore.db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]

# Номер сборки — последним слоем: новый номер не пересобирает ничего выше
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

CMD ["uvicorn", "--factory", "app.main:create_app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-access-log"]
