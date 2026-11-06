from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import create_engine, pool

import app.models  # noqa: F401  регистрирует таблицы в Base.metadata
from app.config import get_settings
from app.db import Base, UTCDateTime

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    # В БД это обычный DATETIME; декоратор нужен только приложению
    if type_ == "type" and isinstance(obj, UTCDateTime):
        return "sa.DateTime()"
    return False


def _configure(**kwargs: Any) -> None:
    context.configure(
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
        render_item=render_item,
        **kwargs,
    )


def url() -> str:
    return config.get_main_option("sqlalchemy.url") or get_settings().db_url


if context.is_offline_mode():
    _configure(url=url(), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
