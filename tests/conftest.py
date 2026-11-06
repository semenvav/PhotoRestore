from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.intake import Intake
from app.main import create_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db" / "restore.db",
        intake_enabled=False,
        display_tz="UTC",
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:  # lifespan: миграции, каталоги, настройки по умолчанию
        yield c


@pytest.fixture
def factory(app: FastAPI, client: TestClient) -> sessionmaker[Session]:
    return app.state.factory  # type: ignore[no-any-return]


@pytest.fixture
def intake(app: FastAPI, client: TestClient) -> Intake:
    return app.state.intake  # type: ignore[no-any-return]


@pytest.fixture
def inbox(settings: Settings, client: TestClient) -> Path:
    return settings.inbox_dir
