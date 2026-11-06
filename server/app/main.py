"""Точка входа: `uvicorn --factory app.main:create_app`."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app import presets
from app.api import ui
from app.config import Settings, get_settings
from app.db import make_engine, make_session_factory, migrate, session_scope
from app.intake import Intake
from app.storage import Store
from app.views import make_templates

STATIC_DIR = Path(__file__).resolve().parent / "static"

log = logging.getLogger("app")


async def intake_loop(intake: Intake, interval: float, wake: asyncio.Event) -> None:
    while True:
        try:
            await asyncio.to_thread(intake.poll)
        except Exception:
            log.exception("intake: опрос упал")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(wake.wait(), timeout=interval)
        wake.clear()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

    engine = make_engine(settings.db_url)
    factory = make_session_factory(engine)
    store = Store(settings.store_dir)
    intake = Intake(settings, factory, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(migrate, settings.db_url)
        intake.ensure_dirs()
        with session_scope(factory) as s:
            presets.ensure_defaults(s)

        app.state.wake = asyncio.Event()
        task = None
        if settings.intake_enabled:
            task = asyncio.create_task(intake_loop(intake, settings.intake_poll_s, app.state.wake))
        log.info(
            "photo-restore %s запущен: data=%s db=%s",
            settings.app_version,
            settings.data_dir,
            settings.db_path,
        )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            engine.dispose()

    app = FastAPI(title="photo-restore", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.factory = factory
    app.state.store = store
    app.state.intake = intake
    app.state.templates = make_templates(settings.display_tz, settings.app_version)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(ui.router)
    return app
