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
from sqlalchemy.orm import Session, sessionmaker

from app import jobs, presets, runs
from app.api import agent, pc, review, ui
from app.config import Settings, get_settings
from app.db import make_engine, make_session_factory, migrate, session_scope
from app.intake import Intake
from app.storage import Store
from app.views import make_templates
from app.workflows import Registry, load_registry

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


def sweep(factory: sessionmaker[Session], registry: Registry) -> None:
    """Просроченные аренды → обратно в очередь; запуск без задач → завершён."""
    with session_scope(factory) as s:
        expired = jobs.expire_leases(s)
        if expired:
            log.warning("аренда истекла: задачи %s возвращены в очередь", expired)
        if (run := runs.active_run(s)) is not None:
            runs.maybe_complete(s, run, registry.job_types())


async def sweep_loop(factory: sessionmaker[Session], registry: Registry, interval: float) -> None:
    while True:
        try:
            await asyncio.to_thread(sweep, factory, registry)
        except Exception:
            log.exception("sweep упал")
        await asyncio.sleep(interval)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

    # битая привязка workflow — сервер не стартует (ТЗ §7.3)
    registry = load_registry(settings.workflows_dir)
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
        tasks = [asyncio.create_task(sweep_loop(factory, registry, settings.sweep_interval_s))]
        if settings.intake_enabled:
            tasks.append(
                asyncio.create_task(intake_loop(intake, settings.intake_poll_s, app.state.wake))
            )
        if settings.agent_token is None:
            log.warning("AGENT_TOKEN не задан: API агента отвечает 503")
        log.info(
            "photo-restore %s запущен: data=%s db=%s",
            settings.app_version,
            settings.data_dir,
            settings.db_path,
        )
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            engine.dispose()

    app = FastAPI(title="photo-restore", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.factory = factory
    app.state.store = store
    app.state.intake = intake
    app.state.registry = registry
    app.state.templates = make_templates(settings.display_tz, settings.app_version)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    # pc и review раньше ui: у ui есть общий POST /photo/{id}/{action}
    app.include_router(pc.router)
    app.include_router(review.router)
    app.include_router(ui.router)
    app.include_router(agent.router)
    return app
