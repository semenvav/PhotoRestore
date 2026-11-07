"""API агента (ТЗ §6): `/api/agent/v1`, `Authorization: Bearer <AGENT_TOKEN>`, JSON.

Агент всегда инициатор, сервер на ПК не ходит. Задачи выдаются только из активного `run`.
"""

import hmac
import json
import logging
from collections.abc import Iterator
from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import jobs, results, runs
from app.config import Settings
from app.db import session_scope, utcnow
from app.models import Agent, Job, JobStatus, JobType
from app.storage import Store
from app.workflows import Registry

log = logging.getLogger(__name__)

PREFIX = "/api/agent/v1"
# Самый старый агент, с которым совместим этот сервер; старше — 426 «обновите агента»
MIN_AGENT_VERSION = (0, 2, 0)
HELLO_INTERVAL_S = 15
POLL_ACTIVE_S = 5


def parse_version(version: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(p) for p in version.split(".")[:3])
    except ValueError:
        return None


def agent_compatible(version: str | None) -> bool:
    parsed = parse_version(version or "")
    return parsed is not None and parsed >= MIN_AGENT_VERSION


def require_agent(request: Request) -> None:
    settings: Settings = request.app.state.settings
    if settings.agent_token is None:
        raise HTTPException(503, "AGENT_TOKEN не задан на сервере")
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    expected = settings.agent_token.get_secret_value().encode()
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip().encode(), expected):
        raise HTTPException(401, "неверный токен", headers={"WWW-Authenticate": "Bearer"})


router = APIRouter(prefix=PREFIX, dependencies=[Depends(require_agent)])


def get_session(request: Request) -> Iterator[Session]:
    with session_scope(request.app.state.factory) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def registry_of(request: Request) -> Registry:
    registry: Registry = request.app.state.registry
    return registry


def store_of(request: Request) -> Store:
    store: Store = request.app.state.store
    return store


def touch_agent(session: Session, agent_id: str) -> Agent:
    agent = session.get(Agent, agent_id)
    if agent is None:
        agent = Agent(id=agent_id)
        session.add(agent)
    agent.last_seen = utcnow()
    return agent


def load_job(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, f"задачи {job_id} нет")
    return job


def gone(job: Job) -> JSONResponse:
    """Задача больше не у агента: отменена, возвращена в очередь по аренде или закрыта."""
    return JSONResponse({"error": f"задача {job.id} в статусе {job.status}"}, status_code=409)


# --- пульс и выдача ---------------------------------------------------------------------


class HelloIn(BaseModel):
    agent_id: str = Field(min_length=1, max_length=64)
    version: str = Field(max_length=40)
    name: str | None = Field(default=None, max_length=100)
    comfy_ok: bool = False
    gpu_name: str | None = Field(default=None, max_length=100)
    vram_free_mb: int | None = None


@router.post("/hello")
def hello(body: HelloIn, request: Request, session: SessionDep) -> Response:
    agent = touch_agent(session, body.agent_id)
    agent.version = body.version
    agent.name = body.name
    agent.comfy_ok = body.comfy_ok
    agent.gpu_name = body.gpu_name
    agent.vram_free_mb = body.vram_free_mb
    if not agent_compatible(body.version):
        need = ".".join(map(str, MIN_AGENT_VERSION))
        return JSONResponse(
            {"error": f"агент {body.version} несовместим с сервером, нужен ≥ {need}"},
            status_code=426,
        )
    run_active = runs.active_run(session) is not None
    return JSONResponse(
        {
            "run_active": run_active,
            "poll_interval_s": POLL_ACTIVE_S if run_active else HELLO_INTERVAL_S,
        }
    )


class ClaimIn(BaseModel):
    agent_id: str = Field(min_length=1, max_length=64)
    prefer_type: JobType | None = None


def job_for_agent(job: Job, registry: Registry) -> dict[str, Any]:
    workflow = registry.for_type(job.type)
    assert workflow is not None  # claim выдаёт только типы из реестра
    payload = job.payload
    values = {**(payload.get("params") or {}), "prompt": payload.get("prompt")}
    return {
        "id": job.id,
        "type": job.type.value,
        "photo_id": job.photo_id,
        "inputs": [{"name": "source", "url": f"{PREFIX}/jobs/{job.id}/inputs/source"}],
        "workflow": workflow.build(values),
        "bindings": workflow.agent_bindings(),
        "outputs": [
            {"index": i, "seed": seed} for i, seed in enumerate(payload.get("seeds") or [None])
        ],
        "lease_until": job.lease_until.isoformat().replace("+00:00", "Z")
        if job.lease_until
        else None,
    }


@router.post("/claim")
def claim(body: ClaimIn, request: Request, session: SessionDep) -> Response:
    touch_agent(session, body.agent_id)
    registry = registry_of(request)
    settings: Settings = request.app.state.settings
    job = runs.claim(
        session, body.agent_id, body.prefer_type, settings.lease_s, registry.job_types()
    )
    if job is None:
        return Response(status_code=204)
    workflow = registry.for_type(job.type)
    assert workflow is not None
    # версия workflow на момент выдачи — её запишем в варианты
    job.payload = {**job.payload, "workflow": workflow.describe()}
    log.info("claim: задача %s (%s) → %s", job.id, job.type, body.agent_id)
    return JSONResponse(job_for_agent(job, registry))


# --- выполнение задачи ------------------------------------------------------------------


@router.get("/jobs/{job_id}/inputs/{name}")
def job_input(job_id: int, name: str, request: Request, session: SessionDep) -> Response:
    job = load_job(session, job_id)
    if job.status != JobStatus.CLAIMED:
        return gone(job)
    try:
        path = results.input_path(session, store_of(request), job, name)
    except results.ResultError as e:
        raise HTTPException(e.status_code, str(e)) from e
    if not path.is_file():
        raise HTTPException(404, f"нет файла входа {name!r}")
    return FileResponse(path, media_type="image/png", filename=f"job{job.id}_{name}.png")


class ProgressIn(BaseModel):
    done: int = 0
    total: int = 0
    stage: str | None = Field(default=None, max_length=200)


@router.post("/jobs/{job_id}/progress")
def job_progress(
    job_id: int, body: ProgressIn, request: Request, session: SessionDep
) -> dict[str, bool]:
    job = load_job(session, job_id)
    if job.status != JobStatus.CLAIMED:
        return {"cancel": True}  # отменена или аренда истекла — агенту её больше не трогать
    settings: Settings = request.app.state.settings
    job.lease_until = utcnow() + _lease(settings)
    job.progress = body.model_dump()
    if job.agent_id:
        touch_agent(session, job.agent_id)
    return {"cancel": False}


def _lease(settings: Settings) -> timedelta:
    return timedelta(seconds=settings.lease_s)


@router.post("/jobs/{job_id}/outputs/{index}")
def job_output(
    job_id: int,
    index: int,
    request: Request,
    session: SessionDep,
    file: Annotated[UploadFile, File()],
    meta: Annotated[str, Form()] = "{}",
) -> Response:
    job = load_job(session, job_id)
    if job.status == JobStatus.DONE:
        return JSONResponse({"ok": True, "ignored": "задача уже закрыта"})
    if job.status != JobStatus.CLAIMED:
        return gone(job)
    try:
        meta_dict = json.loads(meta)
        if not isinstance(meta_dict, dict):
            raise ValueError("meta должен быть объектом")
    except ValueError as e:
        raise HTTPException(400, f"meta: {e}") from e
    try:
        results.save_output(store_of(request), job, index, file.file.read(), meta_dict)
    except results.ResultError as e:
        raise HTTPException(e.status_code, str(e)) from e
    settings: Settings = request.app.state.settings
    job.lease_until = utcnow() + _lease(settings)
    return JSONResponse({"ok": True})


@router.post("/jobs/{job_id}/complete")
def job_complete(job_id: int, request: Request, session: SessionDep) -> Response:
    job = load_job(session, job_id)
    if job.status == JobStatus.DONE:
        return JSONResponse({"ok": True})  # повторный complete
    if job.status != JobStatus.CLAIMED:
        return gone(job)
    store = store_of(request)
    try:
        variants = results.complete(session, store, registry_of(request), job)
    except results.ResultError as e:
        raise HTTPException(e.status_code, str(e)) from e
    session.commit()
    results.cleanup(store, job)
    run = runs.active_run(session)
    if run is not None and runs.maybe_complete(session, run, registry_of(request).job_types()):
        session.commit()
    log.info("complete: задача %s → %s", job.id, [v.label for v in variants])
    return JSONResponse({"ok": True, "variants": [v.id for v in variants]})


class FailIn(BaseModel):
    error: str = Field(max_length=4000)
    retryable: bool = True


@router.post("/jobs/{job_id}/fail")
def job_fail(job_id: int, body: FailIn, request: Request, session: SessionDep) -> Response:
    job = load_job(session, job_id)
    if job.status != JobStatus.CLAIMED:
        return JSONResponse({"ok": True, "ignored": f"задача в статусе {job.status}"})
    jobs.fail(session, job, body.error, body.retryable)
    results.cleanup(store_of(request), job)
    log.warning("fail: задача %s (%s): %s", job.id, job.status, body.error)
    return JSONResponse({"ok": True, "status": job.status.value})
