"""Панель «ПК» (ТЗ §5.3): агент, очередь, запуски; отмена и повтор задач."""

from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import jobs, runs
from app.api.agent import MIN_AGENT_VERSION, agent_compatible
from app.api.ui import (
    SessionDep,
    agent_online,
    current_agent,
    header_ctx,
    load_photo,
    render,
    safe_back,
    with_msg,
)
from app.models import Job, JobStatus
from app.workflows import Registry

router = APIRouter()

RECENT_JOBS = 15


def pc_ctx(session: Session, registry: Registry) -> dict[str, Any]:
    agent = current_agent(session)
    queue = runs.queue_summary(session)
    queue_rows = []
    for job_type in sorted(jobs.LOCAL_TYPES):
        count = queue.get(job_type, 0)
        if not count and job_type not in registry.job_types():
            continue
        typical = runs.typical_duration_s(session, job_type)
        queue_rows.append(
            {
                "type": job_type,
                "count": count,
                "typical_s": typical,
                "estimate_s": typical * count if typical and count else None,
                "has_workflow": job_type in registry.job_types(),
            }
        )
    run = runs.active_run(session)
    active = session.scalars(
        select(Job)
        .where(Job.status.in_(jobs.ACTIVE), Job.type.in_(jobs.LOCAL_TYPES))
        .order_by(Job.status.desc(), Job.priority.desc(), Job.id)
        .limit(50)
    ).all()
    recent = session.scalars(
        select(Job)
        .where(Job.status.not_in(jobs.ACTIVE), Job.type.in_(jobs.LOCAL_TYPES))
        .order_by(Job.finished_at.desc(), Job.id.desc())
        .limit(RECENT_JOBS)
    ).all()
    return {
        "agent": agent,
        "online": agent_online(agent),
        "outdated": agent is not None and not agent_compatible(agent.version),
        "min_version": ".".join(map(str, MIN_AGENT_VERSION)),
        "queue_rows": queue_rows,
        "queued_total": sum(r["count"] for r in queue_rows if r["has_workflow"]),
        "run": run,
        "progress": runs.progress(session, run, registry.job_types()) if run else None,
        "active_jobs": active,
        "recent_jobs": recent,
    }


@router.get("/pc")
def pc_page(request: Request, session: SessionDep, msg: str | None = None) -> Response:
    ctx = pc_ctx(session, request.app.state.registry)
    return render(request, "pc.html", {**ctx, "hdr": header_ctx(session), "msg": msg})


@router.get("/partials/pc")
def pc_partial(request: Request, session: SessionDep) -> Response:
    return render(request, "_pc_body.html", pc_ctx(session, request.app.state.registry))


@router.post("/runs/start")
def run_start(
    request: Request, session: SessionDep, back: Annotated[str, Form()] = "/pc"
) -> Response:
    try:
        run = runs.start(session, request.app.state.registry.job_types())
        session.commit()
        msg = f"Запуск #{run.id} начат — агент берёт задачи"
    except runs.RunError as e:
        msg = str(e)
    return RedirectResponse(with_msg(safe_back(back), msg), status_code=303)


@router.post("/runs/stop")
def run_stop(session: SessionDep, back: Annotated[str, Form()] = "/pc") -> Response:
    try:
        run = runs.stop(session)
        session.commit()
        msg = f"Запуск #{run.id} остановлен: новые задачи не выдаются, текущая доработается"
    except runs.RunError as e:
        msg = str(e)
    return RedirectResponse(with_msg(safe_back(back), msg), status_code=303)


@router.post("/jobs/{job_id}/cancel")
def job_cancel(session: SessionDep, job_id: int, back: Annotated[str, Form()] = "") -> Response:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Задача не найдена")
    try:
        was_running = job.status == JobStatus.CLAIMED
        jobs.cancel(session, job)
        session.commit()
        msg = f"Задача #{job.id} отменена" + (" — агент прервёт её" if was_running else "")
    except jobs.JobError as e:
        msg = str(e)
    return RedirectResponse(
        with_msg(safe_back(back) if back else f"/photo/{job.photo_id}", msg), status_code=303
    )


@router.post("/photo/{photo_id}/retry")
def photo_retry(session: SessionDep, photo_id: int) -> Response:
    photo = load_photo(session, photo_id)
    try:
        job = jobs.retry(session, photo)
        session.commit()
        msg = f"Задача #{job.id} в очереди — нужен «Запустить»"
    except jobs.JobError as e:
        msg = str(e)
    return RedirectResponse(with_msg(f"/photo/{photo_id}", msg), status_code=303)
