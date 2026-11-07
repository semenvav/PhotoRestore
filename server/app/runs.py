"""Запуски очереди на ПК (ТЗ §5.3) и выдача задач агенту.

Агент получает задачи только из активного `run`; активен максимум один. `scope=all` — все
локальные задачи в очереди, в том числе созданные после старта; `scope=selected` — только
задачи, бывшие в очереди у выбранных фото на момент старта. Запуск завершается сам, когда
в его области не осталось задач.
"""

from collections.abc import Collection
from dataclasses import dataclass
from datetime import timedelta
from statistics import median
from typing import Any, cast

from sqlalchemy import CursorResult, Select, case, func, select, update
from sqlalchemy.orm import Session

from app import states
from app.db import utcnow
from app.jobs import LOCAL_TYPES, STAGES
from app.models import Actor, Job, JobStatus, JobType, Photo, Run, RunScope, RunState
from app.states import log_event

CLAIM_CANDIDATES = 20


class RunError(Exception):
    pass


def active_run(session: Session) -> Run | None:
    return session.scalar(select(Run).where(Run.state == RunState.ACTIVE).limit(1))


def _queued(run: Run | None, job_types: Collection[JobType]) -> Select[Job]:
    query = select(Job).where(Job.status == JobStatus.QUEUED, Job.type.in_(job_types))
    if run is not None and run.scope == RunScope.SELECTED:
        query = query.where(Job.id.in_(run.job_ids or []))
    return query


def start(
    session: Session,
    job_types: Collection[JobType],
    photo_ids: Collection[int] | None = None,
    actor: Actor = Actor.OWNER,
) -> Run:
    """«Запустить всё» (photo_ids=None) или «Запустить выбранные»."""
    if (current := active_run(session)) is not None:
        raise RunError(f"Запуск #{current.id} уже идёт")
    query = _queued(None, job_types)
    if photo_ids is not None:
        query = query.where(Job.photo_id.in_(photo_ids))
    jobs = session.scalars(query.order_by(Job.id)).all()
    if not jobs:
        raise RunError(
            "В очереди нет задач для ПК"
            if photo_ids is None
            else "У выбранных фото нет задач в очереди на ПК"
        )
    run = Run(
        state=RunState.ACTIVE,
        scope=RunScope.ALL if photo_ids is None else RunScope.SELECTED,
        job_ids=None if photo_ids is None else [j.id for j in jobs],
    )
    session.add(run)
    session.flush()
    log_event(
        session,
        None,
        actor,
        "run_started",
        {"run_id": run.id, "scope": run.scope.value, "jobs": len(jobs)},
    )
    return run


def stop(session: Session, actor: Actor = Actor.OWNER) -> Run:
    """«Стоп»: новые задачи не выдаются, взятая дорабатывается."""
    run = active_run(session)
    if run is None:
        raise RunError("Запуска нет")
    run.state = RunState.STOPPED
    run.ended_at = utcnow()
    log_event(session, None, actor, "run_stopped", {"run_id": run.id})
    return run


@dataclass
class Progress:
    done: int
    failed: int
    running: int
    remaining: int

    @property
    def total(self) -> int:
        return self.done + self.failed + self.running + self.remaining


def progress(session: Session, run: Run, job_types: Collection[JobType]) -> Progress:
    counts = dict(
        session.execute(
            select(Job.status, func.count()).where(Job.run_id == run.id).group_by(Job.status)
        ).all()
    )
    remaining = 0
    if run.state == RunState.ACTIVE:
        remaining = (
            session.scalar(select(func.count()).select_from(_queued(run, job_types).subquery()))
            or 0
        )
    return Progress(
        done=counts.get(JobStatus.DONE, 0),
        failed=counts.get(JobStatus.FAILED, 0) + counts.get(JobStatus.CANCELLED, 0),
        running=counts.get(JobStatus.CLAIMED, 0),
        remaining=remaining,
    )


def maybe_complete(session: Session, run: Run, job_types: Collection[JobType]) -> bool:
    if run.state != RunState.ACTIVE:
        return False
    p = progress(session, run, job_types)
    if p.remaining or p.running:
        return False
    run.state = RunState.COMPLETED
    run.ended_at = utcnow()
    log_event(
        session,
        None,
        Actor.SYSTEM,
        "run_completed",
        {"run_id": run.id, "done": p.done, "failed": p.failed},
    )
    return True


def claim(
    session: Session,
    agent_id: str,
    prefer_type: JobType | None,
    lease_s: float,
    job_types: Collection[JobType],
) -> Job | None:
    """Выдаёт задачу из активного запуска или None. Сначала того же типа, что предыдущая
    (меньше перезагрузок моделей), дальше по приоритету и времени создания.

    Атомарность — условным `UPDATE … WHERE status='queued'`: из двух одновременных
    `claim` задачу получает тот, чей UPDATE изменил строку."""
    run = active_run(session)
    if run is None:
        return None
    types = [t for t in job_types if t in LOCAL_TYPES]
    order: list[Any] = []
    if prefer_type in types:
        order.append(case((Job.type == prefer_type, 0), else_=1))
    candidates = session.scalars(
        _queued(run, types)
        .with_only_columns(Job.id)
        .order_by(*order, Job.priority.desc(), Job.created_at, Job.id)
        .limit(CLAIM_CANDIDATES)
    ).all()

    now = utcnow()
    for job_id in candidates:
        result = cast(
            CursorResult[Any],
            session.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == JobStatus.QUEUED)
                .values(
                    status=JobStatus.CLAIMED,
                    agent_id=agent_id,
                    run_id=run.id,
                    lease_until=now + timedelta(seconds=lease_s),
                    progress=None,
                )
                .execution_options(synchronize_session=False)
            ),
        )
        if result.rowcount != 1:
            continue  # задачу только что взял другой claim
        job = session.get(Job, job_id)
        assert job is not None
        session.refresh(job)
        job.started_at = job.started_at or now
        photo = session.get(Photo, job.photo_id)
        assert photo is not None
        stage = STAGES[job.type]
        if photo.status != stage.queued:
            # фото ушло из очереди мимо задачи — такую задачу выполнять нельзя
            job.status = JobStatus.CANCELLED
            job.finished_at = now
            job.error = f"фото в статусе {photo.status}, а не {stage.queued}"
            continue
        states.transition(session, photo, stage.running, Actor.AGENT, {"job_id": job.id})
        return job

    maybe_complete(session, run, types)
    return None


def queue_summary(session: Session) -> dict[JobType, int]:
    rows = session.execute(
        select(Job.type, func.count())
        .where(Job.status == JobStatus.QUEUED, Job.type.in_(LOCAL_TYPES))
        .group_by(Job.type)
    ).all()
    return {t: n for t, n in rows}


def typical_duration_s(session: Session, job_type: JobType, sample: int = 20) -> float | None:
    """Медиана длительности последних выполненных задач типа — для оценки времени."""
    rows = session.execute(
        select(Job.started_at, Job.finished_at)
        .where(Job.type == job_type, Job.status == JobStatus.DONE, Job.started_at.is_not(None))
        .order_by(Job.finished_at.desc())
        .limit(sample)
    ).all()
    durations = [(f - s).total_seconds() for s, f in rows if s and f]
    return median(durations) if durations else None
