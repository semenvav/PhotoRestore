"""Задачи для агента и Gemini: создание и отмена. Выдача агенту — в M2 (протокол §6)."""

import secrets

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import presets
from app.db import utcnow
from app.models import Actor, Job, JobStatus, JobType, Photo
from app.states import log_event

SEED_LIMIT = 2**48


def random_seed() -> int:
    return secrets.randbelow(SEED_LIMIT)


def create_edit_job(
    session: Session,
    photo: Photo,
    actor: Actor,
    prompt: str | None = None,
    variants: int | None = None,
    source: str = "prepared",
) -> Job:
    """Задача `edit_local`: N вариантов со случайными seed. Ничего не запускает —
    агент возьмёт её только в рамках активного `run`."""
    if variants is None:
        variants = int(presets.get_setting(session, "edit")["variants"])
    job = Job(
        photo_id=photo.id,
        type=JobType.EDIT_LOCAL,
        status=JobStatus.QUEUED,
        payload={
            "prompt": prompt or presets.base_prompt(session),
            "seeds": [random_seed() for _ in range(variants)],
            "source": source,
            "round": photo.round,
        },
    )
    session.add(job)
    session.flush()
    log_event(session, photo.id, actor, "job_created", {"job_id": job.id, "type": job.type.value})
    return job


def cancel_queued(session: Session, photo_id: int, actor: Actor) -> list[int]:
    """Отменяет ещё не взятые задачи фото (для «Отложить»)."""
    jobs = session.scalars(
        select(Job).where(Job.photo_id == photo_id, Job.status == JobStatus.QUEUED)
    ).all()
    now = utcnow()
    for job in jobs:
        job.status = JobStatus.CANCELLED
        job.finished_at = now
    if jobs:
        log_event(session, photo_id, actor, "jobs_cancelled", {"job_ids": [j.id for j in jobs]})
    return [j.id for j in jobs]
