"""Задачи для агента и Gemini: создание, возврат в очередь, неудача, отмена, повтор.

Выдача агенту — `runs.claim`, приём результатов — `results`.
"""

import math
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import presets, states
from app.db import utcnow
from app.models import Actor, Job, JobStatus, JobType, Photo, Status, Variant
from app.states import log_event

SEED_LIMIT = 2**48
ACTIVE = (JobStatus.QUEUED, JobStatus.CLAIMED)
# «Переделать апскейл»: масштабы на выбор (×4 от 1,5 Мп — 25 Мп, на 3090 Ti ≈ 40 с)
UPSCALE_SCALES = (1.5, 2.0, 3.0, 4.0)


@dataclass(frozen=True)
class Stage:
    """Статусы фото вокруг задачи: ждёт в очереди → выполняется → результат готов."""

    queued: Status
    running: Status
    done: Status


STAGES = {
    JobType.EDIT_LOCAL: Stage(Status.QUEUED_EDIT, Status.EDITING, Status.REVIEW),
    JobType.UPSCALE_LOCAL: Stage(Status.QUEUED_UPSCALE, Status.UPSCALING, Status.FINAL_REVIEW),
}
LOCAL_TYPES = frozenset(STAGES)


def random_seed() -> int:
    return secrets.randbelow(SEED_LIMIT)


def create_edit_job(
    session: Session,
    photo: Photo,
    actor: Actor,
    prompt: str | None = None,
    variants: int | None = None,
    source_variant_id: int | None = None,
) -> Job:
    """Задача `edit_local`: N вариантов со случайными seed. Ничего не запускает —
    агент возьмёт её только в рамках активного `run`."""
    edit = presets.get_setting(session, "edit")
    if variants is None:
        variants = int(edit["variants"])
    job = Job(
        photo_id=photo.id,
        type=JobType.EDIT_LOCAL,
        status=JobStatus.QUEUED,
        payload={
            "prompt": prompt or presets.base_prompt(session),
            "seeds": [random_seed() for _ in range(variants)],
            "source": "variant" if source_variant_id else "prepared",
            "source_variant_id": source_variant_id,
            "round": photo.round,
            "params": {"steps": edit["steps"], "megapixels": edit["megapixels"]},
        },
    )
    session.add(job)
    session.flush()
    log_event(session, photo.id, actor, "job_created", {"job_id": job.id, "type": job.type.value})
    return job


def auto_scale(width: int, height: int, target_mp: float, max_scale: float) -> float:
    """Масштаб апскейла по умолчанию: до `target_mp` мегапикселей, но не больше
    ×`max_scale` от входа и не меньше ×1 (большой вход не уменьшается)."""
    scale = math.sqrt(target_mp * 1_000_000 / (width * height))
    return round(min(max_scale, max(1.0, scale)), 2)


def create_upscale_job(
    session: Session,
    photo: Photo,
    variant: Variant,
    actor: Actor,
    model: str | None = None,
    scale: float | None = None,
) -> Job:
    """Задача `upscale_local` для варианта; без явных модели и масштаба — из настроек
    «upscale» (масштаб — `auto_scale` по размеру варианта)."""
    up = presets.get_setting(session, "upscale")
    if scale is None:
        scale = auto_scale(
            variant.width, variant.height, float(up["target_mp"]), float(up["max_scale"])
        )
    job = Job(
        photo_id=photo.id,
        type=JobType.UPSCALE_LOCAL,
        status=JobStatus.QUEUED,
        payload={
            "seeds": [random_seed()],
            "source": "variant",
            "source_variant_id": variant.id,
            "round": photo.round,
            "params": {"model": model or up["model"], "scale": scale},
        },
    )
    session.add(job)
    session.flush()
    log_event(session, photo.id, actor, "job_created", {"job_id": job.id, "type": job.type.value})
    return job


def outputs_count(job: Job) -> int:
    return len(job.payload.get("seeds") or [None])


def active_job(session: Session, photo_id: int) -> Job | None:
    return session.scalar(
        select(Job).where(Job.photo_id == photo_id, Job.status.in_(ACTIVE)).order_by(Job.id)
    )


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


def _release(job: Job) -> None:
    job.agent_id = None
    job.lease_until = None


def _back_to_queue(session: Session, job: Job, actor: Actor, data: dict[str, Any]) -> Photo:
    """Фото выполняемой задачи возвращается в статус ожидания своей стадии."""
    photo = session.get(Photo, job.photo_id)
    assert photo is not None
    stage = STAGES[job.type]
    if photo.status == stage.running:
        states.transition(session, photo, stage.queued, actor, {"job_id": job.id, **data})
    return photo


def _flag(photo: Photo, reason: str) -> None:
    photo.attention = True
    photo.attention_reason = reason
    photo.updated_at = utcnow()


def fail(session: Session, job: Job, error: str, retryable: bool) -> None:
    """Неудача выполняемой задачи (ошибка агента или истёкшая аренда).

    Повторяемая — задача снова в очереди, пока не исчерпаны попытки; иначе `failed`,
    а фото ждёт в очереди с флагом `attention` и кнопкой «Повторить».
    """
    job.attempts += 1
    _release(job)
    job.progress = None
    job.error = error
    final = not retryable or job.attempts >= job.max_attempts
    photo = _back_to_queue(
        session, job, Actor.SYSTEM, {"error": error, "attempt": job.attempts, "final": final}
    )
    if final:
        job.status = JobStatus.FAILED
        job.finished_at = utcnow()
        _flag(photo, f"Задача #{job.id} не выполнена: {error}")
    else:
        job.status = JobStatus.QUEUED


def expire_leases(session: Session, now: datetime | None = None) -> list[int]:
    """Взятые задачи с истёкшей арендой — обратно в очередь (агент пропал)."""
    now = now or utcnow()
    expired = session.scalars(
        select(Job).where(Job.status == JobStatus.CLAIMED, Job.lease_until < now)
    ).all()
    for job in expired:
        fail(session, job, "агент не отвечал дольше срока аренды", retryable=True)
    return [j.id for j in expired]


class JobError(Exception):
    pass


def cancel(session: Session, job: Job, actor: Actor = Actor.OWNER) -> None:
    """«Отменить» задачу. Взятую агент прервёт при следующем `progress` (`cancel: true`).
    Фото остаётся ждать в очереди с флагом: «Повторить» или «Отложить»."""
    if job.status not in ACTIVE:
        raise JobError(f"Задача #{job.id} уже не выполняется")
    was_running = job.status == JobStatus.CLAIMED
    job.status = JobStatus.CANCELLED
    job.finished_at = utcnow()
    _release(job)
    if was_running:
        photo = _back_to_queue(session, job, Actor.SYSTEM, {"cancelled": True})
    else:
        queued = session.get(Photo, job.photo_id)
        assert queued is not None
        photo = queued
    log_event(session, photo.id, actor, "jobs_cancelled", {"job_ids": [job.id]})
    _flag(photo, f"Задача #{job.id} отменена")


def cancel_active(session: Session, photo_id: int, actor: Actor = Actor.OWNER) -> list[int]:
    """Массовое «Отменить»: все задачи фото в очереди и на ПК (см. `cancel`)."""
    active = session.scalars(
        select(Job).where(Job.photo_id == photo_id, Job.status.in_(ACTIVE)).order_by(Job.id)
    ).all()
    for job in active:
        cancel(session, job, actor)
    return [j.id for j in active]


def raise_priority(session: Session, photo_ids: list[int], actor: Actor = Actor.OWNER) -> int:
    """«Поднять приоритет»: задачи фото в очереди — выше всех остальных в очереди."""
    top = session.scalar(select(func.max(Job.priority)).where(Job.status == JobStatus.QUEUED))
    queued = session.scalars(
        select(Job).where(Job.photo_id.in_(photo_ids), Job.status == JobStatus.QUEUED)
    ).all()
    for job in queued:
        job.priority = (top or 0) + 1
    for photo_id in sorted({j.photo_id for j in queued}):
        ids = [j.id for j in queued if j.photo_id == photo_id]
        log_event(
            session, photo_id, actor, "priority", {"job_ids": ids, "priority": (top or 0) + 1}
        )
    return len(queued)


def can_retry(session: Session, photo: Photo) -> bool:
    queued = {s.queued for s in STAGES.values()}
    return photo.status in queued and active_job(session, photo.id) is None


def retry(session: Session, photo: Photo, actor: Actor = Actor.OWNER) -> Job:
    """«Повторить»: новая задача с тем же заданием, флаг снимается."""
    if not can_retry(session, photo):
        raise JobError(f"{photo.code}: повторять нечего")
    last = session.scalar(
        select(Job)
        .where(Job.photo_id == photo.id, Job.type.in_(LOCAL_TYPES))
        .order_by(Job.id.desc())
        .limit(1)
    )
    if last is None:
        raise JobError(f"{photo.code}: нет задачи, которую можно повторить")
    job = Job(
        photo_id=photo.id, type=last.type, status=JobStatus.QUEUED, payload=dict(last.payload)
    )
    session.add(job)
    session.flush()
    photo.attention = False
    photo.attention_reason = None
    photo.updated_at = utcnow()
    log_event(
        session,
        photo.id,
        actor,
        "job_created",
        {"job_id": job.id, "type": job.type.value, "retry_of": last.id},
    )
    return job
