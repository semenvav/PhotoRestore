"""Принятие варианта, апскейл и финал (ТЗ §5.4–5.6).

REVIEW → «Принять» (QUEUED_UPSCALE + задача апскейла) или «Принять без апскейла»
(FINAL_REVIEW, финал = принятый вариант). На финале сравниваются принятый вариант и апскейл:
«Утвердить» (→ DELIVERED, копия во «Готово»), «Переделать апскейл» (→ QUEUED_UPSCALE с другой
моделью или масштабом), «Без апскейла» (финал = принятый, статус тот же), «Назад к вариантам»
(→ REVIEW).
"""

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import delivery, jobs, states
from app.db import utcnow
from app.models import Actor, Job, Photo, Status, Variant, VariantSource
from app.states import log_event
from app.storage import Store


class FinalError(Exception):
    pass


def _variant(session: Session, photo: Photo, variant_id: int | None, what: str) -> Variant:
    variant = session.get(Variant, variant_id) if variant_id else None
    if variant is None or variant.photo_id != photo.id:
        raise FinalError(f"{photo.code}: нет {what}")
    return variant


def accepted(session: Session, photo: Photo) -> Variant:
    return _variant(session, photo, photo.accepted_variant_id, "принятого варианта")


def candidates(session: Session, photo: Photo) -> list[Variant]:
    """Что можно утвердить: принятый вариант и апскейлы из него, новые первыми."""
    if photo.accepted_variant_id is None:
        return []
    ups = session.scalars(
        select(Variant)
        .where(
            Variant.photo_id == photo.id,
            Variant.parent_id == photo.accepted_variant_id,
            Variant.source == VariantSource.UPSCALE,
        )
        .order_by(Variant.id.desc())
    ).all()
    base = session.get(Variant, photo.accepted_variant_id)
    return [*ups, *([base] if base else [])]


def accept(session: Session, photo: Photo, variant: Variant, actor: Actor = Actor.OWNER) -> Job:
    """«Принять» → QUEUED_UPSCALE и задача апскейла с параметрами по умолчанию."""
    if variant.photo_id != photo.id:
        raise FinalError(f"{variant.label} — вариант другого фото")
    states.transition(session, photo, Status.QUEUED_UPSCALE, actor, {"accepted": variant.label})
    photo.accepted_variant_id = variant.id
    photo.final_variant_id = None
    return jobs.create_upscale_job(session, photo, variant, actor)


def accept_without_upscale(
    session: Session, photo: Photo, variant: Variant, actor: Actor = Actor.OWNER
) -> None:
    """«Принять без апскейла» → FINAL_REVIEW, финал = этот вариант."""
    if variant.photo_id != photo.id:
        raise FinalError(f"{variant.label} — вариант другого фото")
    states.transition(
        session, photo, Status.FINAL_REVIEW, actor, {"accepted": variant.label, "final": "same"}
    )
    photo.accepted_variant_id = variant.id
    photo.final_variant_id = variant.id


def redo_upscale(
    session: Session, photo: Photo, model: str, scale: float | None, actor: Actor = Actor.OWNER
) -> Job:
    """«Переделать апскейл» принятого варианта: модель и масштаб (None — авто)."""
    base = accepted(session, photo)
    states.transition(
        session,
        photo,
        Status.QUEUED_UPSCALE,
        actor,
        {"redo": True, "model": model, "scale": scale or "auto"},
    )
    return jobs.create_upscale_job(session, photo, base, actor, model=model, scale=scale)


def without_upscale(session: Session, photo: Photo, actor: Actor = Actor.OWNER) -> Variant:
    """«Без апскейла»: финал = принятый вариант; статус не меняется."""
    if photo.status != Status.FINAL_REVIEW:
        raise FinalError(f"{photo.code}: не на финале")
    base = accepted(session, photo)
    photo.final_variant_id = base.id
    photo.updated_at = utcnow()
    log_event(session, photo.id, actor, "final_set", {"variant": base.label, "upscale": False})
    return base


def back_to_variants(session: Session, photo: Photo, actor: Actor = Actor.OWNER) -> None:
    states.transition(session, photo, Status.REVIEW, actor, {"back": True})
    photo.accepted_variant_id = None
    photo.final_variant_id = None


def approve(
    session: Session,
    store: Store,
    delivered_dir: Path,
    photo: Photo,
    display_tz: str,
    variant_id: int | None = None,
    actor: Actor = Actor.OWNER,
) -> Path:
    """«Утвердить»: финал (или выбранный кандидат) → «Готово», статус DELIVERED."""
    if photo.status != Status.FINAL_REVIEW:
        raise FinalError(f"{photo.code}: утвердить можно только на финале")
    allowed = {v.id: v for v in candidates(session, photo)}
    chosen = variant_id or photo.final_variant_id
    if chosen is None or chosen not in allowed:
        raise FinalError(f"{photo.code}: нет финального варианта")
    return delivery.deliver(
        session, store, delivered_dir, photo, allowed[chosen], display_tz, actor
    )
