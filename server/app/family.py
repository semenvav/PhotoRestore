"""Приёмка отцом (ТЗ §5.6): «Принять» (DELIVERED → ACCEPTED) и «Вернуть» с обязательным
комментарием (DELIVERED → REVIEW, новый раунд, копия из «Готово» удаляется — финал остаётся
в `store/`)."""

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import delivery, states
from app.models import Actor, Photo, Status, Variant

log = logging.getLogger(__name__)

MAX_COMMENT = 2000
SHOWN = (Status.DELIVERED, Status.ACCEPTED)


class FamilyError(Exception):
    pass


def waiting(session: Session) -> list[Photo]:
    """Ждут ответа — в порядке отдачи."""
    return list(
        session.scalars(
            select(Photo)
            .where(Photo.status == Status.DELIVERED)
            .order_by(Photo.updated_at, Photo.id)
        )
    )


def accepted(session: Session) -> list[Photo]:
    """Принятые — последние сверху."""
    return list(
        session.scalars(
            select(Photo)
            .where(Photo.status == Status.ACCEPTED)
            .order_by(Photo.updated_at.desc(), Photo.id.desc())
        )
    )


def next_waiting(session: Session, after: Photo) -> Photo | None:
    """Следующее ждущее ответа после `after` (по кругу), без него самого."""
    queue = [p for p in waiting(session) if p.id != after.id]
    if not queue:
        return None
    later = [p for p in queue if (p.updated_at, p.id) > (after.updated_at, after.id)]
    return later[0] if later else queue[0]


def final_variant(session: Session, photo: Photo) -> Variant | None:
    variant = session.get(Variant, photo.final_variant_id) if photo.final_variant_id else None
    return variant if variant is not None and variant.photo_id == photo.id else None


def accept(session: Session, photo: Photo) -> None:
    final = final_variant(session, photo)
    states.transition(
        session, photo, Status.ACCEPTED, Actor.FAMILY, {"variant": final.label if final else None}
    )


def clean_comment(text: str) -> str:
    comment = "\n".join(line.rstrip() for line in text.strip().splitlines())
    if not comment:
        raise FamilyError("Напишите, что нужно исправить")
    if len(comment) > MAX_COMMENT:
        raise FamilyError(f"Комментарий слишком длинный: не больше {MAX_COMMENT} знаков")
    return comment


def remove_delivered(delivered_dir: Path, photo: Photo) -> dict[str, Any]:
    """Убирает из «Готово» копию этого фото. Чужой или заменённый файл (метаданное `Title`
    не наше) не трогается; что вышло — в событие."""
    rel = photo.delivered_path
    if not rel:
        return {}
    path = delivered_dir / rel
    if not path.resolve().is_relative_to(delivered_dir.resolve()):
        return {"delivered_kept": rel, "reason": "путь вне «Готово»"}
    if not path.is_file():
        return {"delivered_missing": rel}
    if delivery.read_text(path).get("Title") != photo.code:
        return {"delivered_kept": rel, "reason": "файл заменён"}
    try:
        path.unlink()
    except OSError as e:
        log.warning("%s: не удалось убрать %s из «Готово»: %s", photo.code, path, e)
        return {"delivered_kept": rel, "reason": f"не удалось удалить: {e.strerror or e}"}
    return {"delivered_removed": rel}


def send_back(session: Session, delivered_dir: Path, photo: Photo, text: str) -> dict[str, Any]:
    """«Вернуть»: → REVIEW (раунд + 1) с комментарием, копия из «Готово» удаляется,
    принятый и финальный варианты сбрасываются (финал остаётся в ленте вариантов)."""
    comment = clean_comment(text)
    final = final_variant(session, photo)
    event = states.transition(
        session,
        photo,
        Status.REVIEW,
        Actor.FAMILY,
        {"comment": comment, "variant": final.label if final else None},
    )
    outcome = remove_delivered(delivered_dir, photo)
    event.data = {**event.data, **outcome}
    photo.family_comment = comment
    photo.delivered_path = None
    photo.accepted_variant_id = None
    photo.final_variant_id = None
    return outcome
