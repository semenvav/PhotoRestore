"""Единственное место смены статуса фото (SPEC §3).

Любое изменение `photo.status` идёт через `transition()`: она сверяет переход и исполнителя
с таблицей, пишет событие в историю и ведёт счётчик раунда.
"""

from typing import Any

from sqlalchemy.orm import Session

from app.db import utcnow
from app.models import Actor, Event, Photo, Status

S = Status
A = Actor

LABELS: dict[Status, str] = {
    S.PREP: "Подготовка",
    S.QUEUED_EDIT: "В очереди на ПК",
    S.EDITING: "Обработка на ПК",
    S.REVIEW: "Проверка",
    S.GEMINI: "В Gemini",
    S.QUEUED_UPSCALE: "Ждёт апскейла",
    S.UPSCALING: "Апскейл на ПК",
    S.FINAL_REVIEW: "Финал",
    S.DELIVERED: "Отдано",
    S.ACCEPTED: "Принято",
    S.DUPLICATE: "Дубль",
    S.ON_HOLD: "Отложено",
}

# Порядок колонок на доске
BOARD_ORDER: tuple[Status, ...] = (
    S.PREP,
    S.QUEUED_EDIT,
    S.EDITING,
    S.REVIEW,
    S.GEMINI,
    S.QUEUED_UPSCALE,
    S.UPSCALING,
    S.FINAL_REVIEW,
    S.DELIVERED,
    S.ACCEPTED,
    S.ON_HOLD,
    S.DUPLICATE,
)

_NO_HOLD = {S.DELIVERED, S.ACCEPTED, S.EDITING, S.UPSCALING, S.GEMINI, S.ON_HOLD}

_ROWS: list[tuple[Status | None, Status, Actor]] = [
    (None, S.PREP, A.SYSTEM),
    (None, S.DUPLICATE, A.SYSTEM),
    (S.PREP, S.QUEUED_EDIT, A.OWNER),
    (S.QUEUED_EDIT, S.EDITING, A.AGENT),
    (S.EDITING, S.REVIEW, A.AGENT),
    (S.EDITING, S.QUEUED_EDIT, A.SYSTEM),
    (S.REVIEW, S.QUEUED_EDIT, A.OWNER),
    (S.REVIEW, S.GEMINI, A.OWNER),
    (S.GEMINI, S.REVIEW, A.SYSTEM),
    (S.REVIEW, S.REVIEW, A.OWNER),
    (S.REVIEW, S.QUEUED_UPSCALE, A.OWNER),
    (S.REVIEW, S.FINAL_REVIEW, A.OWNER),
    (S.QUEUED_UPSCALE, S.UPSCALING, A.AGENT),
    (S.UPSCALING, S.FINAL_REVIEW, A.AGENT),
    (S.UPSCALING, S.QUEUED_UPSCALE, A.SYSTEM),
    (S.FINAL_REVIEW, S.DELIVERED, A.OWNER),
    (S.FINAL_REVIEW, S.QUEUED_UPSCALE, A.OWNER),
    (S.FINAL_REVIEW, S.REVIEW, A.OWNER),
    (S.DELIVERED, S.ACCEPTED, A.FAMILY),
    (S.DELIVERED, S.REVIEW, A.FAMILY),
    *((src, S.ON_HOLD, A.OWNER) for src in Status if src not in _NO_HOLD),
    (S.ON_HOLD, S.PREP, A.OWNER),
    (S.DUPLICATE, S.PREP, A.OWNER),
]

TRANSITIONS: dict[tuple[Status | None, Status], frozenset[Actor]] = {}
for _src, _dst, _actor in _ROWS:
    TRANSITIONS[(_src, _dst)] = TRANSITIONS.get((_src, _dst), frozenset()) | {_actor}

# Переходы, начинающие новый раунд вариантов
_NEW_ROUND = {(S.REVIEW, S.QUEUED_EDIT, A.OWNER), (S.DELIVERED, S.REVIEW, A.FAMILY)}


class TransitionError(Exception):
    pass


def can_transition(src: Status | None, dst: Status, actor: Actor) -> bool:
    return actor in TRANSITIONS.get((src, dst), frozenset())


def targets(src: Status, actor: Actor) -> set[Status]:
    return {dst for (s, dst), actors in TRANSITIONS.items() if s == src and actor in actors}


def transition(
    session: Session,
    photo: Photo,
    to: Status,
    actor: Actor,
    data: dict[str, Any] | None = None,
) -> Event:
    """Переводит фото в статус `to` и пишет событие `status`. Новое фото (без id)
    регистрируется переходом из `None`."""
    src: Status | None = photo.status if photo.id is not None else None
    if not can_transition(src, to, actor):
        code = photo.code if photo.id is not None else "новое фото"
        raise TransitionError(f"{code}: переход {src} → {to} для «{actor}» запрещён")

    if (src, to, actor) in _NEW_ROUND:
        photo.round += 1
    photo.status = to
    photo.updated_at = utcnow()
    if photo.id is None:
        session.add(photo)
        session.flush()

    event = Event(
        photo_id=photo.id,
        actor=actor,
        kind="status",
        data={"from": src.value if src else None, "to": to.value, **(data or {})},
    )
    session.add(event)
    return event


def log_event(
    session: Session,
    photo_id: int | None,
    actor: Actor,
    kind: str,
    data: dict[str, Any] | None = None,
) -> Event:
    event = Event(photo_id=photo_id, actor=actor, kind=kind, data=data or {})
    session.add(event)
    return event
