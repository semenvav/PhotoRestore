"""Подготовка фото (SPEC §5.2): сервер сам кропает и поворачивает оригинал по параметрам
из редактора и пишет `prepared.png`; первая подготовка ставит задачу `edit_local`."""

import os
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import imaging, jobs, states
from app.imaging import PrepParams
from app.models import Actor, Photo, Status
from app.storage import Store

# «Переподготовить» доступно до DELIVERED, но не пока фото обрабатывается
PREPARABLE = frozenset(
    {
        Status.PREP,
        Status.QUEUED_EDIT,
        Status.REVIEW,
        Status.QUEUED_UPSCALE,
        Status.FINAL_REVIEW,
        Status.ON_HOLD,
    }
)


class PrepError(Exception):
    pass


def save(
    session: Session, store: Store, photo: Photo, params: PrepParams, actor: Actor = Actor.OWNER
) -> dict[str, Any]:
    if photo.status not in PREPARABLE:
        raise PrepError(f"{photo.code}: подготовка недоступна в статусе {photo.status}")

    original = store.original(photo.id, photo.original_name)
    img = imaging.load_oriented(original)
    try:
        prepared = imaging.prepare(img, params)
    except ValueError as e:
        raise PrepError(str(e)) from e

    target = store.prepared(photo.id)
    tmp = target.with_name("prepared.png.tmp")
    imaging.save_png(prepared, tmp)
    archived = None
    if target.exists():
        archived = store.next_prepared_archive(photo.id)
        os.replace(target, archived)
    os.replace(tmp, target)
    imaging.save_thumb(prepared, store.thumb(photo.id))
    imaging.save_preview(prepared, store.preview(photo.id, "prepared"))

    photo.prep_params = params.model_dump() | {
        "width": prepared.width,
        "height": prepared.height,
    }
    summary: dict[str, Any] = {
        "rotate90": params.rotate90,
        "angle": params.angle,
        "cropped": params.crop is not None,
        "width": prepared.width,
        "height": prepared.height,
    }
    if archived is not None:
        summary["archived"] = archived.name

    if photo.status == Status.PREP:
        states.transition(session, photo, Status.QUEUED_EDIT, actor, {"prepared": summary})
        jobs.create_edit_job(session, photo, actor)
    else:
        states.log_event(session, photo.id, actor, "reprepared", summary)
    return summary


def next_in_queue(session: Session, after_id: int | None = None) -> Photo | None:
    """Следующее фото в PREP в порядке поступления; после последнего — снова с начала."""
    query = select(Photo).where(Photo.status == Status.PREP).order_by(Photo.id)
    if after_id is not None:
        nxt = session.scalar(query.where(Photo.id > after_id).limit(1))
        if nxt is not None:
            return nxt
    return session.scalar(query.limit(1))
