"""Удаление фото из сервиса.

Из БД уходят фото, его варианты, задачи и история; в общей истории остаётся событие
`deleted` с исходным именем и хешем. Каталог `store/R-xxxx` переносится в `store/.deleted/`:
ТЗ запрещает удалять оригинал, место освобождают вручную. Запись `inbox_file` остаётся
(без photo_id) — файл во «Входящих» лежит на месте и повторно не принимается.

Если у удаляемого фото есть дубли, они остаются: основным становится первый из них.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.models import Actor, Event, InboxFile, Job, Photo, Status, Variant
from app.states import log_event
from app.storage import Store

log = logging.getLogger(__name__)

# фото в обработке не удаляются: исполнитель держит задачу и вернёт результат
BUSY = frozenset({Status.EDITING, Status.UPSCALING, Status.GEMINI})


@dataclass
class DeleteResult:
    deleted: list[str] = field(default_factory=list)  # коды удалённых фото
    busy: list[str] = field(default_factory=list)  # коды фото, которые сейчас не удалить
    promoted: dict[str, str] = field(default_factory=dict)  # удалённое → новый основной


def can_delete(photo: Photo) -> bool:
    return photo.status not in BUSY


def duplicates_of(session: Session, photo_id: int) -> list[Photo]:
    return list(
        session.scalars(select(Photo).where(Photo.duplicate_of == photo_id).order_by(Photo.id))
    )


def delete_photos(
    session: Session, store: Store, photo_ids: Iterable[int], actor: Actor = Actor.OWNER
) -> DeleteResult:
    """Удаляет фото и фиксирует транзакцию; затем переносит их каталоги в `.deleted/`."""
    result = DeleteResult()
    photos = session.scalars(
        select(Photo).where(Photo.id.in_(set(photo_ids))).order_by(Photo.id)
    ).all()
    doomed = [p for p in photos if can_delete(p)]
    result.busy = [p.code for p in photos if not can_delete(p)]
    if not doomed:
        return result
    ids = sorted(p.id for p in doomed)

    for photo in doomed:
        if photo.duplicate_of is None:
            _promote_duplicate(session, photo, set(ids), actor, result)

    # что было удалено — для общей истории, сами строки сейчас исчезнут
    snapshots = [
        (
            p.id,
            p.code,
            {
                "photo_id": p.id,
                "code": p.code,
                "original_name": p.original_name,
                "album": p.album,
                "sha256": p.sha256,
                "status": p.status.value,
            },
        )
        for p in doomed
    ]
    session.execute(
        update(Photo)
        .where(Photo.id.in_(ids))
        .values(accepted_variant_id=None, final_variant_id=None)
    )
    session.execute(delete(Variant).where(Variant.photo_id.in_(ids)))
    session.execute(delete(Job).where(Job.photo_id.in_(ids)))
    session.execute(delete(Event).where(Event.photo_id.in_(ids)))
    session.execute(update(InboxFile).where(InboxFile.photo_id.in_(ids)).values(photo_id=None))
    session.execute(delete(Photo).where(Photo.id.in_(ids)))
    for _, _, data in snapshots:
        log_event(session, None, actor, "deleted", data)
    session.commit()

    for photo_id, code, _ in snapshots:
        try:
            moved = store.set_aside_deleted(photo_id)
            log.info("удалено %s, файлы → %s", code, moved)
        except OSError:
            # в БД фото уже нет, ID не переиспользуются — каталог просто останется на месте
            log.exception("удалено %s, но каталог не перенесён в .deleted", code)
        result.deleted.append(code)
    return result


def _promote_duplicate(
    session: Session, primary: Photo, doomed: set[int], actor: Actor, result: DeleteResult
) -> None:
    """Первый оставшийся дубль становится основным, остальные ссылаются на него.

    Порядок важен: sha256 уникален среди фото без `duplicate_of`, а внешние ключи
    проверяются сразу — поэтому удаляемое фото сначала само становится «дублем» нового
    основного и только потом новый основной теряет ссылку.
    """
    survivors = [d for d in duplicates_of(session, primary.id) if d.id not in doomed]
    if not survivors:
        return
    head, *rest = survivors
    for dup in rest:
        dup.duplicate_of = head.id
    primary.duplicate_of = head.id
    session.flush()
    head.duplicate_of = None
    session.flush()
    log_event(session, head.id, actor, "promoted", {"deleted": primary.id})
    result.promoted[primary.code] = head.code
