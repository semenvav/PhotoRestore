"""Экран отца `/family` (ТЗ §5.6): только «Отдано» и «Принято», крупное «было / стало»,
кнопки «Принять» и «Вернуть» с обязательным комментарием, без технических деталей.

Отвечать может только роль «отец» (хост `FAMILY_HOST`, см. `auth.py`); владелец видит тот
же экран как предпросмотр, без кнопок.
"""

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy.orm import Session

from app import family
from app.api.ui import SessionDep, StoreDep, _file, render, with_msg
from app.auth import FAMILY_HOME, Role, parse_hosts, role
from app.config import Settings
from app.models import Photo, Status, Variant
from app.states import TransitionError
from app.storage import Store

router = APIRouter(prefix=FAMILY_HOME)

_CACHE = {"Cache-Control": "private, max-age=31536000, immutable"}
ONLY_FAMILY = "Принять или вернуть фото может только отец — со своего адреса"


def stem(photo: Photo) -> str:
    return Path(photo.original_name).stem or photo.original_name


def title(photo: Photo) -> str:
    """Имя для отца: исходное имя без расширения, `_` — пробелы."""
    return " ".join(stem(photo).replace("_", " ").split()) or stem(photo)


def shown_photo(session: Session, photo_id: int) -> tuple[Photo, Variant]:
    photo = session.get(Photo, photo_id)
    final = family.final_variant(session, photo) if photo else None
    if photo is None or photo.status not in family.SHOWN or final is None:
        raise HTTPException(404, "Фото не найдено")
    return photo, final


def has_prepared(store: Store, photo: Photo) -> bool:
    return store.prepared(photo.id).is_file()


def card(photo: Photo) -> dict[str, Any]:
    return {
        "id": photo.id,
        "title": title(photo),
        "album": photo.album,
        "when": photo.updated_at,
        "thumb": f"{FAMILY_HOME}/{photo.id}/thumb.webp?v={photo.final_variant_id}",
        "comment": photo.family_comment if photo.status == Status.DELIVERED else None,
    }


def base_ctx(request: Request, msg: str | None) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    hosts = sorted(parse_hosts(settings.family_host))
    return {"preview": role(request) is not Role.FAMILY, "family_hosts": hosts, "msg": msg}


@router.get("")
def family_home(request: Request, session: SessionDep, msg: str | None = None) -> Response:
    ctx = base_ctx(request, msg)
    ctx["waiting"] = [card(p) for p in family.waiting(session)]
    ctx["accepted"] = [card(p) for p in family.accepted(session)]
    return render(request, "family.html", ctx)


@router.get("/{photo_id}")
def family_photo(
    request: Request, session: SessionDep, store: StoreDep, photo_id: int, msg: str | None = None
) -> Response:
    try:
        photo, final = shown_photo(session, photo_id)
    except HTTPException:
        return RedirectResponse(with_msg(FAMILY_HOME, "Этого фото здесь уже нет"), 303)
    params = photo.prep_params or {}
    prepared = has_prepared(store, photo)
    before_preview = f"{FAMILY_HOME}/{photo.id}/before.jpg?v={int(photo.updated_at.timestamp())}"
    after_full = f"{FAMILY_HOME}/{photo.id}/after.png?v={final.id}"
    ctx = base_ctx(request, msg)
    nxt = family.next_waiting(session, photo) if photo.status == Status.DELIVERED else None
    ctx |= {
        "photo": photo,
        "title": title(photo),
        "waiting": photo.status == Status.DELIVERED,
        "can_answer": photo.status == Status.DELIVERED and not ctx["preview"],
        "comment": photo.family_comment if photo.status == Status.DELIVERED else None,
        "download": f"{after_full}&download=1",
        "next_id": nxt.id if nxt else None,
        "max_comment": family.MAX_COMMENT,
        "config": {
            "images": [
                {
                    "key": "prepared",
                    "label": "Было",
                    "preview": before_preview,
                    "full": f"{FAMILY_HOME}/{photo.id}/before.png" if prepared else before_preview,
                    "width": params.get("width", photo.width) if prepared else photo.width,
                    "height": params.get("height", photo.height) if prepared else photo.height,
                },
                {
                    "key": "after",
                    "label": "Стало",
                    "preview": f"{FAMILY_HOME}/{photo.id}/after.jpg?v={final.id}",
                    "full": after_full,
                    "width": final.width,
                    "height": final.height,
                },
            ],
            "right": "after",
            "nextUrl": f"{FAMILY_HOME}/{nxt.id}" if nxt else None,
        },
    }
    return render(request, "family_photo.html", ctx)


# --- картинки: только у показанных фото ------------------------------------------------


@router.get("/{photo_id}/thumb.webp")
def family_thumb(session: SessionDep, store: StoreDep, photo_id: int) -> Response:
    photo, final = shown_photo(session, photo_id)
    return _file(
        store.variant_thumb(photo.id, final.number), media_type="image/webp", headers=_CACHE
    )


@router.get("/{photo_id}/before.jpg")
def family_before(session: SessionDep, store: StoreDep, photo_id: int) -> Response:
    photo, _ = shown_photo(session, photo_id)
    kind = "prepared" if has_prepared(store, photo) else "original"
    return _file(store.preview(photo.id, kind), media_type="image/jpeg", headers=_CACHE)


@router.get("/{photo_id}/before.png")
def family_before_full(session: SessionDep, store: StoreDep, photo_id: int) -> Response:
    photo, _ = shown_photo(session, photo_id)
    return _file(store.prepared(photo.id), media_type="image/png")


@router.get("/{photo_id}/after.jpg")
def family_after(session: SessionDep, store: StoreDep, photo_id: int) -> Response:
    photo, final = shown_photo(session, photo_id)
    return _file(store.preview(photo.id, final.label), media_type="image/jpeg", headers=_CACHE)


@router.get("/{photo_id}/after.png")
def family_after_full(
    session: SessionDep,
    store: StoreDep,
    photo_id: int,
    download: Annotated[bool, Query()] = False,
) -> Response:
    photo, final = shown_photo(session, photo_id)
    return _file(
        store.photo_dir(photo.id) / final.path,
        media_type="image/png",
        filename=f"{stem(photo)}_restored.png" if download else None,
        headers=_CACHE,
    )


# --- ответ отца -------------------------------------------------------------------------


def after_answer(session: Session, photo: Photo, msg: str) -> Response:
    nxt = family.next_waiting(session, photo)
    url = f"{FAMILY_HOME}/{nxt.id}" if nxt else FAMILY_HOME
    return RedirectResponse(with_msg(url, msg), status_code=303)


def answered_already(photo: Photo) -> str:
    if photo.status == Status.ACCEPTED:
        return "Это фото уже принято"
    return "Ответ на это фото уже получен"


@router.post("/{photo_id}/accept")
def family_accept(request: Request, session: SessionDep, photo_id: int) -> Response:
    here = f"{FAMILY_HOME}/{photo_id}"
    if role(request) is not Role.FAMILY:
        return RedirectResponse(with_msg(here, ONLY_FAMILY), status_code=303)
    photo, _ = shown_photo(session, photo_id)
    try:
        family.accept(session, photo)
    except TransitionError:
        return RedirectResponse(with_msg(FAMILY_HOME, answered_already(photo)), 303)
    session.commit()
    return after_answer(session, photo, f"Спасибо! «{title(photo)}» принято.")


@router.post("/{photo_id}/return")
def family_return(
    request: Request,
    session: SessionDep,
    photo_id: int,
    comment: Annotated[str, Form()] = "",
) -> Response:
    here = f"{FAMILY_HOME}/{photo_id}"
    if role(request) is not Role.FAMILY:
        return RedirectResponse(with_msg(here, ONLY_FAMILY), status_code=303)
    photo, _ = shown_photo(session, photo_id)
    if photo.status != Status.DELIVERED:
        return RedirectResponse(with_msg(FAMILY_HOME, answered_already(photo)), 303)
    settings: Settings = request.app.state.settings
    try:
        family.send_back(session, settings.delivered_dir, photo, comment)
    except family.FamilyError as e:
        return RedirectResponse(with_msg(f"{here}?return=1", str(e)), status_code=303)
    except TransitionError:
        session.rollback()
        return RedirectResponse(with_msg(FAMILY_HOME, answered_already(photo)), 303)
    session.commit()
    return after_answer(session, photo, f"«{title(photo)}» вернули на доработку. Спасибо!")
