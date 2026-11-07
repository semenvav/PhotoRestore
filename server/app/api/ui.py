"""Интерфейс владельца: доска, карточка фото, подготовка, загрузка, файлы."""

import hashlib
import os
import re
import shutil
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import deletion, jobs, prep, runs, states
from app.api.agent import agent_compatible
from app.db import session_scope, utcnow
from app.imaging import PrepParams, UnsupportedImage
from app.intake import Intake
from app.models import Actor, Agent, Event, Job, Photo, Status, Variant
from app.states import BOARD_ORDER, LABELS, TransitionError
from app.storage import ALLOWED_EXT, Store
from app.views import AGENT_OFFLINE_AFTER

router = APIRouter()

_PREVIEW_KIND = re.compile(r"^(original|prepared|v\d{3,})$")
ALL_ALBUMS = "*"
_BAD_NAME_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]')
_PHOTO_REF = re.compile(r"^\s*(?:r\s*-?\s*)?0*(\d+)\s*$", re.IGNORECASE)


def get_session(request: Request) -> Iterator[Session]:
    with session_scope(request.app.state.factory) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def get_store(request: Request) -> Store:
    store: Store = request.app.state.store
    return store


StoreDep = Annotated[Store, Depends(get_store)]


def render(request: Request, name: str, ctx: dict[str, Any], status_code: int = 200) -> Response:
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)


def safe_back(url: str) -> str:
    """Возврат только на свой путь: `//host` и абсолютные адреса заменяются доской."""
    return url if url.startswith("/") and not url.startswith("//") else "/"


def with_msg(url: str, msg: str | None) -> str:
    if not msg:
        return url
    return f"{url}{'&' if '?' in url else '?'}msg={quote(msg)}"


def load_photo(session: Session, photo_id: int) -> Photo:
    photo = session.get(Photo, photo_id)
    if photo is None:
        raise HTTPException(404, "Фото не найдено")
    return photo


def current_agent(session: Session) -> Agent | None:
    return session.scalar(select(Agent).order_by(Agent.last_seen.desc()).limit(1))


def agent_online(agent: Agent | None) -> bool:
    return bool(agent and agent.last_seen and utcnow() - agent.last_seen < AGENT_OFFLINE_AFTER)


def status_count(session: Session, status: Status) -> int:
    return session.scalar(select(func.count()).where(Photo.status == status)) or 0


def header_ctx(session: Session) -> dict[str, Any]:
    agent = current_agent(session)
    run = runs.active_run(session)
    return {
        "agent": agent,
        "agent_online": agent_online(agent),
        "agent_outdated": agent is not None and not agent_compatible(agent.version),
        "run": run,
        "run_progress": runs.progress(session, run, jobs.LOCAL_TYPES) if run else None,
        "prep_count": status_count(session, Status.PREP),
        "review_count": status_count(session, Status.REVIEW),
    }


@router.get("/healthz")
def healthz(request: Request) -> dict[str, Any]:
    return {"ok": True, "version": request.app.state.settings.app_version}


@router.get("/partials/header")
def header_partial(request: Request, session: SessionDep) -> Response:
    return render(request, "_header_status.html", {"hdr": header_ctx(session)})


# --- доска ------------------------------------------------------------------------------


def board_token(session: Session) -> str:
    count, updated = session.execute(select(func.count(Photo.id), func.max(Photo.updated_at))).one()
    last_event = session.scalar(select(func.max(Event.id)))
    raw = f"{count}|{updated}|{last_event}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def board_ctx(session: Session, album: str | None, q: str | None) -> dict[str, Any]:
    query = select(Photo)
    if album is not None and album != ALL_ALBUMS:
        query = query.where(Photo.album == album)
    if q and q.strip():
        if m := _PHOTO_REF.match(q):
            query = query.where(Photo.id == int(m.group(1)))
        else:
            query = query.where(Photo.original_name.icontains(q.strip(), autoescape=True))
    photos = session.scalars(query.order_by(Photo.id)).all()
    by_status: dict[Status, list[Photo]] = {s: [] for s in BOARD_ORDER}
    for photo in photos:
        by_status[photo.status].append(photo)
    columns = [(s, LABELS[s], by_status[s]) for s in BOARD_ORDER]
    params = {"album": album} if album is not None else {}
    if q:
        params["q"] = q
    return {
        "columns": columns,
        "qs": urlencode(params),
        "total": len(photos),
        "token": board_token(session),
        "album": album,
        "q": q or "",
    }


@router.get("/")
def board(
    request: Request,
    session: SessionDep,
    album: str | None = None,
    q: str | None = None,
    msg: str | None = None,
) -> Response:
    albums = session.scalars(select(Photo.album).distinct().order_by(Photo.album)).all()
    notices = session.scalars(
        select(Event)
        .where(
            Event.photo_id.is_(None),
            Event.kind == "unsupported",
            Event.ts > utcnow() - timedelta(days=7),
        )
        .order_by(Event.id.desc())
        .limit(10)
    ).all()
    ctx = board_ctx(session, album, q) | {
        "hdr": header_ctx(session),
        "albums": albums,
        "notices": notices,
        "accept": ",".join(f".{ext}" for ext in sorted(ALLOWED_EXT)),
        "msg": msg,
    }
    return render(request, "board.html", ctx)


@router.get("/partials/columns")
def board_columns(
    request: Request,
    session: SessionDep,
    album: str | None = None,
    q: str | None = None,
    v: str | None = None,
) -> Response:
    if v is not None and v == board_token(session):
        return Response(status_code=204)
    return render(request, "_columns.html", board_ctx(session, album, q))


# --- карточка и действия ----------------------------------------------------------------


@router.get("/photo/{photo_id}")
def photo_card(
    request: Request, session: SessionDep, store: StoreDep, photo_id: int, msg: str | None = None
) -> Response:
    photo = load_photo(session, photo_id)
    events = session.scalars(
        select(Event).where(Event.photo_id == photo.id).order_by(Event.id.desc())
    ).all()
    photo_jobs = session.scalars(
        select(Job).where(Job.photo_id == photo.id).order_by(Job.id.desc())
    ).all()
    ctx = {
        "hdr": header_ctx(session),
        "photo": photo,
        "events": events,
        "jobs": photo_jobs,
        "duplicate_of": session.get(Photo, photo.duplicate_of) if photo.duplicate_of else None,
        "has_prepared": store.prepared(photo.id).exists(),
        "can_prep": photo.status in prep.PREPARABLE,
        "can_hold": states.can_transition(photo.status, Status.ON_HOLD, Actor.OWNER),
        "can_unhold": photo.status in (Status.ON_HOLD, Status.DUPLICATE),
        "can_delete": deletion.can_delete(photo),
        "duplicates": deletion.duplicates_of(session, photo.id),
        "can_retry": photo.attention and jobs.can_retry(session, photo),
        "variants_count": session.scalar(select(func.count()).where(Variant.photo_id == photo.id)),
        "msg": msg,
    }
    return render(request, "photo.html", ctx)


def hold(session: Session, photo: Photo, actor: Actor = Actor.OWNER) -> None:
    states.transition(session, photo, Status.ON_HOLD, actor)
    jobs.cancel_queued(session, photo.id, actor)


def unhold(session: Session, photo: Photo, actor: Actor = Actor.OWNER) -> None:
    if photo.status not in (Status.ON_HOLD, Status.DUPLICATE):
        raise TransitionError(f"{photo.code}: «Вернуть в работу» только для отложенных и дублей")
    states.transition(session, photo, Status.PREP, actor)


ACTIONS = {"hold": hold, "unhold": unhold}


def delete_msg(result: deletion.DeleteResult) -> str:
    parts = []
    if result.deleted:
        parts.append(f"Удалено: {', '.join(result.deleted)}")
    for gone, head in result.promoted.items():
        parts.append(f"основным вместо {gone} стал его дубль {head}")
    if result.busy:
        parts.append(f"в обработке, не удалено: {', '.join(result.busy)}")
    return "; ".join(parts) or "Нечего удалять"


@router.post("/photo/{photo_id}/delete")
def photo_delete(
    session: SessionDep,
    store: StoreDep,
    photo_id: int,
    back: Annotated[str, Form()] = "/",
) -> Response:
    photo = load_photo(session, photo_id)
    result = deletion.delete_photos(session, store, [photo.id])
    if not result.deleted:
        return RedirectResponse(with_msg(f"/photo/{photo_id}", delete_msg(result)), status_code=303)
    return RedirectResponse(with_msg(safe_back(back), delete_msg(result)), status_code=303)


@router.post("/photo/{photo_id}/{action}")
def photo_action(session: SessionDep, photo_id: int, action: str) -> Response:
    if action not in ACTIONS:
        raise HTTPException(404)
    photo = load_photo(session, photo_id)
    try:
        ACTIONS[action](session, photo)
        session.commit()
        msg = None
    except TransitionError as e:
        session.rollback()
        msg = str(e)
    return RedirectResponse(with_msg(f"/photo/{photo_id}", msg), status_code=303)


@router.post("/bulk")
def bulk_action(
    request: Request,
    session: SessionDep,
    action: Annotated[str, Form()],
    ids: Annotated[list[int], Form()],
    back: Annotated[str, Form()] = "/",
) -> Response:
    if action == "delete":
        result = deletion.delete_photos(session, request.app.state.store, ids)
        return RedirectResponse(with_msg(safe_back(back), delete_msg(result)), status_code=303)
    if action == "run":
        try:
            run = runs.start(session, request.app.state.registry.job_types(), ids)
            session.commit()
            msg = f"Запуск #{run.id}: задач {len(run.job_ids or [])}"
        except runs.RunError as e:
            msg = str(e)
        return RedirectResponse(with_msg(safe_back(back), msg), status_code=303)
    if action not in ACTIONS:
        raise HTTPException(400, "неизвестное действие")
    done, errors = 0, []
    for photo_id in ids:
        photo = session.get(Photo, photo_id)
        if photo is None:
            continue
        try:
            # transition() проверяет переход до любых изменений — откат не нужен
            ACTIONS[action](session, photo)
            done += 1
        except TransitionError:
            errors.append(photo.code)
    session.commit()
    msg = f"Готово: {done}"
    if errors:
        msg += f"; недоступно для {', '.join(errors)}"
    return RedirectResponse(with_msg(safe_back(back), msg), status_code=303)


# --- подготовка -------------------------------------------------------------------------


@router.get("/prep")
def prep_queue(request: Request, session: SessionDep) -> Response:
    nxt = prep.next_in_queue(session)
    if nxt is not None:
        return RedirectResponse(f"/prep/{nxt.id}", status_code=303)
    return render(request, "prep_empty.html", {"hdr": header_ctx(session)})


@router.get("/prep/{photo_id}/skip")
def prep_skip(session: SessionDep, photo_id: int) -> Response:
    nxt = prep.next_in_queue(session, after_id=photo_id)
    if nxt is None:
        return RedirectResponse("/prep", status_code=303)
    return RedirectResponse(f"/prep/{nxt.id}", status_code=303)


@router.get("/prep/{photo_id}")
def prep_editor(request: Request, session: SessionDep, store: StoreDep, photo_id: int) -> Response:
    photo = load_photo(session, photo_id)
    similar = session.scalars(
        select(Event)
        .where(Event.photo_id == photo.id, Event.kind == "similar")
        .order_by(Event.id.desc())
        .limit(1)
    ).first()
    similar_photos = similar.data["photos"] if similar else []
    if similar_photos:  # похожие могли удалить
        alive = set(
            session.scalars(
                select(Photo.id).where(Photo.id.in_([s["photo_id"] for s in similar_photos]))
            )
        )
        similar_photos = [s for s in similar_photos if s["photo_id"] in alive]
    in_queue = photo.status == Status.PREP
    position = None
    if in_queue:
        before = session.scalar(
            select(func.count()).where(Photo.status == Status.PREP, Photo.id < photo.id)
        )
        position = (before or 0) + 1
    ctx = {
        "hdr": header_ctx(session),
        "photo": photo,
        "editable": photo.status in prep.PREPARABLE,
        "in_queue": in_queue,
        "position": position,
        "similar": similar_photos,
        "can_delete": deletion.can_delete(photo),
        "duplicates": deletion.duplicates_of(session, photo.id),
        # после удаления из очереди — к следующему фото
        "delete_back": f"/prep/{photo.id}/skip" if in_queue else "/",
        "config": {
            "photoId": photo.id,
            "origWidth": photo.width,
            "origHeight": photo.height,
            "saveUrl": f"/prep/{photo.id}",
            "skipUrl": f"/prep/{photo.id}/skip" if in_queue else None,
            "prev": photo.prep_params,
        },
        "has_prepared": store.prepared(photo.id).exists(),
    }
    return render(request, "prep.html", ctx)


@router.post("/prep/{photo_id}")
async def prep_save(request: Request, photo_id: int) -> Response:
    try:
        params = PrepParams.model_validate(await request.json())
    except (ValueError, ValidationError) as e:
        return JSONResponse({"error": f"неверные параметры: {e}"}, status_code=400)
    return await run_in_threadpool(_prep_save_sync, request, photo_id, params)


def _prep_save_sync(request: Request, photo_id: int, params: PrepParams) -> Response:
    with session_scope(request.app.state.factory) as session:
        photo = load_photo(session, photo_id)
        from_queue = photo.status == Status.PREP
        try:
            prep.save(session, request.app.state.store, photo, params)
        except (prep.PrepError, TransitionError, UnsupportedImage) as e:
            session.rollback()
            return JSONResponse({"error": str(e)}, status_code=409)
        session.commit()
        if from_queue:
            nxt = prep.next_in_queue(session, after_id=photo.id)
            next_url = f"/prep/{nxt.id}" if nxt else "/prep"
        else:
            next_url = f"/photo/{photo.id}"
    return JSONResponse({"ok": True, "next": next_url})


# --- загрузка ---------------------------------------------------------------------------


def clean_segment(value: str) -> str:
    value = _BAD_NAME_CHARS.sub("_", value).strip().strip(".")
    return value[:120]


def clean_album(value: str) -> str:
    parts = [clean_segment(p) for p in re.split(r"[\\/]+", value or "")]
    parts = [p for p in parts if p and p not in (".", "..")]
    if parts and parts[0] == "_unsupported":
        parts[0] = "unsupported"
    return "/".join(parts)


def _save_upload(intake: Intake, upload: UploadFile, album: str) -> str | None:
    name = clean_segment(Path(upload.filename or "").name)
    stem, ext = os.path.splitext(name)
    if not stem or ext.lower().lstrip(".") not in ALLOWED_EXT:
        return None
    target_dir = intake.inbox / album if album else intake.inbox
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp = target_dir / f".upload-{uuid.uuid4().hex}.tmp"
    with tmp.open("xb") as dst:
        shutil.copyfileobj(upload.file, dst, 1 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    dest, n = target_dir / name, 1
    while dest.exists():
        dest = target_dir / f"{stem}_{n}{ext}"
        n += 1
    os.replace(tmp, dest)
    rel = dest.relative_to(intake.inbox).as_posix()
    intake.trust(rel)
    return rel


@router.post("/upload")
async def upload(
    request: Request,
    files: Annotated[list[UploadFile], File()],
    album: Annotated[str, Form()] = "",
) -> Response:
    intake: Intake = request.app.state.intake
    album = clean_album(album)
    saved: list[str] = []
    skipped: list[str] = []
    for f in files:
        rel = await run_in_threadpool(_save_upload, intake, f, album)
        (saved if rel else skipped).append(rel or (f.filename or "?"))
    if saved and (wake := getattr(request.app.state, "wake", None)) is not None:
        wake.set()
    msg = f"Загружено во «Входящие»: {len(saved)}"
    if skipped:
        msg += f"; пропущены (не изображения): {', '.join(skipped)}"
    return RedirectResponse(with_msg("/", msg), status_code=303)


# --- файлы ------------------------------------------------------------------------------

_CACHE = {"Cache-Control": "private, max-age=31536000, immutable"}


def _file(path: Path, **kwargs: Any) -> FileResponse:
    if not path.is_file():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(path, **kwargs)


@router.get("/files/{photo_id}/thumb")
def file_thumb(session: SessionDep, store: StoreDep, photo_id: int) -> Response:
    load_photo(session, photo_id)
    return _file(store.thumb(photo_id), media_type="image/webp", headers=_CACHE)


@router.get("/files/{photo_id}/preview/{kind}")
def file_preview(session: SessionDep, store: StoreDep, photo_id: int, kind: str) -> Response:
    if not _PREVIEW_KIND.match(kind):
        raise HTTPException(404)
    load_photo(session, photo_id)
    return _file(store.preview(photo_id, kind), media_type="image/jpeg", headers=_CACHE)


@router.get("/files/{photo_id}/original")
def file_original(session: SessionDep, store: StoreDep, photo_id: int) -> Response:
    photo = load_photo(session, photo_id)
    return _file(store.original(photo.id, photo.original_name), filename=photo.original_name)


@router.get("/files/{photo_id}/prepared")
def file_prepared(
    session: SessionDep, store: StoreDep, photo_id: int, download: Annotated[bool, Query()] = False
) -> Response:
    photo = load_photo(session, photo_id)
    return _file(
        store.prepared(photo.id),
        media_type="image/png",
        filename=f"{photo.code}_prepared.png" if download else None,
    )
