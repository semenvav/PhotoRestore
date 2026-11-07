"""Экран проверки (ТЗ §5.4): лента вариантов, сравнение, доработка локально и в Gemini,
ручная правка, «Скрыть», «Принять»."""

from itertools import groupby
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from PIL import Image, ImageOps
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import final, gemini, imaging, jobs, presets, results, states
from app.api.ui import (
    _CACHE,
    SessionDep,
    StoreDep,
    _file,
    display_tz,
    gemini_worker,
    header_ctx,
    load_photo,
    render,
    with_msg,
)
from app.models import Actor, Event, Photo, Status, Variant, VariantSource
from app.states import TransitionError
from app.storage import ALLOWED_EXT, Store

router = APIRouter()

MAX_VARIANTS = presets.MAX_VARIANTS
# ручная правка: форматы, которые открывает Pillow (как у «Входящих»)
MANUAL_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "TIFF", "AVIF", "MPO"})
SOURCE_LABELS = {
    VariantSource.LOCAL_EDIT: "Локально",
    VariantSource.GEMINI: "Gemini",
    VariantSource.MANUAL: "Вручную",
    VariantSource.UPSCALE: "Апскейл",
}


def load_variant(session: Session, photo_id: int, variant_id: int) -> Variant:
    variant = session.get(Variant, variant_id)
    if variant is None or variant.photo_id != photo_id:
        raise HTTPException(404, "Вариант не найден")
    return variant


def neighbours(
    session: Session, photo_id: int, status: Status = Status.REVIEW
) -> tuple[int | None, int | None]:
    """Предыдущее и следующее фото в статусе `status` (по кругу), без текущего."""
    ids = list(session.scalars(select(Photo.id).where(Photo.status == status).order_by(Photo.id)))
    others = [i for i in ids if i != photo_id]
    if not others:
        return None, None
    before = [i for i in others if i < photo_id]
    after = [i for i in others if i > photo_id]
    return (before[-1] if before else others[-1]), (after[0] if after else others[0])


def variant_tooltip(v: Variant) -> str:
    p = v.params or {}
    parts = [SOURCE_LABELS[v.source], f"раунд {v.round}", f"{v.width}×{v.height}"]
    if p.get("seed") is not None:
        parts.append(f"seed {p['seed']}")
    if p.get("steps"):
        parts.append(f"{p['steps']} шагов")
    if v.source == VariantSource.GEMINI:
        parts.extend(str(p[k]) for k in ("model", "image_size", "aspect_ratio") if p.get(k))
        if p.get("reference"):
            parts.append("с образцом")
        if crop := p.get("cropped"):
            parts.append(f"обрезано {crop['cut']:.0%} под пропорции")
    if v.source == VariantSource.MANUAL and p.get("file"):
        parts.append(f"файл {p['file']}")
    if v.source == VariantSource.UPSCALE:
        if p.get("scale"):
            parts.append(f"×{p['scale']:g}")
        if isinstance(p.get("model"), str):
            parts.append(p["model"])
    if p.get("workflow"):
        parts.append(f"{p['workflow']} v{p.get('workflow_version', '?')}")
    if v.duration_s:
        parts.append(f"{v.duration_s:.0f} с")
    if v.cost_usd is not None and v.source == VariantSource.GEMINI:
        parts.append(f"${v.cost_usd:.3f}")
    text = " · ".join(parts)
    return f"{text}\n\n{v.prompt}" if v.prompt else text


def image_entry(photo: Photo, v: Variant | None, store: Store) -> dict[str, Any]:
    if v is None:
        params = photo.prep_params or {}
        return {
            "key": "prepared",
            "label": "Подготовлено",
            "preview": f"/files/{photo.id}/preview/prepared?v={int(photo.updated_at.timestamp())}",
            "full": f"/files/{photo.id}/prepared",
            "width": params.get("width", photo.width),
            "height": params.get("height", photo.height),
        }
    return {
        "key": f"v{v.id}",
        "id": v.id,
        "label": v.label,
        "badge": SOURCE_LABELS[v.source],
        "round": v.round,
        "preview": f"/files/{photo.id}/preview/{v.label}",
        "full": f"/files/{photo.id}/variant/{v.id}",
        "download": f"/files/{photo.id}/variant/{v.id}?download=1",
        "width": v.width,
        "height": v.height,
        "hidden": v.hidden,
    }


@router.get("/review")
def review_first(session: SessionDep) -> Response:
    first = session.scalar(
        select(Photo.id).where(Photo.status == Status.REVIEW).order_by(Photo.id).limit(1)
    )
    if first is None:
        return RedirectResponse(with_msg("/", "Нет фото на проверке"), status_code=303)
    return RedirectResponse(f"/review/{first}", status_code=303)


@router.get("/review/{photo_id}")
def review_page(
    request: Request,
    session: SessionDep,
    store: StoreDep,
    photo_id: int,
    v: int | None = None,
    hidden: Annotated[bool, Query()] = False,
    msg: str | None = None,
) -> Response:
    photo = load_photo(session, photo_id)
    variants = list(
        session.scalars(
            select(Variant)
            .where(Variant.photo_id == photo.id)
            .order_by(Variant.round.desc(), Variant.id)
        )
    )
    hidden_count = sum(1 for x in variants if x.hidden)
    shown = [x for x in variants if hidden or not x.hidden]
    rounds = [(r, list(group)) for r, group in groupby(shown, key=lambda x: x.round)]
    selected = next((x for x in shown if x.id == v), shown[0] if shown else None)
    prev_id, next_id = neighbours(session, photo.id)
    budget = gemini.budget(session, display_tz(request))
    worker = gemini_worker(request)
    events = session.scalars(
        select(Event).where(Event.photo_id == photo.id).order_by(Event.id.desc())
    ).all()
    ctx = {
        "hdr": header_ctx(request, session),
        "photo": photo,
        "rounds": rounds,
        "numbers": {x.id: i + 1 for i, x in enumerate(shown)},
        "selected": selected,
        "show_hidden": hidden,
        "hidden_count": hidden_count,
        "tooltips": {x.id: variant_tooltip(x) for x in variants},
        "source_labels": SOURCE_LABELS,
        "can_refine": photo.status == Status.REVIEW,
        "accepted_id": photo.accepted_variant_id,
        "has_prepared": store.prepared(photo.id).exists(),
        "addons": presets.addons(session),
        "base_prompt": presets.base_prompt(session),
        "default_variants": presets.edit_prefs(session).variants,
        "max_variants": MAX_VARIANTS,
        "manual_accept": ",".join(f".{ext}" for ext in sorted(ALLOWED_EXT)),
        "gemini": presets.gemini_prefs(session),
        "budget": budget,
        "gemini_block": None if worker.available else "Gemini не настроен: нет GEMINI_API_KEY.",
        "budget_block": budget.check(1),
        "events": events,
        "prev_id": prev_id,
        "next_id": next_id,
        "msg": msg,
        "config": {
            "photoId": photo.id,
            "images": [image_entry(photo, None, store)]
            + [image_entry(photo, x, store) for x in shown],
            "selected": f"v{selected.id}" if selected else None,
            "prevUrl": f"/review/{prev_id}" if prev_id else None,
            "nextUrl": f"/review/{next_id}" if next_id else None,
        },
    }
    return render(request, "review.html", ctx)


@router.post("/review/{photo_id}/refine")
def review_refine(
    session: SessionDep,
    photo_id: int,
    addons: Annotated[list[int] | None, Form()] = None,
    text: Annotated[str, Form(max_length=4000)] = "",
    source: Annotated[str, Form()] = "prepared",
    variant_id: Annotated[int | None, Form()] = None,
    variants: Annotated[int, Form(ge=1, le=MAX_VARIANTS)] = 3,
) -> Response:
    """«Доработать локально»: базовый промпт + добавки + свой текст → QUEUED_EDIT, round+1."""
    photo = load_photo(session, photo_id)
    source_variant = None
    if source == "variant":
        if variant_id is None:
            raise HTTPException(400, "не выбран вариант-источник")
        source_variant = load_variant(session, photo.id, variant_id)
    prompt = presets.compose_prompt(
        presets.base_prompt(session), *presets.addon_texts(session, addons), text
    )
    try:
        states.transition(
            session,
            photo,
            Status.QUEUED_EDIT,
            Actor.OWNER,
            {"source": source_variant.label if source_variant else "prepared"},
        )
    except TransitionError as e:
        return RedirectResponse(with_msg(f"/review/{photo.id}", str(e)), status_code=303)
    jobs.clear_attention(photo)
    jobs.create_edit_job(
        session,
        photo,
        Actor.OWNER,
        prompt=prompt,
        variants=variants,
        source_variant_id=source_variant.id if source_variant else None,
    )
    session.commit()
    msg = f"{photo.code}: доработка в очереди (раунд {photo.round}, вариантов {variants})"
    _, next_id = neighbours(session, photo.id)
    return RedirectResponse(
        with_msg(f"/review/{next_id}" if next_id else "/", msg), status_code=303
    )


@router.post("/review/{photo_id}/gemini")
def review_gemini(
    request: Request,
    session: SessionDep,
    photo_id: int,
    addons: Annotated[list[int] | None, Form()] = None,
    text: Annotated[str, Form(max_length=4000)] = "",
    source: Annotated[str, Form()] = "prepared",
    variant_id: Annotated[int | None, Form()] = None,
    reference: Annotated[bool, Form()] = False,
) -> Response:
    """«В Gemini»: отправляется сразу (без запуска), если хватает бюджета."""
    photo = load_photo(session, photo_id)
    here = f"/review/{photo.id}" + (f"?v={variant_id}" if variant_id else "")
    worker = gemini_worker(request)
    if not worker.available:
        return RedirectResponse(with_msg(here, "Gemini не настроен: нет GEMINI_API_KEY"), 303)
    source_variant = None
    if source == "variant":
        if variant_id is None:
            raise HTTPException(400, "не выбран вариант-источник")
        source_variant = load_variant(session, photo.id, variant_id)
    budget = gemini.budget(session, display_tz(request))
    if (reason := budget.check(1)) is not None:
        return RedirectResponse(with_msg(here, reason), status_code=303)
    user_prompt = presets.compose_prompt(*presets.addon_texts(session, addons), text)
    try:
        job = gemini.send(session, photo, user_prompt, source_variant, reference)
    except TransitionError as e:
        session.rollback()
        return RedirectResponse(with_msg(here, str(e)), status_code=303)
    session.commit()
    worker.wake()
    assert budget.estimate is not None
    cost = gemini.usd(budget.estimate.usd)
    msg = f"{photo.code}: отправлено в Gemini (задача #{job.id}, ≈{cost})"
    _, next_id = neighbours(session, photo.id)
    return RedirectResponse(
        with_msg(f"/review/{next_id}" if next_id else "/", msg), status_code=303
    )


def read_upload(upload: UploadFile) -> Image.Image:
    """Картинка ручной правки: ориентированная, RGB 8 бит."""
    try:
        with Image.open(upload.file) as im:
            if im.format not in MANUAL_FORMATS:
                raise imaging.UnsupportedImage(f"формат {im.format} не поддерживается")
            im.load()
            return imaging.to_rgb8(ImageOps.exif_transpose(im))
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        raise imaging.UnsupportedImage(str(e) or type(e).__name__) from e


@router.post("/review/{photo_id}/manual")
def review_manual(
    session: SessionDep,
    store: StoreDep,
    photo_id: int,
    file: Annotated[UploadFile, File()],
    variant_id: Annotated[int | None, Form()] = None,
) -> Response:
    """«Загрузить правку»: файл из редактора — новый вариант `manual` от выбранного."""
    photo = load_photo(session, photo_id)
    parent = load_variant(session, photo.id, variant_id) if variant_id else None
    here = f"/review/{photo.id}" + (f"?v={parent.id}" if parent else "")
    if not states.can_transition(photo.status, Status.REVIEW, Actor.OWNER):
        return RedirectResponse(with_msg(here, f"{photo.code}: фото не на проверке"), 303)
    try:
        img = read_upload(file)
    except imaging.UnsupportedImage as e:
        return RedirectResponse(with_msg(here, f"Файл не принят: {e}"), status_code=303)
    variant = results.add_variant(
        session,
        store,
        photo,
        lambda tmp: imaging.save_png(img, tmp),
        image=img,
        parent_id=parent.id if parent else None,
        source=VariantSource.MANUAL,
        round=photo.round,
        params={"file": file.filename or ""},
    )
    session.flush()
    states.transition(
        session,
        photo,
        Status.REVIEW,
        Actor.OWNER,
        {"manual": variant.label, "parent": parent.label if parent else "prepared"},
    )
    jobs.clear_attention(photo)
    session.commit()
    msg = f"{photo.code}: правка загружена как {variant.label}"
    return RedirectResponse(with_msg(f"/review/{photo.id}?v={variant.id}", msg), 303)


@router.post("/review/{photo_id}/accept")
def review_accept(
    session: SessionDep,
    photo_id: int,
    variant_id: Annotated[int, Form()],
    upscale: Annotated[bool, Form()] = True,
) -> Response:
    """«Принять» (→ апскейл в очереди) или «Принять без апскейла» (→ финал)."""
    photo = load_photo(session, photo_id)
    variant = load_variant(session, photo.id, variant_id)
    try:
        if upscale:
            final.accept(session, photo, variant)
            msg = f"{photo.code}: принят {variant.label}, апскейл в очереди"
        else:
            final.accept_without_upscale(session, photo, variant)
            msg = f"{photo.code}: принят {variant.label} без апскейла — на финале"
    except (TransitionError, final.FinalError) as e:
        session.rollback()
        return RedirectResponse(with_msg(f"/review/{photo.id}?v={variant.id}", str(e)), 303)
    session.commit()
    _, next_id = neighbours(session, photo.id)
    return RedirectResponse(
        with_msg(f"/review/{next_id}" if next_id else "/", msg), status_code=303
    )


@router.post("/variants/{variant_id}/{action}")
def variant_visibility(session: SessionDep, variant_id: int, action: str) -> Response:
    if action not in ("hide", "show"):
        raise HTTPException(404)
    variant = session.get(Variant, variant_id)
    if variant is None:
        raise HTTPException(404, "Вариант не найден")
    variant.hidden = action == "hide"
    states.log_event(
        session, variant.photo_id, Actor.OWNER, f"variant_{action}", {"variant": variant.label}
    )
    session.commit()
    return RedirectResponse(f"/review/{variant.photo_id}?v={variant.id}", status_code=303)


@router.get("/files/{photo_id}/variant/{variant_id}")
def file_variant(
    session: SessionDep,
    store: StoreDep,
    photo_id: int,
    variant_id: int,
    download: Annotated[bool, Query()] = False,
) -> Response:
    photo = load_photo(session, photo_id)
    variant = load_variant(session, photo.id, variant_id)
    return _file(
        store.photo_dir(photo.id) / variant.path,
        media_type="image/png",
        filename=f"{photo.code}_{variant.label}.png" if download else None,
        headers=_CACHE,
    )


@router.get("/files/{photo_id}/variant/{variant_id}/thumb")
def file_variant_thumb(
    session: SessionDep, store: StoreDep, photo_id: int, variant_id: int
) -> Response:
    variant = load_variant(session, photo_id, variant_id)
    return _file(
        store.variant_thumb(photo_id, variant.number), media_type="image/webp", headers=_CACHE
    )
