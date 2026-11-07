"""Экран проверки (ТЗ §5.4): лента вариантов, сравнение, доработка, «Скрыть», «Принять»."""

from itertools import groupby
from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import final, jobs, presets, states
from app.api.ui import (
    _CACHE,
    SessionDep,
    StoreDep,
    _file,
    header_ctx,
    load_photo,
    render,
    with_msg,
)
from app.models import Actor, Event, Photo, Preset, PresetKind, Status, Variant, VariantSource
from app.states import TransitionError
from app.storage import Store

router = APIRouter()

MAX_VARIANTS = 6
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
    if v.source == VariantSource.UPSCALE:
        if p.get("scale"):
            parts.append(f"×{p['scale']:g}")
        if isinstance(p.get("model"), str):
            parts.append(p["model"])
    if p.get("workflow"):
        parts.append(f"{p['workflow']} v{p.get('workflow_version', '?')}")
    if v.duration_s:
        parts.append(f"{v.duration_s:.0f} с")
    if v.cost_usd:
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
    addons = session.scalars(
        select(Preset).where(Preset.kind == PresetKind.ADDON).order_by(Preset.sort, Preset.id)
    ).all()
    events = session.scalars(
        select(Event).where(Event.photo_id == photo.id).order_by(Event.id.desc())
    ).all()
    ctx = {
        "hdr": header_ctx(session),
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
        "addons": addons,
        "base_prompt": presets.base_prompt(session),
        "default_variants": int(presets.get_setting(session, "edit")["variants"]),
        "max_variants": MAX_VARIANTS,
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


def compose_prompt(base: str, addons: list[str], text: str) -> str:
    parts = [base.strip(), *(a.strip() for a in addons), text.strip()]
    return "\n".join(p for p in parts if p)


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
    addon_texts = (
        session.scalars(
            select(Preset.text).where(Preset.id.in_(addons), Preset.kind == PresetKind.ADDON)
        ).all()
        if addons
        else []
    )
    prompt = compose_prompt(presets.base_prompt(session), list(addon_texts), text)
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
