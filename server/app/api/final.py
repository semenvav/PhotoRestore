"""Экран финала (ТЗ §5.5): «принятый вариант / апскейл», утверждение и отдача."""

from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import delivery, final, jobs, presets
from app.api.review import image_entry, neighbours, variant_tooltip
from app.api.ui import SessionDep, StoreDep, header_ctx, load_photo, render, with_msg
from app.config import Settings
from app.models import Event, JobType, Photo, Status, VariantSource
from app.states import TransitionError
from app.workflows import Registry

router = APIRouter()

AUTO = "auto"


def model_label(key: str) -> str:
    """`7b_fp16` → «7B fp16»."""
    size, *rest = key.split("_")
    return " ".join([size.upper(), *rest])


def model_choices(request: Request) -> dict[str, str]:
    registry: Registry = request.app.state.registry
    wf = registry.for_type(JobType.UPSCALE_LOCAL)
    return {key: model_label(key) for key in (wf.choices("model") if wf else {})}


def scale_label(scale: float) -> str:
    return f"×{scale:g}".replace(".", ",")


def next_final(session: Session, photo_id: int) -> str:
    _, next_id = neighbours(session, photo_id, Status.FINAL_REVIEW)
    return f"/final/{next_id}" if next_id else "/"


@router.get("/final")
def final_first(session: SessionDep) -> Response:
    first = session.scalar(
        select(Photo.id).where(Photo.status == Status.FINAL_REVIEW).order_by(Photo.id).limit(1)
    )
    if first is None:
        return RedirectResponse(with_msg("/", "Нет фото на финале"), status_code=303)
    return RedirectResponse(f"/final/{first}", status_code=303)


@router.get("/final/{photo_id}")
def final_page(
    request: Request,
    session: SessionDep,
    store: StoreDep,
    photo_id: int,
    v: int | None = None,
    msg: str | None = None,
) -> Response:
    photo = load_photo(session, photo_id)
    tiles = final.candidates(session, photo)
    by_id = {t.id: t for t in tiles}
    selected = by_id.get(v or 0) or by_id.get(photo.final_variant_id or 0)
    selected = selected or (tiles[0] if tiles else None)
    accepted_key = f"v{photo.accepted_variant_id}" if photo.accepted_variant_id else None
    prev_id, next_id = neighbours(session, photo.id, Status.FINAL_REVIEW)
    up = presets.upscale_prefs(session)
    events = session.scalars(
        select(Event).where(Event.photo_id == photo.id).order_by(Event.id.desc())
    ).all()
    ctx: dict[str, Any] = {
        "hdr": header_ctx(request, session),
        "photo": photo,
        "tiles": tiles,
        "numbers": {t.id: i + 1 for i, t in enumerate(tiles)},
        "selected": selected,
        "tooltips": {t.id: variant_tooltip(t) for t in tiles},
        "upscale": VariantSource.UPSCALE,
        "can_act": photo.status == Status.FINAL_REVIEW,
        "models": model_choices(request),
        "default_model": up.model,
        "scales": [(str(s), scale_label(s)) for s in jobs.UPSCALE_SCALES],
        "auto_label": f"авто: до {up.target_mp:g} Мп, не больше {scale_label(up.max_scale)}",
        "events": events,
        "prev_id": prev_id,
        "next_id": next_id,
        "msg": msg,
        "config": {
            "photoId": photo.id,
            "images": [image_entry(photo, None, store)]
            + [image_entry(photo, t, store) for t in tiles],
            "selected": f"v{selected.id}" if selected else None,
            "left": accepted_key,
            "prevUrl": f"/final/{prev_id}" if prev_id else None,
            "nextUrl": f"/final/{next_id}" if next_id else None,
        },
    }
    return render(request, "final.html", ctx)


@router.post("/final/{photo_id}/approve")
def final_approve(
    request: Request,
    session: SessionDep,
    store: StoreDep,
    photo_id: int,
    variant_id: Annotated[int | None, Form()] = None,
) -> Response:
    photo = load_photo(session, photo_id)
    settings: Settings = request.app.state.settings
    try:
        final.approve(
            session, store, settings.delivered_dir, photo, settings.display_tz, variant_id
        )
    except (final.FinalError, delivery.DeliveryError, TransitionError) as e:
        session.rollback()
        return RedirectResponse(with_msg(f"/final/{photo_id}", str(e)), status_code=303)
    session.commit()
    msg = f"{photo.code} утверждено → «Готово»/{photo.delivered_path}"
    return RedirectResponse(with_msg(next_final(session, photo.id), msg), status_code=303)


@router.post("/final/{photo_id}/redo")
def final_redo(
    request: Request,
    session: SessionDep,
    photo_id: int,
    model: Annotated[str, Form()],
    scale: Annotated[str, Form()] = AUTO,
) -> Response:
    photo = load_photo(session, photo_id)
    if model not in model_choices(request):
        raise HTTPException(400, f"неизвестная модель {model!r}")
    value: float | None = None
    if scale != AUTO:
        try:
            value = float(scale)
        except ValueError:
            value = None
        if value not in jobs.UPSCALE_SCALES:
            raise HTTPException(400, f"масштаб {scale!r} не из списка")
    try:
        job = final.redo_upscale(session, photo, model, value)
    except (final.FinalError, TransitionError) as e:
        session.rollback()
        return RedirectResponse(with_msg(f"/final/{photo_id}", str(e)), status_code=303)
    session.commit()
    params = job.payload["params"]
    msg = (
        f"{photo.code}: апскейл в очереди ({model_label(params['model'])}, "
        f"{scale_label(params['scale'])})"
    )
    return RedirectResponse(with_msg(next_final(session, photo.id), msg), status_code=303)


@router.post("/final/{photo_id}/no-upscale")
def final_no_upscale(session: SessionDep, photo_id: int) -> Response:
    photo = load_photo(session, photo_id)
    try:
        base = final.without_upscale(session, photo)
    except final.FinalError as e:
        return RedirectResponse(with_msg(f"/final/{photo_id}", str(e)), status_code=303)
    session.commit()
    return RedirectResponse(
        with_msg(f"/final/{photo_id}?v={base.id}", f"Финал — {base.label} без апскейла"),
        status_code=303,
    )


@router.post("/final/{photo_id}/back")
def final_back(session: SessionDep, photo_id: int) -> Response:
    photo = load_photo(session, photo_id)
    try:
        final.back_to_variants(session, photo)
    except TransitionError as e:
        return RedirectResponse(with_msg(f"/final/{photo_id}", str(e)), status_code=303)
    session.commit()
    return RedirectResponse(f"/review/{photo_id}", status_code=303)
