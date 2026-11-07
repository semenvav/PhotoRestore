"""Страница настроек (ТЗ §5.7): параметры edit/upscale по умолчанию, Gemini (модель, размер,
бюджет, цены), пресеты промптов. Значения — в таблице `setting` и `preset`."""

from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails
from sqlalchemy import func, select

from app import gemini, presets
from app.api.final import model_choices
from app.api.ui import SessionDep, display_tz, gemini_worker, header_ctx, render
from app.models import Preset, PresetKind

router = APIRouter()

FIELD_LABELS = {
    "variants": "вариантов",
    "steps": "шагов",
    "megapixels": "мегапикселей",
    "model": "модель",
    "target_mp": "цель, Мп",
    "max_scale": "предел масштаба",
    "image_size": "размер",
    "monthly_budget_usd": "бюджет",
    "max_concurrency": "одновременных запросов",
    "input": "вход",
    "output_text": "текст и мышление",
    "output_image": "картинка",
    "name": "название",
    "text": "текст",
}
NAME_MAX = 100
TEXT_MAX = 4000


def back(section: str, msg: str) -> Response:
    return RedirectResponse(f"/settings?msg={quote(msg)}#{section}", status_code=303)


def _error(err: ErrorDetails) -> str:
    field = FIELD_LABELS.get(str(err["loc"][-1]), str(err["loc"][-1])) if err["loc"] else ""
    ctx = err.get("ctx") or {}
    match err["type"]:
        case "greater_than_equal":
            text = f"не меньше {ctx.get('ge')}"
        case "less_than_equal":
            text = f"не больше {ctx.get('le')}"
        case "int_parsing" | "float_parsing" | "int_from_float":
            text = "нужно число"
        case "string_pattern_mismatch":
            text = "недопустимые символы (латиница, цифры, точка, дефис)"
        case "literal_error":
            text = "недопустимое значение"
        case "missing":
            text = "не заполнено"
        case _:
            text = err["msg"]
    return f"{field}: {text}" if field else text


def errors_text(e: ValidationError) -> str:
    return "Не сохранено — " + "; ".join(_error(err) for err in e.errors())


def save_prefs(session: SessionDep, key: str, schema: type[BaseModel], data: dict[str, Any]) -> str:
    try:
        value = schema.model_validate(data)
    except ValidationError as e:
        return errors_text(e)
    presets.set_setting(session, key, value.model_dump())
    session.commit()
    return "Сохранено"


@router.get("/settings")
def settings_page(request: Request, session: SessionDep, msg: str | None = None) -> Response:
    prefs = presets.gemini_prefs(session)
    prices = presets.gemini_prices(session)
    models = list(dict.fromkeys([*presets.DEFAULT_PRICES, *prices, prefs.model]))
    worker = gemini_worker(request)
    ctx = {
        "hdr": header_ctx(request, session),
        "msg": msg,
        "edit": presets.edit_prefs(session),
        "upscale": presets.upscale_prefs(session),
        "upscale_models": model_choices(request),
        "gemini": prefs,
        "prices": prices,
        "price_rows": [(m, prices.get(m)) for m in models if m in prices or m == prefs.model],
        "known_models": models,
        "sizes": presets.GEMINI_SIZES,
        "budget": gemini.budget(session, display_tz(request)),
        "gemini_state": "fake" if worker.fake else ("on" if worker.available else "off"),
        "base": presets.base_preset(session),
        "default_base": presets.BASE_PROMPT,
        "addons": presets.addons(session),
        "max_variants": presets.MAX_VARIANTS,
        "name_max": NAME_MAX,
        "text_max": TEXT_MAX,
    }
    return render(request, "settings.html", ctx)


@router.post("/settings/edit")
def save_edit(
    session: SessionDep,
    variants: Annotated[str, Form()],
    steps: Annotated[str, Form()],
    megapixels: Annotated[str, Form()],
) -> Response:
    data = {"variants": variants, "steps": steps, "megapixels": megapixels.replace(",", ".")}
    return back("edit", save_prefs(session, "edit", presets.EditPrefs, data))


@router.post("/settings/upscale")
def save_upscale(
    request: Request,
    session: SessionDep,
    model: Annotated[str, Form()],
    target_mp: Annotated[str, Form()],
    max_scale: Annotated[str, Form()],
) -> Response:
    if model not in model_choices(request):
        return back("upscale", f"Не сохранено — нет модели {model!r} в workflow апскейла")
    data = {
        "model": model,
        "target_mp": target_mp.replace(",", "."),
        "max_scale": max_scale.replace(",", "."),
    }
    return back("upscale", save_prefs(session, "upscale", presets.UpscalePrefs, data))


@router.post("/settings/gemini")
def save_gemini(
    session: SessionDep,
    model: Annotated[str, Form()],
    image_size: Annotated[str, Form()],
    monthly_budget_usd: Annotated[str, Form()],
    max_concurrency: Annotated[str, Form()],
) -> Response:
    data = {
        "model": model.strip(),
        "image_size": image_size,
        "monthly_budget_usd": monthly_budget_usd.replace(",", "."),
        "max_concurrency": max_concurrency,
    }
    msg = save_prefs(session, "gemini", presets.GeminiPrefs, data)
    if msg == "Сохранено" and data["model"] not in presets.gemini_prices(session):
        msg += f". Задайте цены для {data['model']} — без них отправка заблокирована"
    return back("gemini", msg)


@router.post("/settings/prices")
def save_prices(
    session: SessionDep,
    model: Annotated[list[str] | None, Form()] = None,
    input: Annotated[list[str] | None, Form()] = None,
    output_text: Annotated[list[str] | None, Form()] = None,
    output_image: Annotated[list[str] | None, Form()] = None,
    delete: Annotated[list[str] | None, Form()] = None,
    new_model: Annotated[str, Form()] = "",
    new_input: Annotated[str, Form()] = "",
    new_output_text: Annotated[str, Form()] = "",
    new_output_image: Annotated[str, Form()] = "",
) -> Response:
    rows = list(zip(model or [], input or [], output_text or [], output_image or [], strict=False))
    if new_model.strip():
        rows.append((new_model.strip(), new_input, new_output_text, new_output_image))
    current = presets.gemini_prefs(session).model
    removed = set(delete or [])
    if current in removed:
        return back("prices", f"Не сохранено — {current} выбрана в настройках Gemini")
    prices: dict[str, dict[str, float]] = {}
    for name, inp, text, image in rows:
        if name in removed:
            continue
        if not _valid_model(name):
            return back("prices", f"Не сохранено — недопустимое имя модели {name!r}")
        try:
            price = presets.Price.model_validate(
                {
                    "input": inp.replace(",", "."),
                    "output_text": text.replace(",", "."),
                    "output_image": image.replace(",", "."),
                }
            )
        except ValidationError as e:
            return back("prices", f"{errors_text(e)} ({name})")
        prices[name] = price.model_dump()
    presets.set_setting(session, "gemini_prices", prices)
    session.commit()
    return back("prices", "Цены сохранены")


def _valid_model(name: str) -> bool:
    try:
        presets.GeminiPrefs.model_validate({"model": name})
    except ValidationError:
        return False
    return True


# --- пресеты --------------------------------------------------------------------------


def _clean(name: str, text: str) -> tuple[str, str] | str:
    name, text = name.strip(), text.strip()
    if not name:
        return "Не сохранено — название: не заполнено"
    if not text:
        return "Не сохранено — текст: не заполнено"
    if len(name) > NAME_MAX or len(text) > TEXT_MAX:
        return f"Не сохранено — название до {NAME_MAX}, текст до {TEXT_MAX} символов"
    return name, text


@router.post("/settings/presets/base")
def save_base(
    session: SessionDep,
    text: Annotated[str, Form()] = "",
    reset: Annotated[bool, Form()] = False,
) -> Response:
    if reset:
        text = presets.BASE_PROMPT
    cleaned = _clean("Базовый", text)
    if isinstance(cleaned, str):
        return back("presets", cleaned)
    preset = presets.base_preset(session)
    if preset is None:
        session.add(Preset(kind=PresetKind.BASE, name="Базовый", text=cleaned[1], sort=0))
    else:
        preset.text = cleaned[1]
    session.commit()
    return back("presets", "Базовый промпт сброшен" if reset else "Базовый промпт сохранён")


@router.post("/settings/presets/addons")
def add_addon(
    session: SessionDep,
    name: Annotated[str, Form()] = "",
    text: Annotated[str, Form()] = "",
) -> Response:
    cleaned = _clean(name, text)
    if isinstance(cleaned, str):
        return back("presets", cleaned)
    top = session.scalar(select(func.max(Preset.sort)).where(Preset.kind == PresetKind.ADDON))
    session.add(
        Preset(kind=PresetKind.ADDON, name=cleaned[0], text=cleaned[1], sort=(top or 0) + 1)
    )
    session.commit()
    return back("presets", f"Добавка «{cleaned[0]}» добавлена")


def _addon(session: SessionDep, preset_id: int) -> Preset | None:
    preset = session.get(Preset, preset_id)
    return preset if preset is not None and preset.kind == PresetKind.ADDON else None


@router.post("/settings/presets/{preset_id}")
def save_addon(
    session: SessionDep,
    preset_id: int,
    name: Annotated[str, Form()] = "",
    text: Annotated[str, Form()] = "",
) -> Response:
    preset = _addon(session, preset_id)
    if preset is None:
        return back("presets", "Добавка не найдена")
    cleaned = _clean(name, text)
    if isinstance(cleaned, str):
        return back("presets", cleaned)
    preset.name, preset.text = cleaned
    session.commit()
    return back("presets", f"Добавка «{preset.name}» сохранена")


@router.post("/settings/presets/{preset_id}/{action}")
def addon_action(session: SessionDep, preset_id: int, action: str) -> Response:
    preset = _addon(session, preset_id)
    if preset is None or action not in ("delete", "up", "down"):
        return back("presets", "Добавка не найдена")
    if action == "delete":
        session.delete(preset)
        session.commit()
        return back("presets", f"Добавка «{preset.name}» удалена")
    order = presets.addons(session)
    i = next(i for i, p in enumerate(order) if p.id == preset.id)
    j = i - 1 if action == "up" else i + 1
    if 0 <= j < len(order):
        order[i], order[j] = order[j], order[i]
        for n, p in enumerate(order):  # заодно выравнивает номера
            p.sort = n
        session.commit()
    return back("presets", "Порядок изменён")
