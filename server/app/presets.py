"""Пресеты промптов и пользовательские настройки (таблица `setting`).

Настройка — JSON под ключом; читается через схему ниже. Ключи, которых нет в сохранённом
значении (появились в новой версии), берутся из значений по умолчанию.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Preset, PresetKind, Setting

log = logging.getLogger(__name__)

BASE_PROMPT = (
    "Restore this old photograph. Remove scratches, creases, dust, stains, tears and print "
    "artifacts.\n"
    "Reduce grain while keeping natural photographic texture. Improve sharpness and contrast.\n"
    "Keep the faces, facial features, expressions, hair, clothing, poses, composition and "
    "framing exactly the same.\n"
    "Keep the original black-and-white tonality, do not colorize. Do not add or remove "
    "anything."
)

# Чипы-добавки к промпту (редактируются на странице настроек)
DEFAULT_ADDONS = [
    (
        "Сильнее чистка",
        "Remove damage more aggressively: deep scratches, cracks, creases, "
        "missing emulsion and large stains.",
    ),
    ("Сохранить зерно", "Keep the original film grain and paper texture; do not smooth the image."),
    ("Лица чётче", "Make faces slightly sharper and clearer without changing any facial features."),
    ("Мягкий контраст", "Keep contrast gentle: do not crush shadows or blow out highlights."),
]

MAX_VARIANTS = 6
GEMINI_SIZES = ("1K", "2K", "4K")
GeminiSize = Literal["1K", "2K", "4K"]


class EditPrefs(BaseModel):
    """`edit_local` по умолчанию: число вариантов, шаги, размер входа модели."""

    variants: int = Field(default=3, ge=1, le=MAX_VARIANTS)
    steps: int = Field(default=8, ge=1, le=50)
    megapixels: float = Field(default=1.5, ge=0.3, le=4.0)


class UpscalePrefs(BaseModel):
    """`upscale_local` по умолчанию: модель (ключ `choices` реестра), цель и предел масштаба."""

    model: str = "7b_fp16"
    target_mp: float = Field(default=4.0, ge=1.0, le=40.0)
    max_scale: float = Field(default=2.0, ge=1.0, le=4.0)


class GeminiPrefs(BaseModel):
    model: str = Field(default="gemini-3-pro-image", pattern=r"^[a-z0-9][a-z0-9.\-]{1,62}$")
    image_size: GeminiSize = "2K"
    monthly_budget_usd: float = Field(default=20.0, ge=0, le=10_000)
    max_concurrency: int = Field(default=2, ge=1, le=2)  # ТЗ §8: не больше 2 одновременных


class Price(BaseModel):
    """Цены модели Gemini, $ за 1 млн токенов."""

    input: float = Field(ge=0, le=1000)  # вход: текст и картинки
    output_text: float = Field(ge=0, le=1000)  # выход: текст и мышление
    output_image: float = Field(ge=0, le=1000)  # выход: картинки


# Платный уровень Gemini Developer API, ноябрь 2026 (ai.google.dev/gemini-api/docs/pricing)
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "gemini-3-pro-image": {"input": 2.0, "output_text": 12.0, "output_image": 120.0},
    "gemini-3.1-flash-image": {"input": 0.5, "output_text": 3.0, "output_image": 60.0},
    "gemini-nano-banana-2.1": {"input": 1.5, "output_text": 7.5, "output_image": 30.0},
}

SCHEMAS: dict[str, type[BaseModel]] = {
    "edit": EditPrefs,
    "upscale": UpscalePrefs,
    "gemini": GeminiPrefs,
}

DEFAULT_SETTINGS: dict[str, Any] = {
    **{key: schema().model_dump() for key, schema in SCHEMAS.items()},
    "gemini_prices": DEFAULT_PRICES,
}


def ensure_defaults(session: Session) -> None:
    """Досоздаёт отсутствующие настройки и пресеты; существующие не трогает."""
    for key, value in DEFAULT_SETTINGS.items():
        if session.get(Setting, key) is None:
            session.add(Setting(key=key, value=value))
    has_base = session.scalar(select(Preset.id).where(Preset.kind == PresetKind.BASE).limit(1))
    if has_base is None:
        session.add(Preset(kind=PresetKind.BASE, name="Базовый", text=BASE_PROMPT, sort=0))
    has_addon = session.scalar(select(Preset.id).where(Preset.kind == PresetKind.ADDON).limit(1))
    if has_addon is None:
        for i, (name, text) in enumerate(DEFAULT_ADDONS):
            session.add(Preset(kind=PresetKind.ADDON, name=name, text=text, sort=i))


def get_setting(session: Session, key: str) -> Any:
    """Сохранённое значение; у словаря недостающие ключи — из значений по умолчанию."""
    row = session.get(Setting, key)
    default = DEFAULT_SETTINGS.get(key)
    if row is None:
        return default
    if key in SCHEMAS and isinstance(row.value, dict) and isinstance(default, dict):
        return {**default, **row.value}
    return row.value


def set_setting(session: Session, key: str, value: Any) -> None:
    row = session.get(Setting, key)
    if row is None:
        session.add(Setting(key=key, value=value))
    else:
        row.value = value


def _prefs[T: BaseModel](session: Session, key: str, schema: type[T]) -> T:
    try:
        return schema.model_validate(get_setting(session, key))
    except ValidationError:
        log.exception("настройка %s повреждена — используются значения по умолчанию", key)
        return schema()


def edit_prefs(session: Session) -> EditPrefs:
    return _prefs(session, "edit", EditPrefs)


def upscale_prefs(session: Session) -> UpscalePrefs:
    return _prefs(session, "upscale", UpscalePrefs)


def gemini_prefs(session: Session) -> GeminiPrefs:
    return _prefs(session, "gemini", GeminiPrefs)


def gemini_prices(session: Session) -> dict[str, Price]:
    raw = get_setting(session, "gemini_prices") or {}
    prices = {}
    for model, value in raw.items():
        try:
            prices[model] = Price.model_validate(value)
        except ValidationError:
            log.warning("цены модели %s повреждены — пропущены", model)
    return prices


def base_preset(session: Session) -> Preset | None:
    return session.scalar(
        select(Preset)
        .where(Preset.kind == PresetKind.BASE)
        .order_by(Preset.sort, Preset.id)
        .limit(1)
    )


def base_prompt(session: Session) -> str:
    preset = base_preset(session)
    return preset.text if preset is not None and preset.text.strip() else BASE_PROMPT


def addons(session: Session) -> list[Preset]:
    return list(
        session.scalars(
            select(Preset).where(Preset.kind == PresetKind.ADDON).order_by(Preset.sort, Preset.id)
        )
    )


def addon_texts(session: Session, ids: list[int] | None) -> list[str]:
    """Тексты выбранных добавок в порядке пресетов (порядок галочек не важен)."""
    if not ids:
        return []
    return [p.text for p in addons(session) if p.id in set(ids)]


def compose_prompt(*parts: str) -> str:
    return "\n".join(p.strip() for p in parts if p and p.strip())
