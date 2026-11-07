"""Пресеты промптов и пользовательские настройки по умолчанию."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Preset, PresetKind, Setting

BASE_PROMPT = (
    "Restore this old photograph. Remove scratches, creases, dust, stains, tears and print "
    "artifacts.\n"
    "Reduce grain while keeping natural photographic texture. Improve sharpness and contrast.\n"
    "Keep the faces, facial features, expressions, hair, clothing, poses, composition and "
    "framing exactly the same.\n"
    "Keep the original black-and-white tonality, do not colorize. Do not add or remove "
    "anything."
)

# Чипы-добавки к промпту для «Доработать локально» (редактор пресетов — в M4)
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

DEFAULT_SETTINGS: dict[str, Any] = {
    "edit": {"variants": 3, "steps": 8, "megapixels": 1.5},
    "upscale": {"model": "7b_fp16", "target_mp": 4.0, "max_scale": 2.0},
    "gemini": {
        "model": "gemini-3-pro-image",
        "image_size": "2K",
        "monthly_budget_usd": 20.0,
        "max_concurrency": 2,
    },
}


def ensure_defaults(session: Session) -> None:
    """Досоздаёт отсутствующие настройки и базовый пресет; существующие не трогает."""
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
    row = session.get(Setting, key)
    return row.value if row is not None else DEFAULT_SETTINGS.get(key)


def base_prompt(session: Session) -> str:
    text = session.scalar(
        select(Preset.text)
        .where(Preset.kind == PresetKind.BASE)
        .order_by(Preset.sort, Preset.id)
        .limit(1)
    )
    return text or BASE_PROMPT
