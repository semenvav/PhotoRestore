"""Операции с изображениями: проверка, ориентация, превью, подготовка (кроп и поворот)."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from pydantic import BaseModel, Field, field_validator

THUMB_PX = 400
PREVIEW_PX = 2048
PREVIEW_QUALITY = 90
FILL = (255, 255, 255)
SIMILAR_MAX_DISTANCE = 6


class UnsupportedImage(Exception):
    pass


def to_rgb8(img: Image.Image) -> Image.Image:
    """RGB, 8 бит на канал. 16-битные сканы масштабируются, а не обрезаются."""
    if img.mode.startswith("I"):
        arr = np.asarray(img).astype(np.int64)
        if img.mode.startswith("I;16") or arr.max(initial=0) > 255:
            arr = arr >> 8
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "L")
    elif img.mode == "F":
        arr = np.asarray(img)
        scale = 255.0 if arr.max(initial=0) <= 1.0 else 1.0
        img = Image.fromarray((arr * scale).clip(0, 255).astype(np.uint8), "L")

    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, FILL)
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img if img.mode == "RGB" else img.convert("RGB")


def load_oriented(path: Path) -> Image.Image:
    """Открывает файл, применяет EXIF-ориентацию и приводит к RGB 8 бит."""
    try:
        with Image.open(path) as im:
            im.load()
            oriented = ImageOps.exif_transpose(im)
            return to_rgb8(oriented)
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        raise UnsupportedImage(str(e) or type(e).__name__) from e


@dataclass
class Inspected:
    image: Image.Image  # ориентированное RGB
    phash: str


def inspect(path: Path) -> Inspected:
    img = load_oriented(path)
    return Inspected(image=img, phash=str(imagehash.phash(img)))


def phash_distance(a: str, b: str) -> int:
    return int(imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b))


def _tmp(dest: Path) -> Path:
    return dest.with_name(dest.name + ".tmp")


def save_thumb(img: Image.Image, dest: Path) -> None:
    thumb = img.copy()
    thumb.thumbnail((THUMB_PX, THUMB_PX), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(_tmp(dest), format="WEBP", quality=80, method=4)
    os.replace(_tmp(dest), dest)


def save_preview(img: Image.Image, dest: Path) -> None:
    preview = img.copy()
    preview.thumbnail((PREVIEW_PX, PREVIEW_PX), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    preview.save(_tmp(dest), format="JPEG", quality=PREVIEW_QUALITY, optimize=True)
    os.replace(_tmp(dest), dest)


def save_png(img: Image.Image, dest: Path, text: dict[str, str] | None = None) -> None:
    """Пишет PNG как есть; атомарность (временный файл + os.replace) — на вызывающем."""
    info = PngInfo()
    for key, value in (text or {}).items():
        info.add_text(key, value)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="PNG", compress_level=4, pnginfo=info)


# --- Подготовка -----------------------------------------------------------------------


class Crop(BaseModel):
    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class PrepParams(BaseModel):
    """Параметры редактора (cropper.js 1.x).

    Порядок: EXIF-транспонирование → `rotate90` четвертей по часовой → поворот на `angle`
    градусов по часовой вокруг центра. `crop` задан в пикселях полного разрешения внутри
    описанного прямоугольника повёрнутого изображения — ровно как `cropper.getData()`.
    `crop = None` — без обрезки (весь описанный прямоугольник).
    """

    rotate90: int = 0
    angle: float = Field(default=0.0, ge=-10.0, le=10.0)
    crop: Crop | None = None

    @field_validator("rotate90")
    @classmethod
    def _norm_quarters(cls, v: int) -> int:
        return v % 4


_QUARTER = {
    1: Image.Transpose.ROTATE_270,  # PIL вращает против часовой
    2: Image.Transpose.ROTATE_180,
    3: Image.Transpose.ROTATE_90,
}


def rotated_bbox(width: float, height: float, angle: float) -> tuple[float, float]:
    """Размер описанного прямоугольника, как `getRotatedSizes` в cropper.js."""
    arc = math.radians(abs(angle))
    sin, cos = math.sin(arc), math.cos(arc)
    return width * cos + height * sin, width * sin + height * cos


def oriented_frame(width: int, height: int, params: PrepParams) -> tuple[float, float]:
    """Размер кадра, в котором задан `crop`, для ориентированного изображения width×height."""
    w, h = (height, width) if params.rotate90 % 2 else (width, height)
    return rotated_bbox(w, h, params.angle)


def prepare(img: Image.Image, params: PrepParams) -> Image.Image:
    """Кроп и поворот на полном разрешении за одно пересэмплирование."""
    if params.rotate90:
        img = img.transpose(_QUARTER[params.rotate90])
    w1, h1 = img.size
    w2, h2 = rotated_bbox(w1, h1, params.angle)

    crop = params.crop or Crop(x=0, y=0, w=w2, h=h2)
    x0, y0 = max(0.0, crop.x), max(0.0, crop.y)
    x1, y1 = min(w2, crop.x + crop.w), min(h2, crop.y + crop.h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError("рамка вне изображения")

    if params.angle == 0:
        box = (round(x0), round(y0), round(x1), round(y1))
        return img.crop(box)

    out_w, out_h = max(1, round(x1 - x0)), max(1, round(y1 - y0))
    theta = math.radians(params.angle)
    cos, sin = math.cos(theta), math.sin(theta)
    # Обратное отображение: точка выхода (u, v) → кадр (x0+u, y0+v) → центр кадра →
    # поворот против часовой на angle → координаты источника.
    ox, oy = x0 - w2 / 2, y0 - h2 / 2
    coeffs = (
        cos,
        sin,
        cos * ox + sin * oy + w1 / 2,
        -sin,
        cos,
        -sin * ox + cos * oy + h1 / 2,
    )
    return img.transform(
        (out_w, out_h),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=FILL,
    )
