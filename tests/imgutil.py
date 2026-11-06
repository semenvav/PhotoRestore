"""Генерация тестовых изображений."""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

EXIF_ORIENTATION = 0x0112


def gradient(width: int, height: int, seed: int = 0, noise: bool = True) -> Image.Image:
    """RGB-картинка, у которой каждый пиксель различим: R ~ x, G ~ y, B — шум
    (без шума — плавный, чтобы JPEG сжимал почти без потерь)."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 255, width, dtype=np.float64)[None, :].repeat(height, 0)
    ys = np.linspace(0, 255, height, dtype=np.float64)[:, None].repeat(width, 1)
    noise_ch = rng.integers(0, 256, (height, width)) if noise else np.full((height, width), seed)
    arr = np.stack([xs, ys, noise_ch], axis=-1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def scene(width: int, height: int, seed: int = 0) -> Image.Image:
    """«Снимок» из размытых фигур: у градиентов pHash неустойчив, у такого — нет."""
    rng = np.random.default_rng(seed)
    img = Image.new("L", (width, height), 128)
    draw = ImageDraw.Draw(img)
    for _ in range(12):
        x0, y0 = rng.integers(0, width), rng.integers(0, height)
        rx, ry = rng.integers(10, width // 3), rng.integers(10, height // 3)
        draw.ellipse((x0 - rx, y0 - ry, x0 + rx, y0 + ry), fill=int(rng.integers(0, 256)))
    return img.filter(ImageFilter.GaussianBlur(2)).convert("RGB")


def with_marker(width: int, height: int, cx: int, cy: int, r: int = 2, bg: int = 0) -> Image.Image:
    """Тёмный фон с красным квадратом (2r+1)² с центром в пикселе (cx, cy).
    Красный — чтобы не путать с белой заливкой углов при повороте."""
    arr = np.full((height, width, 3), bg, dtype=np.uint8)
    arr[cy - r : cy + r + 1, cx - r : cx + r + 1] = (255, 0, 0)
    return Image.fromarray(arr, "RGB")


def encode(
    img: Image.Image, fmt: str = "PNG", orientation: int | None = None, **kw: object
) -> bytes:
    buf = io.BytesIO()
    if orientation is not None:
        exif = Image.Exif()
        exif[EXIF_ORIENTATION] = orientation
        kw["exif"] = exif.tobytes()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def marker_centroid(img: Image.Image, threshold: int = 16) -> tuple[float, float]:
    """Центр масс «красноты» (R − G) в координатах «центр пикселя = i + 0.5»."""
    rgb = np.asarray(img.convert("RGB")).astype(np.float64)
    arr = np.clip(rgb[..., 0] - rgb[..., 1], 0, None)
    mask = arr >= threshold
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise AssertionError("маркер не найден")
    weights = arr[mask]
    return (
        float(np.average(xs + 0.5, weights=weights)),
        float(np.average(ys + 0.5, weights=weights)),
    )
