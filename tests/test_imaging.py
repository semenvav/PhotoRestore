import math
from pathlib import Path

import numpy as np
import pytest
from imgutil import encode, gradient, marker_centroid, with_marker, write
from PIL import Image

from app.imaging import (
    Crop,
    PrepParams,
    load_oriented,
    oriented_frame,
    prepare,
    rotated_bbox,
    to_rgb8,
)

# EXIF Orientation → как получить «правильную» картинку из хранимой
ORIENTATION_FIX = {
    2: [Image.Transpose.FLIP_LEFT_RIGHT],
    3: [Image.Transpose.ROTATE_180],
    4: [Image.Transpose.FLIP_TOP_BOTTOM],
    5: [Image.Transpose.TRANSPOSE],
    6: [Image.Transpose.ROTATE_270],
    7: [Image.Transpose.TRANSVERSE],
    8: [Image.Transpose.ROTATE_90],
}


def apply(img: Image.Image, ops: list[Image.Transpose]) -> Image.Image:
    for op in ops:
        img = img.transpose(op)
    return img


def cropper_rotated_sizes(width: float, height: float, degree: float) -> tuple[float, float]:
    """Порт getRotatedSizes из cropper.js 1.7.0."""
    degree = abs(degree) % 180
    if degree == 90:
        return height, width
    arc = (degree % 90) * math.pi / 180
    w = width * math.cos(arc) + height * math.sin(arc)
    h = width * math.sin(arc) + height * math.cos(arc)
    return (h, w) if degree > 90 else (w, h)


@pytest.mark.parametrize("orientation", sorted(ORIENTATION_FIX))
def test_exif_orientation_png_exact(tmp_path: Path, orientation: int) -> None:
    stored = gradient(64, 48)
    path = write(tmp_path / "x.png", encode(stored, "PNG", orientation=orientation))
    oriented = load_oriented(path)
    expected = apply(stored, ORIENTATION_FIX[orientation])
    assert oriented.size == expected.size
    assert np.array_equal(np.asarray(oriented), np.asarray(expected))


@pytest.mark.parametrize("fmt", ["JPEG", "AVIF"])
@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_exif_orientation_lossy(tmp_path: Path, orientation: int, fmt: str) -> None:
    stored = gradient(96, 64, noise=False)
    data = encode(stored, fmt, orientation=orientation, quality=95)
    path = write(tmp_path / f"x.{fmt.lower()}", data)
    oriented = load_oriented(path)
    expected = apply(stored, ORIENTATION_FIX[orientation])
    assert oriented.size == expected.size
    diff = np.abs(np.asarray(oriented, np.int16) - np.asarray(expected, np.int16))
    assert diff.mean() < 2


def test_original_file_untouched_by_loading(tmp_path: Path) -> None:
    data = encode(gradient(32, 32), "JPEG", orientation=6)
    path = write(tmp_path / "x.jpg", data)
    load_oriented(path)
    assert path.read_bytes() == data


@pytest.mark.parametrize("degree", [0, 0.1, -3.7, 10, -10, 90, 85, 95, 180, 270, 264.5])
def test_frame_matches_cropper(degree: float) -> None:
    quarters, fine = divmod(degree, 90)
    if fine > 10:  # приводим к виду 90k + angle, angle ∈ [-10, 10]
        quarters, fine = quarters + 1, fine - 90
    params = PrepParams(rotate90=int(quarters), angle=fine)
    got = oriented_frame(300, 200, params)
    want = cropper_rotated_sizes(300, 200, degree)
    assert got == pytest.approx(want, abs=1e-6)


def test_quarter_turns_and_crop_are_exact() -> None:
    img = gradient(120, 80)
    for quarters, op in [
        (1, Image.Transpose.ROTATE_270),
        (2, Image.Transpose.ROTATE_180),
        (3, Image.Transpose.ROTATE_90),
    ]:
        crop = Crop(x=10, y=5, w=40, h=30)
        out = prepare(img, PrepParams(rotate90=quarters, crop=crop))
        ref = img.transpose(op).crop((10, 5, 50, 35))
        assert np.array_equal(np.asarray(out), np.asarray(ref))


def test_no_crop_keeps_whole_frame() -> None:
    img = gradient(120, 80)
    out = prepare(img, PrepParams())
    assert np.array_equal(np.asarray(out), np.asarray(img))
    out = prepare(img, PrepParams(rotate90=1, angle=5))
    w2, h2 = rotated_bbox(80, 120, 5)
    assert out.size == (round(w2), round(h2))


def forward(
    px: float, py: float, width: int, height: int, params: PrepParams
) -> tuple[float, float]:
    """Куда точка исходника попадает в кадре cropper: четверти по часовой, затем angle."""
    for _ in range(params.rotate90):
        px, py, width, height = height - py, px, height, width
    w2, h2 = rotated_bbox(width, height, params.angle)
    t = math.radians(params.angle)
    dx, dy = px - width / 2, py - height / 2
    return (
        math.cos(t) * dx - math.sin(t) * dy + w2 / 2,
        math.sin(t) * dx + math.cos(t) * dy + h2 / 2,
    )


@pytest.mark.parametrize(
    ("rotate90", "angle", "crop"),
    [
        (0, 7.3, (20.4, 15.2, 160.0, 110.0)),
        (0, -4.2, (0.0, 0.0, 180.0, 130.0)),
        (1, 2.5, (12.7, 30.1, 100.0, 150.0)),
        (3, -9.9, (5.5, 5.5, 140.0, 190.0)),
        (2, 0.1, (40.0, 20.0, 120.0, 90.0)),
    ],
)
def test_marker_lands_where_cropper_shows_it(
    rotate90: int, angle: float, crop: tuple[float, float, float, float]
) -> None:
    width, height, cx, cy = 200, 140, 70, 45
    img = with_marker(width, height, cx, cy)
    params = PrepParams(
        rotate90=rotate90, angle=angle, crop=Crop(x=crop[0], y=crop[1], w=crop[2], h=crop[3])
    )
    out = prepare(img, params)
    ex, ey = forward(cx + 0.5, cy + 0.5, width, height, params)
    gx, gy = marker_centroid(out)
    assert gx == pytest.approx(ex - crop[0], abs=0.35)
    assert gy == pytest.approx(ey - crop[1], abs=0.35)
    assert out.size == (round(crop[2]), round(crop[3]))


def test_crop_clamped_to_frame() -> None:
    img = gradient(100, 80)
    out = prepare(img, PrepParams(crop=Crop(x=-5, y=-5, w=200, h=200)))
    assert out.size == (100, 80)
    with pytest.raises(ValueError):
        prepare(img, PrepParams(crop=Crop(x=500, y=500, w=10, h=10)))


def test_angle_limits() -> None:
    with pytest.raises(ValueError):
        PrepParams(angle=10.5)
    assert PrepParams(rotate90=-1).rotate90 == 3


def test_16bit_gray_is_scaled_not_clipped() -> None:
    arr = np.array([[0, 32768, 65535]], dtype=np.uint16)
    img = Image.fromarray(arr)
    assert img.mode == "I;16"
    out = np.asarray(to_rgb8(img))
    assert out.shape == (1, 3, 3)
    assert out[0, :, 0].tolist() == [0, 128, 255]


def test_alpha_composited_on_white() -> None:
    img = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    img.putpixel((1, 0), (10, 20, 30, 255))
    out = to_rgb8(img)
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((1, 0)) == (10, 20, 30)
