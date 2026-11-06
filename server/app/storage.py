"""Пути внутри `store/`. Строятся только из ID фото и фиксированных имён — имена файлов
от пользователя в пути не попадают (исходное имя хранится в БД)."""

import os
import re
import shutil
from pathlib import Path

from app.db import utcnow
from app.models import photo_code

ALLOWED_EXT = frozenset({"jpg", "jpeg", "png", "webp", "tif", "tiff"})
_PREVIEW_KIND = re.compile(r"^[a-z0-9_]{1,40}$")


def original_ext(original_name: str) -> str:
    ext = Path(original_name).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXT:
        raise ValueError(f"недопустимое расширение: {original_name!r}")
    return ext


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def staging_dir(self) -> Path:
        return self.root / ".staging"

    @property
    def orphans_dir(self) -> Path:
        return self.root / ".orphans"

    def photo_dir(self, photo_id: int) -> Path:
        return self.root / photo_code(photo_id)

    def original(self, photo_id: int, original_name: str) -> Path:
        return self.photo_dir(photo_id) / f"original.{original_ext(original_name)}"

    def prepared(self, photo_id: int) -> Path:
        return self.photo_dir(photo_id) / "prepared.png"

    def next_prepared_archive(self, photo_id: int) -> Path:
        n = 1
        while (path := self.photo_dir(photo_id) / f"prepared_{n}.png").exists():
            n += 1
        return path

    def cache_dir(self, photo_id: int) -> Path:
        return self.photo_dir(photo_id) / "cache"

    def thumb(self, photo_id: int) -> Path:
        return self.cache_dir(photo_id) / "thumb.webp"

    def preview(self, photo_id: int, kind: str) -> Path:
        if not _PREVIEW_KIND.match(kind):
            raise ValueError(f"недопустимое имя превью: {kind!r}")
        return self.cache_dir(photo_id) / f"preview_{kind}.jpg"

    def set_aside_orphan(self, photo_id: int) -> Path | None:
        """Если каталог для нового ID уже есть (сбой до commit или БД восстановлена из
        бэкапа), убирает его в `.orphans/`, ничего не удаляя."""
        src = self.photo_dir(photo_id)
        if not src.exists():
            return None
        self.orphans_dir.mkdir(parents=True, exist_ok=True)
        dest = self.orphans_dir / f"{src.name}-{utcnow():%Y%m%dT%H%M%S%f}"
        shutil.move(src, dest)
        return dest


def replace_atomic(tmp: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, dest)
