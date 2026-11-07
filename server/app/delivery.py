"""Отдача (ТЗ §5.6): «Утвердить» кладёт финал в «Готово».

Файл — `delivered/<альбом>/<исходное_имя>_restored.png`; если это имя уже занято другим
фото — `<исходное_имя>_restored_R-0042.png`. Чей файл, видно по метаданным: в PNG пишутся
ID, исходное имя и дата (tEXt; не-latin-1 текст Pillow-совместимо — iTXt в UTF-8).

PNG не перекодируется: копируются чанки изображения как есть, прежние текстовые чанки
(ComfyUI кладёт туда граф и промпт) отбрасываются, наши вставляются перед IDAT.
"""

import os
import struct
import uuid
import zlib
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app import states
from app.db import utcnow
from app.models import Actor, Photo, Status, Variant
from app.storage import Store

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TEXT_CHUNKS = frozenset({b"tEXt", b"zTXt", b"iTXt"})
SOFTWARE = "photo-restore"


class DeliveryError(Exception):
    pass


# --- PNG-чанки ---------------------------------------------------------------------------


def iter_chunks(data: bytes) -> Iterator[tuple[bytes, bytes, bytes]]:
    """(тип, данные, чанк целиком) по порядку; бросает DeliveryError на битом PNG."""
    if not data.startswith(PNG_SIGNATURE):
        raise DeliveryError("файл не PNG")
    pos = len(PNG_SIGNATURE)
    while pos < len(data):
        if pos + 8 > len(data):
            raise DeliveryError("PNG обрезан")
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        end = pos + 12 + length
        if end > len(data):
            raise DeliveryError("PNG обрезан")
        yield kind, data[pos + 8 : pos + 8 + length], data[pos:end]
        pos = end
        if kind == b"IEND":
            return
    raise DeliveryError("в PNG нет IEND")


def _chunk(kind: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(kind + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def text_chunk(key: str, value: str) -> bytes:
    keyword = key.encode("latin-1")
    try:
        return _chunk(b"tEXt", keyword + b"\0" + value.encode("latin-1"))
    except UnicodeEncodeError:
        # iTXt: без сжатия, без языка и перевода ключа
        return _chunk(b"iTXt", keyword + b"\0\0\0\0\0" + value.encode("utf-8"))


def with_text(data: bytes, items: dict[str, str]) -> bytes:
    """Тот же PNG без прежних текстовых чанков и с `items` перед первым IDAT."""
    out = [PNG_SIGNATURE]
    inserted = False
    for kind, _, raw in iter_chunks(data):
        if kind in TEXT_CHUNKS:
            continue
        if kind == b"IDAT" and not inserted:
            out.extend(text_chunk(k, v) for k, v in items.items())
            inserted = True
        out.append(raw)
    if not inserted:
        raise DeliveryError("в PNG нет IDAT")
    return b"".join(out)


def read_text(path: Path) -> dict[str, str]:
    """Текстовые метаданные PNG до первого IDAT (наши лежат там)."""
    try:
        data = path.read_bytes()
        result: dict[str, str] = {}
        for kind, body, _ in iter_chunks(data):
            if kind == b"IDAT":
                break
            if kind == b"tEXt":
                key, _, value = body.partition(b"\0")
                result[key.decode("latin-1")] = value.decode("latin-1")
            elif kind == b"iTXt":
                key, _, rest = body.partition(b"\0")
                if rest[:1] == b"\0":  # только несжатые — других мы не пишем
                    _, _, rest = rest[2:].partition(b"\0")  # язык
                    _, _, value = rest.partition(b"\0")  # переведённый ключ
                    result[key.decode("latin-1")] = value.decode("utf-8", "replace")
        return result
    except (OSError, DeliveryError):
        return {}


# --- куда класть -------------------------------------------------------------------------


def album_dir(delivered_dir: Path, album: str) -> Path:
    """Подпапка альбома внутри «Готово»; путь, выходящий наружу, не принимается."""
    parts = [p for p in PurePosixPath(album.replace("\\", "/")).parts if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise DeliveryError(f"недопустимый альбом: {album!r}")
    return delivered_dir.joinpath(*parts)


def target_path(delivered_dir: Path, photo: Photo) -> Path:
    folder = album_dir(delivered_dir, photo.album)
    stem = Path(photo.original_name).stem or photo.code
    base = folder / f"{stem}_restored.png"
    if not base.exists() or read_text(base).get("Title") == photo.code:
        return base
    return folder / f"{stem}_restored_{photo.code}.png"


# --- «Утвердить» -------------------------------------------------------------------------


def deliver(
    session: Session,
    store: Store,
    delivered_dir: Path,
    photo: Photo,
    variant: Variant,
    display_tz: str,
    actor: Actor = Actor.OWNER,
) -> Path:
    """Копия `variant` во «Готово», затем FINAL_REVIEW → DELIVERED. Файл пишется атомарно
    (временное имя + rename); повторная отдача того же фото перезаписывает его файл."""
    if not states.can_transition(photo.status, Status.DELIVERED, actor):
        raise states.TransitionError(f"{photo.code}: утвердить можно только из «Финала»")
    if variant.photo_id != photo.id:
        raise DeliveryError(f"{variant.label} — вариант другого фото")
    when: datetime = utcnow().astimezone(ZoneInfo(display_tz))
    data = with_text(
        (store.photo_dir(photo.id) / variant.path).read_bytes(),
        {
            "Title": photo.code,
            "Source": photo.original_name,
            "Restored": when.strftime("%Y-%m-%d"),
            "Software": SOFTWARE,
        },
    )
    dest = target_path(delivered_dir, photo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{photo.code}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    rel = dest.relative_to(delivered_dir).as_posix()
    photo.final_variant_id = variant.id
    photo.delivered_path = rel
    states.transition(
        session, photo, Status.DELIVERED, actor, {"variant": variant.label, "delivered": rel}
    )
    return dest
