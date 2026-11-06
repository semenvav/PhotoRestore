import hashlib
import os
import stat
from pathlib import Path

import pytest
from imgutil import encode, gradient, scene, write
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.intake import Intake
from app.models import Event, InboxFile, Photo, Status


def photos(factory: sessionmaker[Session]) -> list[Photo]:
    with factory() as s:
        return list(s.scalars(select(Photo).order_by(Photo.id)))


def events(factory: sessionmaker[Session], kind: str) -> list[Event]:
    with factory() as s:
        return list(s.scalars(select(Event).where(Event.kind == kind).order_by(Event.id)))


def jpeg(seed: int = 0, size: tuple[int, int] = (320, 240), **kw: object) -> bytes:
    return encode(gradient(*size, seed=seed), "JPEG", quality=kw.pop("quality", 92), **kw)


def test_file_is_taken_after_two_identical_polls(
    intake: Intake, inbox: Path, settings: Settings, factory: sessionmaker[Session]
) -> None:
    data = jpeg()
    src = write(inbox / "scan 001.jpg", data)

    first = intake.poll()
    assert first.registered == [] and first.pending == 1
    assert photos(factory) == []

    second = intake.poll()
    assert len(second.registered) == 1
    [photo] = photos(factory)
    assert photo.status == Status.PREP
    assert photo.code == "R-0001"
    assert photo.original_name == "scan 001.jpg"
    assert photo.album == ""
    assert photo.sha256 == hashlib.sha256(data).hexdigest()
    assert (photo.width, photo.height) == (320, 240)

    original = settings.store_dir / "R-0001" / "original.jpg"
    assert original.read_bytes() == data
    assert not os.stat(original).st_mode & stat.S_IWUSR, "оригинал должен быть только для чтения"
    assert (settings.store_dir / "R-0001" / "cache" / "thumb.webp").is_file()
    assert (settings.store_dir / "R-0001" / "cache" / "preview_original.jpg").is_file()

    # файл остаётся во «Входящих» и повторно не принимается
    assert src.read_bytes() == data
    assert intake.poll().registered == []
    assert len(photos(factory)) == 1


def test_changing_file_waits(intake: Intake, inbox: Path, factory: sessionmaker[Session]) -> None:
    path = write(inbox / "a.jpg", jpeg()[:1000])
    intake.poll()
    path.write_bytes(jpeg())  # «докопировался»
    assert intake.poll().registered == []
    assert len(intake.poll().registered) == 1
    assert len(photos(factory)) == 1


def test_duplicate_by_sha256(
    intake: Intake, inbox: Path, settings: Settings, factory: sessionmaker[Session]
) -> None:
    data = jpeg()
    write(inbox / "first.jpg", data)
    intake.poll(), intake.poll()
    write(inbox / "Альбом" / "copy.JPG", data)
    intake.poll()
    result = intake.poll()
    assert len(result.duplicates) == 1

    first, dup = photos(factory)
    assert first.status == Status.PREP
    assert dup.status == Status.DUPLICATE
    assert dup.duplicate_of == first.id
    assert dup.album == "Альбом"
    assert (settings.store_dir / dup.code / "original.jpg").read_bytes() == data


def test_similar_photo_only_warns(
    intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    img = scene(320, 240, seed=1)
    write(inbox / "a.jpg", encode(img, "JPEG", quality=95))
    write(inbox / "b.jpg", encode(img, "JPEG", quality=60))
    intake.poll(), intake.poll()

    a, b = photos(factory)
    assert a.status == b.status == Status.PREP
    [similar] = events(factory, "similar")
    assert similar.photo_id == b.id
    assert similar.data["photos"][0]["photo_id"] == a.id
    assert similar.data["photos"][0]["distance"] <= 6


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("notes.txt", b"hello"),
        ("doc.pdf", b"%PDF-1.4"),
        ("broken.jpg", b"definitely not a jpeg"),
        ("truncated.jpg", jpeg(size=(640, 480))[:3000]),
    ],
)
def test_unsupported_moved_aside(
    intake: Intake,
    inbox: Path,
    settings: Settings,
    factory: sessionmaker[Session],
    name: str,
    data: bytes,
) -> None:
    write(inbox / "Альбом" / name, data)
    intake.poll(), intake.poll()
    assert photos(factory) == []
    assert not (inbox / "Альбом" / name).exists()
    assert (settings.unsupported_dir / "Альбом" / name).read_bytes() == data
    [ev] = events(factory, "unsupported")
    assert ev.photo_id is None
    assert ev.data["path"] == f"Альбом/{name}"
    # больше не трогаем
    assert intake.poll().unsupported == []


def test_ignored_files_left_alone(
    intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    names = ["Thumbs.db", "desktop.ini", ".hidden.jpg", "~$lock.jpg", ".recycle/x.jpg"]
    for name in names:
        write(inbox / name, jpeg())
    intake.poll(), intake.poll()
    assert photos(factory) == []
    assert all((inbox / n).exists() for n in names)


def test_exif_orientation_sets_dimensions(
    intake: Intake, inbox: Path, settings: Settings, factory: sessionmaker[Session]
) -> None:
    data = jpeg(size=(320, 240), orientation=6)
    write(inbox / "rotated.jpg", data)
    intake.poll(), intake.poll()
    [photo] = photos(factory)
    assert (photo.width, photo.height) == (240, 320)
    assert (settings.store_dir / photo.code / "original.jpg").read_bytes() == data


def test_trusted_upload_taken_on_first_poll(
    intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    write(inbox / "up.png", encode(gradient(50, 40)))
    intake.trust("up.png")
    assert len(intake.poll().registered) == 1
    [photo] = photos(factory)
    assert photo.original_name == "up.png"


def test_replaced_file_at_same_path_is_new_photo(
    intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    path = write(inbox / "scan.jpg", jpeg(seed=1))
    intake.poll(), intake.poll()
    path.write_bytes(jpeg(seed=2, size=(200, 300)))
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    intake.poll(), intake.poll()
    assert len(photos(factory)) == 2
    with factory() as s:
        [row] = s.scalars(select(InboxFile)).all()
        assert row.photo_id == 2


def test_orphan_store_dir_is_set_aside(
    intake: Intake, inbox: Path, settings: Settings, factory: sessionmaker[Session]
) -> None:
    orphan = write(settings.store_dir / "R-0001" / "original.jpg", b"old bytes")
    write(inbox / "new.jpg", jpeg())
    intake.poll(), intake.poll()
    [photo] = photos(factory)
    assert photo.id == 1
    assert (settings.store_dir / "R-0001" / "original.jpg").read_bytes() != b"old bytes"
    assert not orphan.exists() or orphan.read_bytes() != b"old bytes"
    [aside] = list((settings.store_dir / ".orphans").iterdir())
    assert (aside / "original.jpg").read_bytes() == b"old bytes"


def test_ids_follow_arrival_order(
    intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    for i, name in enumerate(["c.jpg", "a.jpg", "b.jpg"]):
        p = write(inbox / name, jpeg(seed=10 + i))
        t = (1_000_000_000 + i * 60) * 10**9  # NTFS хранит mtime с шагом 100 нс
        os.utime(p, ns=(t, t))
    intake.poll(), intake.poll()
    assert [p.original_name for p in photos(factory)] == ["c.jpg", "a.jpg", "b.jpg"]
