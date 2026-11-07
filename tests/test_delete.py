from pathlib import Path
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient
from imgutil import encode, gradient, scene, write
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.intake import Intake
from app.models import Event, InboxFile, Job, Photo, Status
from app.storage import Store


def ingest(intake: Intake, inbox: Path, name: str, data: bytes) -> None:
    write(inbox / name, data)
    intake.trust(name)
    result = intake.poll()
    assert result.registered or result.duplicates


def png(seed: int = 0) -> bytes:
    return encode(gradient(200, 150, seed=seed))


def photos(factory: sessionmaker[Session]) -> dict[int, Photo]:
    with factory() as s:
        return {p.id: p for p in s.scalars(select(Photo))}


def set_status(factory: sessionmaker[Session], photo_id: int, status: Status) -> None:
    """Только для теста: статус «в обработке» в M1 иначе не получить."""
    with factory() as s:
        s.execute(update(Photo).where(Photo.id == photo_id).values(status=status))
        s.commit()


def location(r: object) -> str:
    return unquote(r.headers["location"])  # type: ignore[attr-defined]


def test_delete_from_card(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    settings: Settings,
    factory: sessionmaker[Session],
) -> None:
    data = png()
    ingest(intake, inbox, "a.png", data)
    client.post("/prep/1", json={"crop": None})  # prepared.png, задача, события

    r = client.post("/photo/1/delete", data={"back": "/"}, follow_redirects=False)
    assert r.status_code == 303
    assert location(r).startswith("/?msg=Удалено: R-0001")

    assert photos(factory) == {}
    with factory() as s:
        assert s.scalars(select(Job)).all() == []
        assert s.scalars(select(Event).where(Event.photo_id.is_not(None))).all() == []
        [ev] = s.scalars(select(Event).where(Event.kind == "deleted")).all()
        assert ev.photo_id is None
        assert ev.data["code"] == "R-0001" and ev.data["original_name"] == "a.png"
        [row] = s.scalars(select(InboxFile)).all()
        assert row.photo_id is None and row.path == "a.png"

    # оригинал не удалён — каталог перенесён в .deleted
    assert not (settings.store_dir / "R-0001").exists()
    [moved] = Store(settings.store_dir).deleted_dir.iterdir()
    assert moved.name.startswith("R-0001-")
    assert (moved / "original.png").read_bytes() == data
    assert (moved / "prepared.png").is_file()

    # файл во «Входящих» на месте и повторно не принимается
    assert (inbox / "a.png").read_bytes() == data
    intake.poll(), intake.poll()
    assert photos(factory) == {}

    assert client.get("/photo/1").status_code == 404
    assert "Удалено: R-0001" in client.get(r.headers["location"]).text


def test_ids_are_not_reused(
    client: TestClient, intake: Intake, inbox: Path, settings: Settings
) -> None:
    ingest(intake, inbox, "a.png", png(1))
    client.post("/photo/1/delete")
    ingest(intake, inbox, "b.png", png(2))
    assert (settings.store_dir / "R-0002" / "original.png").is_file()
    assert not (settings.store_dir / "R-0001").exists()


def test_deleting_primary_promotes_first_duplicate(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    data = png()
    for name in ["a.png", "b.png", "c.png"]:
        ingest(intake, inbox, name, data)
    assert photos(factory)[2].duplicate_of == 1

    card = client.get("/photo/1").text
    assert "основным станет R-0002" in card

    r = client.post("/photo/1/delete", follow_redirects=False)
    assert "основным вместо R-0001 стал его дубль R-0002" in location(r)

    left = photos(factory)
    assert sorted(left) == [2, 3]
    assert left[2].duplicate_of is None
    assert left[3].duplicate_of == 2
    with factory() as s:
        promoted = s.scalar(select(Event).where(Event.kind == "promoted"))
        assert promoted is not None and promoted.photo_id == 2
    assert "Стало основным: дубль удалённого R-0001" in client.get("/photo/2").text

    # тот же файл снова — дубль нового основного
    ingest(intake, inbox, "d.png", data)
    assert photos(factory)[4].duplicate_of == 2


def test_deleting_duplicate_keeps_primary(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    data = png()
    ingest(intake, inbox, "a.png", data)
    ingest(intake, inbox, "b.png", data)
    client.post("/photo/2/delete")
    left = photos(factory)
    assert sorted(left) == [1]
    assert left[1].duplicate_of is None


def test_bulk_delete_skips_busy(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    data = png()
    ingest(intake, inbox, "a.png", data)
    ingest(intake, inbox, "a-copy.png", data)
    ingest(intake, inbox, "b.png", png(5))
    ingest(intake, inbox, "c.png", png(6))
    set_status(factory, 4, Status.EDITING)

    r = client.post(
        "/bulk",
        data={"action": "delete", "ids": ["1", "2", "3", "4"], "back": "/?album=*"},
        follow_redirects=False,
    )
    msg = location(r)
    assert msg.startswith("/?album=*&msg=Удалено: R-0001, R-0002, R-0003")
    assert "в обработке, не удалено: R-0004" in msg
    assert "основным" not in msg  # дубль удалён вместе с основным
    assert sorted(photos(factory)) == [4]


def test_busy_photo_cannot_be_deleted(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    ingest(intake, inbox, "a.png", png())
    set_status(factory, 1, Status.EDITING)
    assert "delete-dialog" not in client.get("/photo/1").text

    r = client.post("/photo/1/delete", follow_redirects=False)
    assert location(r).startswith("/photo/1?msg=в обработке, не удалено: R-0001")
    assert sorted(photos(factory)) == [1]


def test_delete_from_prep_goes_to_next(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    for i, name in enumerate(["a.png", "b.png", "c.png"]):
        ingest(intake, inbox, name, png(i))
    page = client.get("/prep/2").text
    assert 'name="back" value="/prep/2/skip"' in page

    r = client.post("/photo/2/delete", data={"back": "/prep/2/skip"})
    assert r.url.path == "/prep/3"
    assert sorted(photos(factory)) == [1, 3]


@pytest.mark.parametrize("back", ["//evil.example/x", "https://evil.example/", "javascript:x"])
def test_back_must_be_local(client: TestClient, intake: Intake, inbox: Path, back: str) -> None:
    ingest(intake, inbox, "a.png", png())
    r = client.post("/photo/1/delete", data={"back": back}, follow_redirects=False)
    assert location(r).startswith("/?msg=")


def test_delete_needs_confirmation_in_ui(client: TestClient, intake: Intake, inbox: Path) -> None:
    """Кнопка только открывает диалог; удаляющая форма живёт внутри <dialog>."""
    ingest(intake, inbox, "a.png", png())
    for url, back in [("/photo/1", "/"), ("/prep/1", "/prep/1/skip")]:
        page = client.get(url).text
        assert "showModal()" in page
        dialog = page[page.index('<dialog class="confirm" id="delete-dialog"') :]
        dialog = dialog[: dialog.index("</dialog>")]
        assert page.count('action="/photo/1/delete"') == 1
        assert 'action="/photo/1/delete"' in dialog
        assert f'name="back" value="{back}"' in dialog
        assert 'formmethod="dialog"' in dialog  # «Отмена» закрывает без отправки

    board = client.get("/").text
    dialog = board[board.index('x-ref="del"') :]
    dialog = dialog[: dialog.index("</dialog>")]
    assert 'value="delete"' in dialog
    assert board.count('value="delete"') == 1


def test_similar_link_to_deleted_photo_disappears(
    client: TestClient, intake: Intake, inbox: Path
) -> None:
    img = scene(320, 240, seed=1)
    ingest(intake, inbox, "a.jpg", encode(img, "JPEG", quality=95))
    ingest(intake, inbox, "b.jpg", encode(img, "JPEG", quality=60))
    assert 'href="/photo/1"' in client.get("/prep/2").text
    client.post("/photo/1/delete")
    assert 'href="/photo/1"' not in client.get("/prep/2").text


def test_failed_move_still_deletes(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    settings: Settings,
    factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest(intake, inbox, "a.png", png())

    def broken(self: Store, photo_id: int) -> Path:
        raise OSError("NFS недоступен")

    monkeypatch.setattr(Store, "set_aside_deleted", broken)
    r = client.post("/photo/1/delete", follow_redirects=False)
    assert "Удалено: R-0001" in location(r)
    assert photos(factory) == {}
    assert (settings.store_dir / "R-0001" / "original.png").is_file()
