import io
import re
import time
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest
from fastapi.testclient import TestClient
from imgutil import encode, gradient, write
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.intake import Intake
from app.main import create_app
from app.models import Event, Job, JobStatus, JobType, Photo, Status
from app.presets import BASE_PROMPT


def ingest(intake: Intake, inbox: Path, name: str, data: bytes) -> None:
    write(inbox / name, data)
    intake.trust(Path(name).as_posix())
    result = intake.poll()
    assert result.registered or result.duplicates


def get_photo(factory: sessionmaker[Session], photo_id: int = 1) -> Photo:
    with factory() as s:
        photo = s.get(Photo, photo_id)
        assert photo is not None
        return photo


def test_board_empty(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Фото пока нет" in r.text


def test_build_version_reported(settings: Settings) -> None:
    settings.app_version = "0.1.7"
    with TestClient(create_app(settings)) as c:
        assert c.get("/healthz").json() == {"ok": True, "version": "0.1.7"}
        assert '<span class="ver" title="Номер сборки">0.1.7</span>' in c.get("/").text


def test_board_lists_photo_and_polls_cheaply(
    client: TestClient, intake: Intake, inbox: Path
) -> None:
    ingest(intake, inbox, "Семья/scan.jpg", encode(gradient(300, 200), "JPEG"))
    r = client.get("/")
    assert r.status_code == 200
    assert "R-0001" in r.text and "scan.jpg" in r.text

    token = re.search(r"/partials/columns\?[^\"]*v=([0-9a-f]+)", r.text)
    assert token is not None
    assert client.get(f"/partials/columns?v={token.group(1)}").status_code == 204
    assert client.get("/partials/columns?v=stale").status_code == 200

    assert "R-0001" in client.get("/?q=r-1").text
    assert "R-0001" in client.get("/?q=SCAN").text
    assert "R-0001" not in client.get("/?q=nothing").text
    assert "R-0001" in client.get("/?album=Семья").text
    assert "R-0001" not in client.get("/?album=").text


def test_card_and_files(client: TestClient, intake: Intake, inbox: Path) -> None:
    data = encode(gradient(300, 200), "JPEG")
    ingest(intake, inbox, "Фото дедушки.jpg", data)
    r = client.get("/photo/1")
    assert r.status_code == 200
    assert "Принято из «Входящих»" in r.text

    thumb = client.get("/files/1/thumb")
    assert thumb.headers["content-type"] == "image/webp"
    assert max(Image.open(io.BytesIO(thumb.content)).size) == 300

    original = client.get("/files/1/original")
    assert original.content == data
    assert "filename*=utf-8''" in original.headers["content-disposition"]

    assert client.get("/files/1/preview/original").status_code == 200
    assert client.get("/files/1/preview/../../x").status_code == 404
    assert client.get("/files/1/prepared").status_code == 404
    assert client.get("/photo/999").status_code == 404


def test_prep_flow_creates_job_and_next(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    settings: Settings,
    factory: sessionmaker[Session],
) -> None:
    ingest(intake, inbox, "a.png", encode(gradient(400, 300, seed=1)))
    ingest(intake, inbox, "b.png", encode(gradient(400, 300, seed=2)))

    r = client.get("/prep", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/prep/1"
    page = client.get("/prep/1")
    assert page.status_code == 200 and "prep-config" in page.text

    r = client.post(
        "/prep/1", json={"rotate90": 1, "angle": 0, "crop": {"x": 10, "y": 20, "w": 200, "h": 300}}
    )
    assert r.status_code == 200, r.text
    assert r.json()["next"] == "/prep/2"

    photo = get_photo(factory)
    assert photo.status == Status.QUEUED_EDIT
    prepared = Image.open(settings.store_dir / "R-0001" / "prepared.png")
    assert prepared.mode == "RGB" and prepared.size == (200, 300)
    assert (settings.store_dir / "R-0001" / "cache" / "preview_prepared.jpg").is_file()

    with factory() as s:
        [job] = s.scalars(select(Job)).all()
    assert job.type == JobType.EDIT_LOCAL and job.status == JobStatus.QUEUED
    assert job.payload["prompt"] == BASE_PROMPT
    assert len(job.payload["seeds"]) == 3 and len(set(job.payload["seeds"])) == 3
    assert job.payload["source"] == "prepared"

    # пропуск последнего в очереди ведёт к нему же (других нет)
    r = client.get("/prep/2/skip", follow_redirects=False)
    assert r.headers["location"] == "/prep/2"
    # «Без обрезки» последнего → экран «нет фото»
    r = client.post("/prep/2", json={"rotate90": 0, "angle": 0, "crop": None})
    assert r.json()["next"] == "/prep"
    assert Image.open(settings.store_dir / "R-0002" / "prepared.png").size == (400, 300)


def test_prepared_matches_editor_frame(
    client: TestClient, intake: Intake, inbox: Path, settings: Settings
) -> None:
    """Критерий M1: prepared.png совпадает с рамкой в редакторе (превью 2048 px)."""
    original = gradient(3000, 2000, seed=5)
    ingest(intake, inbox, "big.png", encode(original))
    preview = Image.open(settings.store_dir / "R-0001" / "cache" / "preview_original.jpg")
    assert max(preview.size) == 2048

    # как это делает prep.js: getData() в пикселях превью × (ширина оригинала / ширина превью)
    scale = 3000 / preview.width
    box = {"x": 100.0, "y": 50.0, "width": 1000.0, "height": 800.0}
    crop = {"x": box["x"] * scale, "y": box["y"] * scale}
    crop |= {"w": box["width"] * scale, "h": box["height"] * scale}
    r = client.post("/prep/1", json={"rotate90": 0, "angle": 0, "crop": crop})
    assert r.status_code == 200, r.text

    prepared = Image.open(settings.store_dir / "R-0001" / "prepared.png")
    x0, y0 = round(crop["x"]), round(crop["y"])
    x1, y1 = round(crop["x"] + crop["w"]), round(crop["y"] + crop["h"])
    expected = original.crop((x0, y0, x1, y1))
    assert prepared.size == expected.size
    assert np.array_equal(np.asarray(prepared), np.asarray(expected))


def test_reprep_keeps_old_file(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    settings: Settings,
    factory: sessionmaker[Session],
) -> None:
    ingest(intake, inbox, "a.png", encode(gradient(400, 300)))
    client.post("/prep/1", json={"crop": None})
    first = (settings.store_dir / "R-0001" / "prepared.png").read_bytes()
    r = client.post("/prep/1", json={"crop": {"x": 0, "y": 0, "w": 100, "h": 100}})
    assert r.json()["next"] == "/photo/1"

    photo_dir = settings.store_dir / "R-0001"
    assert (photo_dir / "prepared_1.png").read_bytes() == first
    assert Image.open(photo_dir / "prepared.png").size == (100, 100)
    assert get_photo(factory).status == Status.QUEUED_EDIT
    with factory() as s:
        assert len(s.scalars(select(Job)).all()) == 1
        assert s.scalar(select(Event).where(Event.kind == "reprepared")) is not None


@pytest.mark.parametrize(
    "body",
    [
        {"angle": 11},
        {"crop": {"x": 0, "y": 0, "w": 0, "h": 5}},
        {"crop": {"x": 900, "y": 900, "w": 5, "h": 5}},
    ],
)
def test_prep_rejects_bad_params(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session], body: dict
) -> None:
    ingest(intake, inbox, "a.png", encode(gradient(400, 300)))
    r = client.post("/prep/1", json=body)
    assert r.status_code in (400, 409)
    assert "error" in r.json()
    assert get_photo(factory).status == Status.PREP


def test_hold_cancels_queued_jobs_and_unhold(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    ingest(intake, inbox, "a.png", encode(gradient(400, 300)))
    client.post("/prep/1", json={"crop": None})

    client.post("/photo/1/hold")
    assert get_photo(factory).status == Status.ON_HOLD
    with factory() as s:
        [job] = s.scalars(select(Job)).all()
    assert job.status == JobStatus.CANCELLED

    r = client.post("/photo/1/hold", follow_redirects=False)
    assert "msg=" in r.headers["location"]

    client.post("/photo/1/unhold")
    assert get_photo(factory).status == Status.PREP


def test_bulk_hold_reports_unavailable(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    data = encode(gradient(200, 100))
    ingest(intake, inbox, "a.png", data)
    ingest(intake, inbox, "dup.png", data)
    ingest(intake, inbox, "c.png", encode(gradient(200, 100, seed=3)))
    client.post("/photo/3/hold")

    r = client.post(
        "/bulk", data={"action": "hold", "ids": ["1", "2", "3"]}, follow_redirects=False
    )
    assert r.status_code == 303
    assert "R-0003" in unquote(r.headers["location"])
    assert [get_photo(factory, i).status for i in (1, 2, 3)] == [Status.ON_HOLD] * 3


def test_upload_goes_through_inbox(
    client: TestClient, intake: Intake, settings: Settings, factory: sessionmaker[Session]
) -> None:
    data = encode(gradient(120, 80), "JPEG")
    r = client.post(
        "/upload",
        files=[
            ("files", ("Бабушка.jpg", data, "image/jpeg")),
            ("files", ("readme.txt", b"x", "text/plain")),
        ],
        data={"album": "../Свадьба 1958/"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert (settings.inbox_dir / "Свадьба 1958" / "Бабушка.jpg").read_bytes() == data
    assert not list(settings.inbox_dir.rglob("readme.txt"))

    intake.poll()
    photo = get_photo(factory)
    assert photo.album == "Свадьба 1958"
    assert photo.original_name == "Бабушка.jpg"


def test_background_loop_registers_file(tmp_path: Path) -> None:
    """Фоновый опрос: файл во «Входящих» сам становится карточкой PREP."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        intake_poll_s=0.05,
    )
    app = create_app(settings)
    with TestClient(app):
        write(settings.inbox_dir / "late.jpg", encode(gradient(64, 64), "JPEG"))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with app.state.factory() as s:
                photo = s.scalar(select(Photo))
            if photo is not None:
                break
            time.sleep(0.05)
        assert photo is not None and photo.status == Status.PREP
