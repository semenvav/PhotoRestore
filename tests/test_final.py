"""M3: «Принять», апскейл, экран финала, отдача во «Готово», массовые действия."""

import io
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pytest
from fastapi.testclient import TestClient
from imgutil import encode, gradient, write
from PIL import Image, PngImagePlugin
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import delivery, jobs
from app.config import Settings
from app.intake import Intake
from app.models import Event, Job, JobStatus, JobType, Photo, Status, Variant, VariantSource

TOKEN = "f" * 43
AUTH = {"Authorization": f"Bearer {TOKEN}"}
API = "/api/agent/v1"


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.agent_token = SecretStr(TOKEN)
    return settings


def msg(r: Any) -> str:
    return unquote(r.headers["location"])


def png(width: int, height: int, seed: int = 0, text: dict[str, str] | None = None) -> bytes:
    """PNG как от ComfyUI: с текстовыми чанками (граф и промпт)."""
    info = PngImagePlugin.PngInfo()
    for k, v in (text or {"prompt": '{"1": {}}', "workflow": "{}"}).items():
        info.add_text(k, v)
    buf = io.BytesIO()
    gradient(width, height, seed=seed).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def queue_photo(client: TestClient, intake: Intake, inbox: Path, name: str, album: str = "") -> int:
    path = inbox / album / name if album else inbox / name
    write(path, encode(gradient(240, 160, seed=sum(map(ord, album + name)))))
    intake.trust((Path(album) / name).as_posix() if album else name)
    [photo_id] = intake.poll().registered
    client.post(f"/prep/{photo_id}", json={"crop": None})
    return photo_id


def agent_does_next(client: TestClient, size: tuple[int, int] = (200, 100)) -> dict[str, Any]:
    """Агент берёт задачу и отдаёт все выходы."""
    job = client.post(f"{API}/claim", json={"agent_id": "pc"}, headers=AUTH).json()
    for out in job["outputs"]:
        client.post(
            f"{API}/jobs/{job['id']}/outputs/{out['index']}",
            headers=AUTH,
            files={"file": ("o.png", png(*size, seed=out["index"]), "image/png")},
            data={"meta": json.dumps({"seed": out["seed"]})},
        )
    assert client.post(f"{API}/jobs/{job['id']}/complete", headers=AUTH).status_code == 200
    return job


def in_review(client: TestClient, intake: Intake, inbox: Path, name: str, album: str = "") -> int:
    photo_id = queue_photo(client, intake, inbox, name, album)
    client.post("/bulk", data={"action": "run", "ids": [str(photo_id)]})
    agent_does_next(client, (1680, 944))
    return photo_id


def variants(factory: sessionmaker[Session], photo_id: int) -> list[Variant]:
    with factory() as s:
        return list(s.scalars(select(Variant).where(Variant.photo_id == photo_id)))


def photo(factory: sessionmaker[Session], photo_id: int) -> Photo:
    with factory() as s:
        p = s.get(Photo, photo_id)
        assert p is not None
        return p


def last_job(factory: sessionmaker[Session], photo_id: int) -> Job:
    with factory() as s:
        job = s.scalars(select(Job).where(Job.photo_id == photo_id).order_by(Job.id.desc())).first()
        assert job is not None
        return job


@pytest.mark.parametrize(
    ("size", "target", "max_scale", "expected"),
    [
        ((1680, 944), 4.0, 2.0, 1.59),  # 1,5 Мп → 4 Мп
        ((800, 600), 4.0, 2.0, 2.0),  # упор в ×2
        ((3000, 2000), 4.0, 2.0, 1.0),  # уже больше 4 Мп — не уменьшаем
        ((1000, 1000), 9.0, 4.0, 3.0),
    ],
)
def test_auto_scale(
    size: tuple[int, int], target: float, max_scale: float, expected: float
) -> None:
    assert jobs.auto_scale(*size, target, max_scale) == expected


def test_accept_queues_upscale(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    pid = in_review(client, intake, inbox, "a.png")
    v2 = variants(factory, pid)[1]
    page = client.get(f"/review/{pid}").text
    assert f'action="/review/{pid}/accept"' in page and "Принять без апскейла" in page

    r = client.post(
        f"/review/{pid}/accept",
        data={"variant_id": v2.id, "upscale": "true"},
        follow_redirects=False,
    )
    assert "принят v002, апскейл в очереди" in msg(r)
    p = photo(factory, pid)
    assert p.status == Status.QUEUED_UPSCALE and p.accepted_variant_id == v2.id
    job = last_job(factory, pid)
    assert job.type == JobType.UPSCALE_LOCAL and job.status == JobStatus.QUEUED
    assert job.payload["source_variant_id"] == v2.id and len(job.payload["seeds"]) == 1
    assert job.payload["params"] == {"model": "7b_fp16", "scale": 1.59}

    # агенту уходит граф SeedVR2 с файлом модели и масштабом
    client.post("/runs/start")
    claimed = client.post(f"{API}/claim", json={"agent_id": "pc"}, headers=AUTH).json()
    nodes = {n["_meta"]["title"]: n for n in claimed["workflow"].values()}
    assert nodes["IN_UPSCALE_MODEL"]["inputs"]["unet_name"] == "seedvr2_7b_fp16.safetensors"
    assert nodes["IN_UPSCALE"]["inputs"]["resize_type.multiplier"] == 1.59
    assert claimed["bindings"]["seed"] == {"node": "IN_SEED", "field": "seed"}
    data = client.get(claimed["inputs"][0]["url"], headers=AUTH).content
    assert Image.open(io.BytesIO(data)).size == (1680, 944)  # вход — принятый вариант


def test_upscale_result_becomes_final(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    pid = in_review(client, intake, inbox, "a.png")
    v1 = variants(factory, pid)[0]
    client.post(f"/review/{pid}/accept", data={"variant_id": v1.id})
    client.post("/runs/start")
    agent_does_next(client, (2670, 1500))

    p = photo(factory, pid)
    up = variants(factory, pid)[-1]
    assert p.status == Status.FINAL_REVIEW and p.final_variant_id == up.id
    assert up.source == VariantSource.UPSCALE and up.parent_id == v1.id
    assert (up.width, up.height) == (2670, 1500) and up.prompt is None
    assert up.params["scale"] == 1.59 and up.params["workflow"] == "upscale_seedvr2"

    assert client.get("/final", follow_redirects=False).headers["location"] == f"/final/{pid}"
    page = client.get(f"/final/{pid}").text
    assert "Утвердить" in page and "Переделать апскейл" in page and "Без апскейла" in page
    assert f'"left": "v{v1.id}"' in page and f'"selected": "v{up.id}"' in page
    assert f"{up.label} · финал" in page and "2670×1500" in page  # размер — в подсказке
    assert "Финал" in client.get("/").text and f'href="/final/{pid}"' in client.get("/").text


def test_approve_delivers_under_original_name(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    pid = in_review(client, intake, inbox, "Бабушка 1952.png", album="Семья")
    v1 = variants(factory, pid)[0]
    client.post(f"/review/{pid}/accept", data={"variant_id": v1.id})
    client.post("/runs/start")
    agent_does_next(client, (2670, 1500))
    up = variants(factory, pid)[-1]

    r = client.post(f"/final/{pid}/approve", data={"variant_id": up.id}, follow_redirects=False)
    assert "утверждено → «Готово»/Семья/Бабушка 1952_restored.png" in msg(r)
    out = settings.delivered_dir / "Семья" / "Бабушка 1952_restored.png"
    p = photo(factory, pid)
    assert p.status == Status.DELIVERED and p.delivered_path == "Семья/Бабушка 1952_restored.png"
    assert p.final_variant_id == up.id

    stored = settings.store_dir / p.code / up.path
    with Image.open(out) as got, Image.open(stored) as src:
        assert np.array_equal(np.asarray(got), np.asarray(src))  # пиксели те же
        assert got.text == {
            "Title": p.code,
            "Source": "Бабушка 1952.png",
            "Restored": got.text["Restored"],
            "Software": "photo-restore",
        }  # граф ComfyUI не утёк
    assert len(got.text["Restored"]) == 10
    assert list(out.parent.glob(".*.tmp")) == []

    # повторно не утвердить: уже отдано
    r = client.post(f"/final/{pid}/approve", follow_redirects=False)
    assert "только на финале" in msg(r)


def test_accept_without_upscale_and_back(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    pid = in_review(client, intake, inbox, "scan.png")
    v3 = variants(factory, pid)[2]
    r = client.post(
        f"/review/{pid}/accept",
        data={"variant_id": v3.id, "upscale": "false"},
        follow_redirects=False,
    )
    assert "без апскейла — на финале" in msg(r)
    p = photo(factory, pid)
    assert p.status == Status.FINAL_REVIEW and p.final_variant_id == p.accepted_variant_id == v3.id
    assert last_job(factory, pid).type == JobType.EDIT_LOCAL  # апскейла нет

    r = client.post(f"/final/{pid}/back", follow_redirects=False)
    assert r.headers["location"] == f"/review/{pid}"
    p = photo(factory, pid)
    assert (
        p.status == Status.REVIEW and p.accepted_variant_id is None and p.final_variant_id is None
    )

    client.post(f"/review/{pid}/accept", data={"variant_id": v3.id, "upscale": "false"})
    client.post(f"/final/{pid}/approve")
    assert (settings.delivered_dir / "scan_restored.png").is_file()


def test_redo_and_without_upscale(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    pid = in_review(client, intake, inbox, "a.png")
    v1 = variants(factory, pid)[0]
    client.post(f"/review/{pid}/accept", data={"variant_id": v1.id})
    client.post("/runs/start")
    agent_does_next(client, (2670, 1500))

    r = client.post(f"/final/{pid}/no-upscale", follow_redirects=False)
    assert f"Финал — {v1.label} без апскейла" in msg(r)
    assert photo(factory, pid).final_variant_id == v1.id
    assert "Без апскейла</button>" not in client.get(f"/final/{pid}").text

    r = client.post(
        f"/final/{pid}/redo", data={"model": "3b_fp16", "scale": "3.0"}, follow_redirects=False
    )
    assert "апскейл в очереди (3B fp16, ×3)" in msg(r)
    assert photo(factory, pid).status == Status.QUEUED_UPSCALE
    job = last_job(factory, pid)
    assert job.payload["params"] == {"model": "3b_fp16", "scale": 3.0}
    assert job.payload["source_variant_id"] == v1.id
    client.post("/runs/start")
    agent_does_next(client, (5040, 2832))
    assert len([v for v in variants(factory, pid) if v.source == VariantSource.UPSCALE]) == 2
    page = client.get(f"/final/{pid}").text
    assert page.count('class="tile"') == 3  # два апскейла и принятый
    with factory() as s:
        kinds = [e.kind for e in s.scalars(select(Event).where(Event.photo_id == pid))]
    assert "final_set" in kinds


@pytest.mark.parametrize(
    "data",
    [{"model": "13b"}, {"model": "7b_fp16", "scale": "5"}, {"model": "7b_fp16", "scale": "x"}],
)
def test_redo_validation(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session], data: dict
) -> None:
    pid = in_review(client, intake, inbox, "a.png")
    client.post(f"/review/{pid}/accept", data={"variant_id": variants(factory, pid)[0].id})
    client.post("/runs/start")
    agent_does_next(client)
    assert client.post(f"/final/{pid}/redo", data=data).status_code == 400


def test_name_conflict_gets_photo_code(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    pids = []
    for album_name in ("x", "y"):  # одинаковые имена файлов в разных подпапках «Входящих»
        pid = queue_photo(client, intake, inbox, "same.png", album=album_name)
        with factory() as s:  # но в один альбом
            p = s.get(Photo, pid)
            assert p is not None
            p.album = "Общий"
            s.commit()
        pids.append(pid)
    client.post("/runs/start")
    for _ in pids:
        agent_does_next(client)
    for pid in pids:
        v = variants(factory, pid)[0]
        client.post(f"/review/{pid}/accept", data={"variant_id": v.id, "upscale": "false"})
    r = client.post("/bulk", data={"action": "approve", "ids": [str(i) for i in pids]})
    assert r.status_code == 200
    first, second = (photo(factory, pid) for pid in pids)
    assert first.delivered_path == "Общий/same_restored.png"
    assert second.delivered_path == f"Общий/same_restored_{second.code}.png"
    assert (
        delivery.read_text(settings.delivered_dir / second.delivered_path)["Title"] == second.code
    )


def test_bulk_approve_skips_photos_not_on_final(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    on_final = in_review(client, intake, inbox, "a.png")
    client.post(
        f"/review/{on_final}/accept",
        data={"variant_id": variants(factory, on_final)[0].id, "upscale": "false"},
    )
    in_rev = in_review(client, intake, inbox, "b.png")
    r = client.post(
        "/bulk",
        data={"action": "approve", "ids": [str(on_final), str(in_rev)]},
        follow_redirects=False,
    )
    assert f"во «Готово»: 1; не на финале или без финала: {photo(factory, in_rev).code}" in msg(r)
    assert photo(factory, on_final).status == Status.DELIVERED
    assert photo(factory, in_rev).status == Status.REVIEW


def test_bulk_priority_changes_claim_order(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    ids = [queue_photo(client, intake, inbox, f"p{i}.png") for i in range(3)]
    r = client.post(
        "/bulk", data={"action": "priority", "ids": [str(ids[2])]}, follow_redirects=False
    )
    assert "Приоритет поднят: задач 1" in msg(r)
    assert "↑1" in client.get(f"/photo/{ids[2]}").text
    client.post("/runs/start")
    claimed = client.post(f"{API}/claim", json={"agent_id": "pc"}, headers=AUTH).json()
    assert claimed["photo_id"] == ids[2]
    nothing = client.post("/bulk", data={"action": "priority", "ids": [str(ids[2])]})
    assert "нет задач в очереди" in unquote(str(nothing.url))


def test_bulk_cancel(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    a = queue_photo(client, intake, inbox, "a.png")
    b = queue_photo(client, intake, inbox, "b.png")
    client.post("/runs/start")
    taken = client.post(f"{API}/claim", json={"agent_id": "pc"}, headers=AUTH).json()
    r = client.post(
        "/bulk", data={"action": "cancel", "ids": [str(a), str(b)]}, follow_redirects=False
    )
    assert "Отменено задач: 2" in msg(r)
    for pid in (a, b):
        p = photo(factory, pid)
        assert p.status == Status.QUEUED_EDIT and p.attention
        assert last_job(factory, pid).status == JobStatus.CANCELLED
    # взятая задача: агент узнает об отмене при следующем progress
    reply = client.post(
        f"{API}/jobs/{taken['id']}/progress", json={"done": 0, "total": 3}, headers=AUTH
    )
    assert reply.status_code == 409 or reply.json()["cancel"]


def test_album_cannot_escape_delivered(tmp_path: Path) -> None:
    assert delivery.album_dir(tmp_path, "Семья/1952") == tmp_path / "Семья" / "1952"
    assert delivery.album_dir(tmp_path, "") == tmp_path
    with pytest.raises(delivery.DeliveryError):
        delivery.album_dir(tmp_path, "../etc")


def test_with_text_rejects_broken_png() -> None:
    with pytest.raises(delivery.DeliveryError, match="не PNG"):
        delivery.with_text(b"GIF89a", {"Title": "R-0001"})
    with pytest.raises(delivery.DeliveryError, match="обрезан"):
        delivery.with_text(png(20, 10)[:-30], {"Title": "R-0001"})
