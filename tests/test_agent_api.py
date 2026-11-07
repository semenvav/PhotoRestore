"""Протокол агента (ТЗ §6, §10 «Протокол»)."""

import io
import json
import os
import stat
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from imgutil import encode, gradient, write
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app import jobs, runs
from app.config import Settings
from app.db import utcnow
from app.intake import Intake
from app.main import create_app
from app.models import (
    Agent,
    Event,
    Job,
    JobStatus,
    JobType,
    Photo,
    Run,
    RunState,
    Status,
    Variant,
)

TOKEN = "x" * 43
AUTH = {"Authorization": f"Bearer {TOKEN}"}
API = "/api/agent/v1"
EDIT = {JobType.EDIT_LOCAL}


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.agent_token = SecretStr(TOKEN)
    return settings


def post(client: TestClient, path: str, body: dict[str, Any] | None = None, **kw: Any) -> Any:
    return client.post(f"{API}{path}", json=body or {}, headers=AUTH, **kw)


def queue_photo(client: TestClient, intake: Intake, inbox: Path, name: str, seed: int = 0) -> int:
    """Фото, прошедшее подготовку: QUEUED_EDIT и задача edit_local на 3 варианта."""
    write(inbox / name, encode(gradient(240, 160, seed=seed)))
    intake.trust(name)
    [photo_id] = intake.poll().registered
    assert client.post(f"/prep/{photo_id}", json={"crop": None}).status_code == 200
    return photo_id


def start_run(factory: sessionmaker[Session], photo_ids: list[int] | None = None) -> int:
    with factory() as s:
        run = runs.start(s, EDIT, photo_ids)
        s.commit()
        return run.id


def photo(factory: sessionmaker[Session], photo_id: int = 1) -> Photo:
    with factory() as s:
        p = s.get(Photo, photo_id)
        assert p is not None
        return p


def job(factory: sessionmaker[Session], job_id: int) -> Job:
    with factory() as s:
        j = s.get(Job, job_id)
        assert j is not None
        return j


def claim(client: TestClient, prefer: str | None = None) -> Any:
    return post(client, "/claim", {"agent_id": "pc", "prefer_type": prefer})


def png(seed: int, size: tuple[int, int] = (120, 80)) -> bytes:
    return encode(gradient(*size, seed=seed))


def upload(client: TestClient, job_id: int, index: int, data: bytes, **meta: Any) -> Any:
    return client.post(
        f"{API}/jobs/{job_id}/outputs/{index}",
        headers=AUTH,
        files={"file": (f"out_{index}.png", data, "image/png")},
        data={"meta": json.dumps(meta)},
    )


# --- доступ ---------------------------------------------------------------------------------


def test_token_required(client: TestClient) -> None:
    body = {"agent_id": "pc", "version": "0.2.0"}
    assert client.post(f"{API}/hello", json=body).status_code == 401
    bad = {"Authorization": "Bearer " + "y" * 43}
    assert client.post(f"{API}/hello", json=body, headers=bad).status_code == 401
    assert client.post(f"{API}/hello", json=body, headers=AUTH).status_code == 200


def test_without_token_agent_api_is_off(tmp_path: Path) -> None:
    s = Settings(
        _env_file=None,
        data_dir=tmp_path / "d",
        db_path=tmp_path / "db.sqlite",  # type: ignore[call-arg]
        intake_enabled=False,
    )
    with TestClient(create_app(s)) as c:
        r = c.post(f"{API}/hello", json={"agent_id": "pc", "version": "0.2.0"}, headers=AUTH)
        assert r.status_code == 503


def test_hello_records_agent_and_rejects_old_version(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    r = post(
        client,
        "/hello",
        {
            "agent_id": "pc",
            "version": "0.2.0",
            "comfy_ok": True,
            "gpu_name": "RTX 3090 Ti",
            "vram_free_mb": 23000,
        },
    )
    assert r.json() == {"run_active": False, "poll_interval_s": 15}
    with factory() as s:
        agent = s.get(Agent, "pc")
        assert agent is not None and agent.comfy_ok and agent.gpu_name == "RTX 3090 Ti"
        assert agent.last_seen is not None

    r = post(client, "/hello", {"agent_id": "pc", "version": "0.1.9"})
    assert r.status_code == 426
    assert "0.2.0" in r.json()["error"]
    assert "Обновите агента" in client.get("/partials/header").text


# --- выдача только в run --------------------------------------------------------------------


def test_claim_without_run_is_204(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    queue_photo(client, intake, inbox, "a.png")
    assert claim(client).status_code == 204
    assert photo(factory).status == Status.QUEUED_EDIT
    assert job(factory, 1).status == JobStatus.QUEUED


def test_claim_gives_job_object(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    queue_photo(client, intake, inbox, "a.png")
    run_id = start_run(factory)
    assert post(client, "/hello", {"agent_id": "pc", "version": "0.2.0"}).json() == {
        "run_active": True,
        "poll_interval_s": 5,
    }

    r = claim(client)
    assert r.status_code == 200
    body = r.json()
    payload = job(factory, 1).payload
    assert body["id"] == 1 and body["type"] == "edit_local" and body["photo_id"] == 1
    assert body["inputs"] == [{"name": "source", "url": f"{API}/jobs/1/inputs/source"}]
    assert body["outputs"] == [{"index": i, "seed": s} for i, s in enumerate(payload["seeds"])]
    assert len(body["outputs"]) == 3
    assert body["bindings"]["image"] == {"node": "IN_IMAGE", "field": "image"}
    titles = {n["_meta"]["title"]: n for n in body["workflow"].values()}
    assert titles["IN_PROMPT"]["inputs"]["prompt"] == payload["prompt"]
    assert titles["IN_STEPS"]["inputs"]["value"] == 8
    assert body["lease_until"].endswith("Z")

    j = job(factory, 1)
    assert j.status == JobStatus.CLAIMED and j.agent_id == "pc" and j.run_id == run_id
    assert j.lease_until is not None
    assert abs((j.lease_until - utcnow()).total_seconds() - settings.lease_s) < 5
    assert j.payload["workflow"]["workflow"] == "edit_qwen2511"
    assert photo(factory).status == Status.EDITING

    r = client.get(f"{API}/jobs/1/inputs/source", headers=AUTH)
    assert r.content == (settings.store_dir / "R-0001" / "prepared.png").read_bytes()
    assert claim(client).status_code == 204  # больше задач нет


def test_concurrent_claims_never_share_a_job(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    for i in range(3):
        queue_photo(client, intake, inbox, f"p{i}.png", seed=i)
    start_run(factory)
    got: list[int | None] = []
    barrier = threading.Barrier(6)

    def worker(n: int) -> None:
        with factory() as s:
            barrier.wait()
            j = runs.claim(s, f"agent{n}", None, 600, EDIT)
            s.commit()
            got.append(j.id if j else None)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    claimed = [g for g in got if g is not None]
    assert sorted(claimed) == [1, 2, 3]  # каждая задача — ровно одному
    assert got.count(None) == 3


def test_selected_scope_and_auto_complete(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    queue_photo(client, intake, inbox, "a.png", 1)
    queue_photo(client, intake, inbox, "b.png", 2)
    run_id = start_run(factory, [2])
    queue_photo(client, intake, inbox, "c.png", 3)  # после старта — не в запуске

    body = claim(client).json()
    assert body["photo_id"] == 2
    for i in range(3):
        assert upload(client, body["id"], i, png(i)).status_code == 200
    assert post(client, f"/jobs/{body['id']}/complete").status_code == 200

    with factory() as s:
        run = s.get(Run, run_id)
        assert run is not None and run.state == RunState.COMPLETED
    assert claim(client).status_code == 204
    assert photo(factory, 1).status == Status.QUEUED_EDIT
    assert photo(factory, 3).status == Status.QUEUED_EDIT


def test_scope_all_takes_new_jobs_and_priority(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    queue_photo(client, intake, inbox, "a.png", 1)
    start_run(factory)
    queue_photo(client, intake, inbox, "b.png", 2)
    with factory() as s:
        s.execute(update(Job).where(Job.photo_id == 2).values(priority=5))
        s.commit()
    assert claim(client).json()["photo_id"] == 2
    assert claim(client).json()["photo_id"] == 1


def test_prefer_type_goes_first(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    queue_photo(client, intake, inbox, "a.png", 1)
    queue_photo(client, intake, inbox, "b.png", 2)
    with factory() as s:  # M3 ещё нет: апскейл подделываем прямо в БД
        s.execute(update(Photo).where(Photo.id == 2).values(status=Status.QUEUED_UPSCALE))
        s.execute(update(Job).where(Job.photo_id == 2).values(type=JobType.UPSCALE_LOCAL))
        runs.start(s, {JobType.EDIT_LOCAL, JobType.UPSCALE_LOCAL})
        s.commit()
    both = {JobType.EDIT_LOCAL, JobType.UPSCALE_LOCAL}
    with factory() as s:
        j = runs.claim(s, "pc", JobType.UPSCALE_LOCAL, 600, both)
        assert j is not None and j.photo_id == 2
        assert s.get(Photo, 2).status == Status.UPSCALING  # type: ignore[union-attr]


def test_stop_blocks_new_but_running_job_finishes(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    queue_photo(client, intake, inbox, "a.png", 1)
    queue_photo(client, intake, inbox, "b.png", 2)
    start_run(factory)
    body = claim(client).json()
    with factory() as s:
        runs.stop(s)
        s.commit()
    assert claim(client).status_code == 204
    for i in range(3):
        upload(client, body["id"], i, png(i))
    assert post(client, f"/jobs/{body['id']}/complete").status_code == 200
    assert photo(factory, body["photo_id"]).status == Status.REVIEW


# --- результаты -----------------------------------------------------------------------------


def test_outputs_complete_and_idempotency(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    queue_photo(client, intake, inbox, "a.png")
    start_run(factory)
    body = claim(client).json()
    seeds = [o["seed"] for o in body["outputs"]]

    assert upload(client, 1, 0, png(10), seed=seeds[0], duration_s=31.5).status_code == 200
    assert upload(client, 1, 1, png(99), seed=seeds[1]).status_code == 200
    assert upload(client, 1, 1, png(11, (150, 100)), seed=seeds[1]).status_code == 200
    r = post(client, "/jobs/1/complete")
    assert r.status_code == 409 and "2" in r.json()["detail"]  # третьего ещё нет
    assert upload(client, 1, 2, png(12), seed=seeds[2]).status_code == 200

    r = post(client, "/jobs/1/complete")
    assert r.status_code == 200
    assert r.json()["variants"] == [1, 2, 3]
    assert post(client, "/jobs/1/complete").status_code == 200  # повтор — 200
    assert upload(client, 1, 0, png(13)).json()["ignored"]

    p = photo(factory)
    assert p.status == Status.REVIEW and not p.attention
    with factory() as s:
        variants = s.scalars(select(Variant).order_by(Variant.id)).all()
    assert [v.label for v in variants] == ["v001", "v002", "v003"]
    assert [v.params["seed"] for v in variants] == seeds
    assert variants[0].duration_s == 31.5
    assert variants[1].width == 150  # перезаписанный выход
    assert variants[0].params["workflow"] == "edit_qwen2511"
    assert variants[0].params["steps"] == 8
    assert all(v.round == 1 and v.job_id == 1 for v in variants)

    photo_dir = settings.store_dir / "R-0001"
    v1 = photo_dir / "variants" / "v001.png"
    assert v1.read_bytes() == png(10)
    assert not os.stat(v1).st_mode & stat.S_IWUSR
    assert (photo_dir / "cache" / "preview_v001.jpg").is_file()
    assert (photo_dir / "cache" / "thumb_v001.webp").is_file()
    assert not (photo_dir / "jobs" / "1").exists()
    assert job(factory, 1).status == JobStatus.DONE


@pytest.mark.parametrize(
    ("index", "data", "code"),
    [
        (3, png(1), 400),
        (-1, png(1), 400),
        (0, encode(gradient(40, 40), "JPEG"), 400),
        (0, b"not an image", 400),
    ],
)
def test_bad_outputs_rejected(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    index: int,
    data: bytes,
    code: int,
) -> None:
    queue_photo(client, intake, inbox, "a.png")
    start_run(factory)
    claim(client)
    assert upload(client, 1, index, data).status_code in (code, 404, 422)


# --- аренда, отказ, отмена ------------------------------------------------------------------


def expire(factory: sessionmaker[Session]) -> list[int]:
    with factory() as s:
        ids = jobs.expire_leases(s, utcnow() + timedelta(minutes=11))
        s.commit()
        return ids


def test_lease_expiry_returns_job_then_fails_after_three(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    queue_photo(client, intake, inbox, "a.png")
    start_run(factory)
    for attempt in (1, 2):
        assert claim(client).status_code == 200
        assert expire(factory) == [1]
        j = job(factory, 1)
        assert j.status == JobStatus.QUEUED and j.attempts == attempt and j.agent_id is None
        assert photo(factory).status == Status.QUEUED_EDIT and not photo(factory).attention

    claim(client)
    # «убитый» агент вернулся: задача уже не его
    expire(factory)
    assert post(client, "/jobs/1/progress", {"done": 1, "total": 3}).json() == {"cancel": True}
    assert upload(client, 1, 0, png(1)).status_code == 409
    j = job(factory, 1)
    assert j.status == JobStatus.FAILED and j.attempts == 3
    p = photo(factory)
    assert p.status == Status.QUEUED_EDIT and p.attention
    assert "аренды" in (p.attention_reason or "")
    assert claim(client).status_code == 204


def test_progress_extends_lease(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    queue_photo(client, intake, inbox, "a.png")
    start_run(factory)
    claim(client)
    with factory() as s:
        s.execute(update(Job).values(lease_until=utcnow() + timedelta(seconds=5)))
        s.commit()
    r = post(client, "/jobs/1/progress", {"done": 1, "total": 3, "stage": "seed 2"})
    assert r.json() == {"cancel": False}
    j = job(factory, 1)
    assert j.lease_until is not None and j.lease_until > utcnow() + timedelta(minutes=9)
    assert j.progress == {"done": 1, "total": 3, "stage": "seed 2"}


def test_fail_retryable_and_final(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    queue_photo(client, intake, inbox, "a.png")
    start_run(factory)
    claim(client)
    r = post(client, "/jobs/1/fail", {"error": "ComfyUI упал", "retryable": True})
    assert r.json()["status"] == "queued"
    assert photo(factory).status == Status.QUEUED_EDIT and not photo(factory).attention

    claim(client)
    r = post(
        client, "/jobs/1/fail", {"error": "нет файла x.safetensors в models/", "retryable": False}
    )
    assert r.json()["status"] == "failed"
    p = photo(factory)
    assert p.status == Status.QUEUED_EDIT and p.attention
    assert "x.safetensors" in (p.attention_reason or "")
    assert post(client, "/jobs/1/fail", {"error": "again"}).json()["ignored"]


def test_cancel_running_job_and_retry(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    queue_photo(client, intake, inbox, "a.png")
    start_run(factory)
    claim(client)
    upload(client, 1, 0, png(1))

    r = client.post("/jobs/1/cancel", follow_redirects=False)
    assert r.status_code == 303
    assert post(client, "/jobs/1/progress").json() == {"cancel": True}
    assert post(client, "/jobs/1/complete").status_code == 409
    p = photo(factory)
    assert p.status == Status.QUEUED_EDIT and p.attention
    assert job(factory, 1).status == JobStatus.CANCELLED

    card = client.get("/photo/1").text
    assert "Повторить" in card
    client.post("/photo/1/retry")
    p = photo(factory)
    assert not p.attention
    j2 = job(factory, 2)
    assert j2.status == JobStatus.QUEUED and j2.payload["seeds"] == job(factory, 1).payload["seeds"]
    # запуск «всё» ещё идёт — повторённую задачу агент возьмёт сразу
    assert claim(client).json()["id"] == 2


def test_sweeper_returns_job_of_killed_agent(tmp_path: Path) -> None:
    """Критерий M2: убитый агент → задача возвращается в очередь (здесь — со сжатым временем)."""
    s = Settings(
        _env_file=None,
        data_dir=tmp_path / "d",
        db_path=tmp_path / "db.sqlite",  # type: ignore[call-arg]
        intake_enabled=False,
        agent_token=SecretStr(TOKEN),
        lease_s=0.3,
        sweep_interval_s=0.1,
    )
    app = create_app(s)
    with TestClient(app) as c:
        intake: Intake = app.state.intake
        factory: sessionmaker[Session] = app.state.factory
        queue_photo(c, intake, s.inbox_dir, "a.png")
        start_run(factory)
        assert claim(c).status_code == 200  # и агент «умер»
        deadline = time.monotonic() + 5
        while job(factory, 1).status != JobStatus.QUEUED and time.monotonic() < deadline:
            time.sleep(0.05)
        assert job(factory, 1).status == JobStatus.QUEUED
        assert photo(factory).status == Status.QUEUED_EDIT
        with factory() as session:
            ev = session.scalars(select(Event).where(Event.kind == "status")).all()[-1]
            assert ev.data["to"] == "QUEUED_EDIT" and ev.actor == "system"


def test_variant_png_is_valid(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    queue_photo(client, intake, inbox, "a.png")
    start_run(factory)
    claim(client)
    for i in range(3):
        upload(client, 1, i, png(i))
    post(client, "/jobs/1/complete")
    img = Image.open(
        io.BytesIO((settings.store_dir / "R-0001" / "variants" / "v002.png").read_bytes())
    )
    assert img.size == (120, 80)
