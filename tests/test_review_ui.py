"""Интерфейс M2: панель «ПК», запуски, экран проверки."""

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient
from imgutil import encode, gradient, write
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import runs
from app.config import Settings
from app.intake import Intake
from app.models import (
    Event,
    Job,
    JobStatus,
    JobType,
    Photo,
    Run,
    RunScope,
    RunState,
    Status,
    Variant,
)

TOKEN = "q" * 43
AUTH = {"Authorization": f"Bearer {TOKEN}"}
API = "/api/agent/v1"


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.agent_token = SecretStr(TOKEN)
    return settings


def queue_photo(client: TestClient, intake: Intake, inbox: Path, name: str, seed: int = 0) -> int:
    write(inbox / name, encode(gradient(240, 160, seed=seed)))
    intake.trust(name)
    [photo_id] = intake.poll().registered
    client.post(f"/prep/{photo_id}", json={"crop": None})
    return photo_id


def finish_job(client: TestClient) -> dict[str, Any]:
    """Агент берёт задачу и отдаёт три результата."""
    job = client.post(f"{API}/claim", json={"agent_id": "pc"}, headers=AUTH).json()
    for out in job["outputs"]:
        data = encode(gradient(160, 100, seed=out["index"] + 7))
        client.post(
            f"{API}/jobs/{job['id']}/outputs/{out['index']}",
            headers=AUTH,
            files={"file": ("o.png", data, "image/png")},
            data={"meta": json.dumps({"seed": out["seed"]})},
        )
    assert client.post(f"{API}/jobs/{job['id']}/complete", headers=AUTH).status_code == 200
    return job


def reviewed(client: TestClient, intake: Intake, inbox: Path, n: int = 1) -> list[int]:
    ids = [queue_photo(client, intake, inbox, f"p{i}.png", i) for i in range(n)]
    client.post("/runs/start")
    for _ in ids:
        finish_job(client)
    return ids


def msg(r: Any) -> str:
    return unquote(r.headers["location"])


def test_pc_page_and_run_controls(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    page = client.get("/pc").text
    assert "Агент ещё не подключался" in page
    assert "Запустить всё" in page and "disabled" in page

    queue_photo(client, intake, inbox, "a.png")
    assert "Запустить всё (1)" in client.get("/pc").text
    r = client.post("/runs/start", follow_redirects=False)
    assert "Запуск #1 начат" in msg(r)
    assert "Запуск #1 уже идёт" in msg(client.post("/runs/start", follow_redirects=False))
    header = client.get("/partials/header").text
    assert "Запуск #1: 0 из 1" in header

    body = client.get("/partials/pc").text
    assert "Стоп" in body
    r = client.post("/runs/stop", follow_redirects=False)
    assert "остановлен" in msg(r)
    with factory() as s:
        run = s.get(Run, 1)
        assert run is not None and run.state == RunState.STOPPED
    assert "Запуска нет" in client.get("/partials/header").text
    assert "Запуска нет" in msg(client.post("/runs/stop", follow_redirects=False))


def test_pc_shows_agent_and_estimate(client: TestClient, intake: Intake, inbox: Path) -> None:
    client.post(
        f"{API}/hello",
        headers=AUTH,
        json={
            "agent_id": "pc",
            "version": "0.2.0",
            "comfy_ok": False,
            "gpu_name": "RTX 3090 Ti",
            "vram_free_mb": 20480,
        },
    )
    reviewed(client, intake, inbox, 1)
    queue_photo(client, intake, inbox, "b.png", 5)
    page = client.get("/pc").text
    assert "RTX 3090 Ti" in page and "20.0 ГБ" in page and "не запущен" in page
    assert "ComfyUI не запущен" in client.get("/partials/header").text
    assert "чистка на ПК" in page


def test_run_selected_from_board(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    queue_photo(client, intake, inbox, "a.png", 1)
    queue_photo(client, intake, inbox, "b.png", 2)
    r = client.post("/bulk", data={"action": "run", "ids": ["2"]}, follow_redirects=False)
    assert "Запуск #1: задач 1" in msg(r)
    with factory() as s:
        run = s.get(Run, 1)
        assert run is not None and run.scope == RunScope.SELECTED and run.job_ids == [2]
    assert "Запустить выбранные" in client.get("/").text


def test_run_selected_without_jobs(client: TestClient, intake: Intake, inbox: Path) -> None:
    write(inbox / "a.png", encode(gradient(50, 50)))
    intake.trust("a.png")
    intake.poll()
    r = client.post("/bulk", data={"action": "run", "ids": ["1"]}, follow_redirects=False)
    assert "нет задач в очереди" in msg(r)


def test_review_page(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    ids = reviewed(client, intake, inbox, 3)
    assert client.get("/review", follow_redirects=False).headers["location"] == "/review/1"
    page = client.get("/review/2").text
    assert "R-0002" in page
    for v in range(4, 7):
        assert f"/files/2/variant/{v}/thumb" in page
    assert 'href="/review/1"' in page and 'href="/review/3"' in page  # ← →
    assert "Доработать локально" in page and "Сильнее чистка" in page
    assert '"selected": "v4"' in page
    assert "Локально" in page and "seed" in page
    assert "Проверить (3)" in client.get("/photo/2").text
    # доска ведёт карточки REVIEW на проверку
    assert 'href="/review/2"' in client.get("/").text
    assert len(ids) == 3


def test_variant_files_and_hide(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    reviewed(client, intake, inbox, 1)
    r = client.get("/files/1/variant/2?download=1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert 'filename="R-0001_v002.png"' in r.headers["content-disposition"]
    assert client.get("/files/1/preview/v002").headers["content-type"] == "image/jpeg"
    assert client.get("/files/1/variant/2/thumb").headers["content-type"] == "image/webp"
    assert client.get("/files/2/variant/2").status_code == 404

    r = client.post("/variants/2/hide", follow_redirects=False)
    assert r.headers["location"] == "/review/1?v=2"
    page = client.get("/review/1").text
    assert "/files/1/variant/2/thumb" not in page and "Показать скрытые (1)" in page
    assert "/files/1/variant/2/thumb" in client.get("/review/1?hidden=1").text
    client.post("/variants/2/show")
    with factory() as s:
        v = s.get(Variant, 2)
        assert v is not None and not v.hidden
        kinds = [e.kind for e in s.scalars(select(Event).where(Event.photo_id == 1))]
        assert "variant_hide" in kinds and "variant_show" in kinds


def test_refine_composes_prompt_and_moves_on(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    reviewed(client, intake, inbox, 2)
    page = client.get("/review/1").text
    addon_id = page.split('name="addons" value="')[1].split('"')[0]
    r = client.post(
        "/review/1/refine",
        data={
            "addons": [addon_id],
            "text": "Fix the left eye.",
            "source": "prepared",
            "variants": "4",
        },
        follow_redirects=False,
    )
    assert msg(r).startswith("/review/2?msg=R-0001: доработка в очереди (раунд 2, вариантов 4)")
    with factory() as s:
        photo = s.get(Photo, 1)
        job = s.scalars(select(Job).where(Job.photo_id == 1).order_by(Job.id.desc())).first()
        assert photo is not None and photo.status == Status.QUEUED_EDIT and photo.round == 2
        assert job is not None and job.status == JobStatus.QUEUED
        assert job.type == JobType.EDIT_LOCAL and len(job.payload["seeds"]) == 4
        assert job.payload["round"] == 2 and job.payload["source_variant_id"] is None
        prompt = job.payload["prompt"]
    assert prompt.startswith("Restore this old photograph.")
    assert "Remove damage more aggressively" in prompt
    assert prompt.endswith("Fix the left eye.")

    # повторно — уже не в REVIEW
    r = client.post("/review/1/refine", data={"source": "prepared"}, follow_redirects=False)
    assert "запрещён" in msg(r)


@pytest.mark.parametrize(
    "data",
    [
        {"source": "variant"},
        {"source": "variant", "variant_id": "999"},
        {"source": "prepared", "variants": "0"},
        {"source": "prepared", "variants": "7"},
    ],
)
def test_refine_validation(
    client: TestClient, intake: Intake, inbox: Path, data: dict[str, str]
) -> None:
    reviewed(client, intake, inbox, 1)
    assert client.post("/review/1/refine", data=data).status_code in (400, 404, 422)


def test_review_empty_states(client: TestClient, intake: Intake, inbox: Path) -> None:
    assert "Нет фото на проверке" in msg(client.get("/review", follow_redirects=False))
    queue_photo(client, intake, inbox, "a.png")
    page = client.get("/review/1").text
    assert "Вариантов пока нет — они появятся после обработки на ПК" in page


def test_header_progress_counts_finished(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    queue_photo(client, intake, inbox, "a.png", 1)
    queue_photo(client, intake, inbox, "b.png", 2)
    client.post("/runs/start")
    finish_job(client)
    assert "Запуск #1: 1 из 2" in client.get("/partials/header").text
    with factory() as s:
        assert runs.active_run(s) is not None
