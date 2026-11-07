"""Сквозной прогон без GPU (ТЗ §10): агент с FAKE_COMFY против настоящего приложения сервера.

Критерии M2: без «Запустить» агент не берёт ни одной задачи; после него по каждому фото
появляются 3 варианта; после опустошения очереди VRAM освобождается.
"""

import io
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from imgutil import encode, gradient, write
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import gemini
from app.config import Settings
from app.intake import Intake
from app.models import Job, JobStatus, Photo, Run, RunState, Status, Variant
from restore_agent import runner
from restore_agent.client import ServerClient
from restore_agent.comfy import ComfyError
from restore_agent.config import Config
from restore_agent.fake import FakeComfy
from restore_agent.runner import Agent

TOKEN = "z" * 43


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.agent_token = SecretStr(TOKEN)
    return settings


class Flaky:
    """HTTP-клиент, который первые `failures` раз роняет загрузку выходов."""

    def __init__(self, http: Any, failures: int) -> None:
        self.http = http
        self.failures = failures

    def request(self, method: str, url: str, **kw: Any) -> Any:
        if "/outputs/" in url and self.failures > 0:
            self.failures -= 1
            raise httpx.ConnectError("сеть пропала")
        return self.http.request(method, url, **kw)


def make_agent(client: TestClient, tmp_path: Path, http: Any = None) -> tuple[Agent, FakeComfy]:
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    cfg = Config(
        server_url="http://testserver",
        token=TOKEN,
        agent_id="pc",
        workdir=tmp_path / "agent",
        fake_comfy=True,
    )
    fake = FakeComfy()
    return Agent(cfg, ServerClient(http or client, backoff_s=0.01), fake), fake


@pytest.fixture
def agent(client: TestClient, tmp_path: Path) -> tuple[Agent, FakeComfy]:
    return make_agent(client, tmp_path)


def prepared_photo(client: TestClient, intake: Intake, inbox: Path, name: str, seed: int) -> int:
    write(inbox / name, encode(gradient(300, 200, seed=seed)))
    intake.poll()
    [photo_id] = intake.poll().registered  # файл стабилен два опроса подряд
    crop = {"x": 10, "y": 10, "w": 200, "h": 150}
    assert client.post(f"/prep/{photo_id}", json={"crop": crop}).status_code == 200
    return photo_id


def variants(factory: sessionmaker[Session], photo_id: int) -> list[Variant]:
    with factory() as s:
        return list(
            s.scalars(select(Variant).where(Variant.photo_id == photo_id).order_by(Variant.id))
        )


def status(factory: sessionmaker[Session], photo_id: int) -> Status:
    with factory() as s:
        p = s.get(Photo, photo_id)
        assert p is not None
        return p.status


def test_pipeline_without_gpu(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
    agent: tuple[Agent, FakeComfy],
) -> None:
    ag, fake = agent
    ids = [prepared_photo(client, intake, inbox, f"scan_{i}.png", i) for i in range(2)]

    # без «Запустить» агент не берёт ни одной задачи и не трогает ComfyUI
    ag.heartbeat()
    assert not ag.run_active and ag.comfy_ok
    for _ in range(3):
        assert ag.step() > 0
    assert fake.runs == [] and fake.inputs == {}
    assert all(status(factory, i) == Status.QUEUED_EDIT for i in ids)

    # «Запустить всё»
    r = client.post("/runs/start", follow_redirects=False)
    assert r.status_code == 303
    ag.heartbeat()
    assert ag.run_active
    assert ag.step() == 0 and ag.step() == 0

    # по каждому фото — 3 варианта с seed задачи, напечатанным на картинке
    for photo_id in ids:
        assert status(factory, photo_id) == Status.REVIEW
        vs = variants(factory, photo_id)
        assert [v.label for v in vs] == ["v001", "v002", "v003"]
        with factory() as s:
            job = s.scalar(select(Job).where(Job.photo_id == photo_id))
            assert job is not None and job.status == JobStatus.DONE
            assert [v.params["seed"] for v in vs] == job.payload["seeds"]
        images = [
            np.asarray(Image.open(settings.store_dir / f"R-{photo_id:04d}" / v.path)) for v in vs
        ]
        assert images[0].shape == (150, 200, 3)  # размер prepared.png
        assert not np.array_equal(images[0], images[1])  # разные seed — разный текст
        page = client.get(f"/review/{photo_id}").text
        assert all(f"/files/{photo_id}/variant/{v.id}/thumb" in page for v in vs)
    assert len(fake.runs) == 6

    # очередь пуста → запуск завершён, VRAM освобождается ровно один раз
    assert fake.frees == 0
    ag.step()
    assert fake.frees == 1
    with factory() as s:
        run = s.scalar(select(Run))
        assert run is not None and run.state == RunState.COMPLETED
    ag.heartbeat()
    assert not ag.run_active
    ag.step()
    assert fake.frees == 1
    assert not (ag.jobs_dir / "1").exists()  # агент убрал за собой


def test_refine_round_from_variant(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    agent: tuple[Agent, FakeComfy],
) -> None:
    ag, _ = agent
    prepared_photo(client, intake, inbox, "a.png", 1)
    client.post("/runs/start")
    ag.heartbeat()
    ag.step()
    v2 = variants(factory, 1)[1]

    r = client.post(
        "/review/1/refine",
        data={
            "source": "variant",
            "variant_id": str(v2.id),
            "text": "Brighten the sky.",
            "variants": "2",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert status(factory, 1) == Status.QUEUED_EDIT
    client.post("/runs/start")
    ag.heartbeat()
    ag.step()

    vs = variants(factory, 1)
    assert [v.round for v in vs] == [1, 1, 1, 2, 2]
    assert all(v.parent_id == v2.id for v in vs[3:])
    assert vs[3].prompt is not None and vs[3].prompt.endswith("Brighten the sky.")
    # вход второго раунда — сам вариант v002
    with factory() as s:
        job = s.scalars(select(Job).order_by(Job.id.desc())).first()
        assert job is not None and job.payload["source_variant_id"] == v2.id
    assert status(factory, 1) == Status.REVIEW


def test_cancel_during_render(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    agent: tuple[Agent, FakeComfy],
) -> None:
    ag, fake = agent
    prepared_photo(client, intake, inbox, "a.png", 1)
    client.post("/runs/start")
    ag.heartbeat()
    fake.before_run = lambda: client.post("/jobs/1/cancel")
    ag.step()
    assert fake.interrupts == 1 and fake.runs == []
    with factory() as s:
        job = s.get(Job, 1)
        photo = s.get(Photo, 1)
        assert job is not None and job.status == JobStatus.CANCELLED
        assert photo is not None and photo.status == Status.QUEUED_EDIT and photo.attention
    assert variants(factory, 1) == []
    assert not (ag.jobs_dir / "1").exists()


def test_missing_model_fails_without_retry(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    agent: tuple[Agent, FakeComfy],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ag, fake = agent
    prepared_photo(client, intake, inbox, "a.png", 1)
    client.post("/runs/start")
    ag.heartbeat()

    def no_model(workflow: dict[str, Any]) -> None:
        raise ComfyError("нет файла qwen.safetensors в models/…", retryable=False)

    monkeypatch.setattr(fake, "check_models", no_model)
    ag.step()
    with factory() as s:
        job = s.get(Job, 1)
        photo = s.get(Photo, 1)
        assert job is not None and job.status == JobStatus.FAILED and job.attempts == 1
        assert photo is not None and photo.attention
        assert "qwen.safetensors" in (photo.attention_reason or "")
    assert "qwen.safetensors" in client.get("/photo/1").text


def test_upload_survives_network_loss(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    ag, _ = make_agent(client, tmp_path, http=Flaky(client, failures=3))
    prepared_photo(client, intake, inbox, "a.png", 1)
    client.post("/runs/start")
    ag.heartbeat()
    ag.step()
    assert len(variants(factory, 1)) == 3
    assert status(factory, 1) == Status.REVIEW


def test_results_in_workdir_are_reused_after_restart(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    agent: tuple[Agent, FakeComfy],
) -> None:
    ag, fake = agent
    prepared_photo(client, intake, inbox, "a.png", 1)
    with factory() as s:
        seed0 = s.get(Job, 1).payload["seeds"][0]  # type: ignore[union-attr]
    job_dir = ag.jobs_dir / "1"
    job_dir.mkdir(parents=True)
    buf = io.BytesIO()
    Image.new("RGB", (200, 150), (9, 9, 9)).save(buf, format="PNG")
    (job_dir / "out_0.png").write_bytes(buf.getvalue())
    (job_dir / "out_0.json").write_text(json.dumps({"seed": seed0, "duration_s": 1.0}))

    client.post("/runs/start")
    ag.heartbeat()
    ag.step()
    assert len(fake.runs) == 2  # первый выход уже был готов
    [v1, *_] = variants(factory, 1)
    assert v1.duration_s == 1.0


def test_outdated_agent_takes_nothing(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    agent: tuple[Agent, FakeComfy],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ag, fake = agent
    monkeypatch.setattr(runner, "__version__", "0.1.0")
    prepared_photo(client, intake, inbox, "a.png", 1)
    client.post("/runs/start")
    ag.heartbeat()
    assert ag.outdated and not ag.run_active
    ag.step()
    assert fake.runs == []
    assert "Обновите агента" in client.get("/pc").text


def test_stop_mid_job_finishes_it_and_frees_vram(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    agent: tuple[Agent, FakeComfy],
) -> None:
    ag, fake = agent
    prepared_photo(client, intake, inbox, "a.png", 1)
    prepared_photo(client, intake, inbox, "b.png", 2)
    client.post("/runs/start")
    ag.heartbeat()

    def stop_once() -> None:
        fake.before_run = None
        client.post("/runs/stop")
        ag.heartbeat()  # пульс узнал про «Стоп», пока идёт задача

    fake.before_run = stop_once
    assert ag.step() == 0
    assert len(variants(factory, 1)) == 3  # взятая задача доделана
    assert status(factory, 2) == Status.QUEUED_EDIT
    assert fake.frees == 0
    ag.step()  # запуска нет — модели выгружаются, задачи не берутся
    ag.step()
    assert fake.frees == 1 and len(fake.runs) == 3


class Rejecting:
    """HTTP-клиент, на котором сервер отвергает вызовы с `parts` в пути (4xx)."""

    def __init__(self, http: Any, *parts: str) -> None:
        self.http = http
        self.parts = parts

    def request(self, method: str, url: str, **kw: Any) -> Any:
        if any(p in url for p in self.parts):
            return httpx.Response(400, json={"detail": "отвергнуто"})
        return self.http.request(method, url, **kw)


@pytest.mark.parametrize("fail_accepted", [True, False])
def test_server_rejection_mid_job_does_not_kill_agent(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    tmp_path: Path,
    fail_accepted: bool,
) -> None:
    parts = ("/outputs/",) if fail_accepted else ("/outputs/", "/fail")
    ag, _ = make_agent(client, tmp_path, http=Rejecting(client, *parts))
    prepared_photo(client, intake, inbox, "a.png", 1)
    client.post("/runs/start")
    ag.heartbeat()
    assert ag.step() == 0  # раньше здесь агент падал целиком
    with factory() as s:
        job = s.get(Job, 1)
        assert job is not None
        if fail_accepted:  # задача вернулась в очередь с попыткой
            assert job.status == JobStatus.QUEUED and job.attempts == 1
            assert "отвергнуто" in (job.error or "")
        else:  # fail тоже не принят — задачу вернёт аренда
            assert job.status == JobStatus.CLAIMED
    assert (ag.jobs_dir / "1" / "out_0.png").is_file()  # готовый результат не выброшен


def test_full_pipeline_to_delivered(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
    agent: tuple[Agent, FakeComfy],
) -> None:
    """ТЗ §10: файл во «Входящих» → подготовка → run → 3 варианта → принять → апскейл →
    утвердить → файл во «Готово» под исходным именем с `_restored`."""
    ag, fake = agent
    pid = prepared_photo(client, intake, inbox, "Семья/Дедушка 1938.png", 3)
    client.post("/runs/start")
    ag.heartbeat()
    assert ag.step() == 0
    v2 = variants(factory, pid)[1]

    client.post(f"/review/{pid}/accept", data={"variant_id": v2.id})
    assert status(factory, pid) == Status.QUEUED_UPSCALE
    ag.step()  # прошлый запуск кончился — без нового агент апскейл не берёт
    assert status(factory, pid) == Status.QUEUED_UPSCALE
    client.post("/runs/start")
    ag.heartbeat()
    assert ag.step() == 0

    up = variants(factory, pid)[-1]
    assert status(factory, pid) == Status.FINAL_REVIEW
    assert up.source == "upscale" and (up.width, up.height) == (400, 300)  # 200×150, авто ×2
    graph = fake.runs[-1]
    assert any(
        n["inputs"].get("unet_name") == "seedvr2_7b_fp16.safetensors" for n in graph.values()
    )

    r = client.post(f"/final/{pid}/approve", follow_redirects=False)
    assert r.status_code == 303
    out = settings.delivered_dir / "Семья" / "Дедушка 1938_restored.png"
    with Image.open(out) as img:
        assert img.size == (400, 300) and img.text["Title"] == f"R-{pid:04d}"
    assert status(factory, pid) == Status.DELIVERED
    ag.step()
    assert fake.frees == 2  # после каждого запуска


def test_ten_selected_photos_run_with_one_button(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    agent: tuple[Agent, FakeComfy],
) -> None:
    """Критерий M3: выбранные 10 фото запускаются одной кнопкой — и чистка, и апскейл."""
    ag, _ = agent
    ids = [prepared_photo(client, intake, inbox, f"s{i:02d}.png", i) for i in range(12)]
    # двое уже прошли чистку и ждут апскейла
    client.post("/bulk", data={"action": "run", "ids": [str(i) for i in ids[:2]]})
    ag.heartbeat()
    while ag.step() == 0:
        pass
    for pid in ids[:2]:
        client.post(f"/review/{pid}/accept", data={"variant_id": variants(factory, pid)[0].id})

    selected = ids[:10]
    r = client.post(
        "/bulk", data={"action": "run", "ids": [str(i) for i in selected]}, follow_redirects=False
    )
    assert "задач 10" in unquote(r.headers["location"])
    ag.heartbeat()
    while ag.step() == 0:
        pass
    assert [status(factory, i) for i in ids] == [Status.FINAL_REVIEW] * 2 + [Status.REVIEW] * 8 + [
        Status.QUEUED_EDIT
    ] * 2
    with factory() as s:
        run = s.scalars(select(Run).order_by(Run.id.desc())).first()
        assert run is not None and run.state == RunState.COMPLETED and len(run.job_ids or []) == 10


def test_full_pipeline_with_gemini(
    app: FastAPI,
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
    agent: tuple[Agent, FakeComfy],
) -> None:
    """ТЗ §10 с FAKE_COMFY + FAKE_GEMINI: 3 варианта → вариант в Gemini (без запуска) →
    принять его → апскейл → утвердить → «Готово»."""
    app.state.gemini.backend = gemini.FakeBackend()
    app.state.gemini.poll_s = 0.05
    ag, _ = agent
    pid = prepared_photo(client, intake, inbox, "Gemini.png", 4)
    client.post("/runs/start")
    ag.heartbeat()
    assert ag.step() == 0
    src = variants(factory, pid)[0]

    client.post(
        f"/review/{pid}/gemini",
        data={"source": "variant", "variant_id": src.id, "reference": "true"},
    )
    deadline = time.monotonic() + 10
    while status(factory, pid) != Status.REVIEW:
        assert time.monotonic() < deadline, "Gemini не ответил"
        time.sleep(0.02)
    gem = variants(factory, pid)[-1]
    assert gem.source == "gemini" and gem.cost_usd == 0 and gem.parent_id == src.id
    assert len(app.state.gemini.backend.requests[0].images) == 2

    client.post(f"/review/{pid}/accept", data={"variant_id": gem.id})
    client.post("/runs/start")
    ag.heartbeat()
    assert ag.step() == 0
    up = variants(factory, pid)[-1]
    assert up.source == "upscale" and up.parent_id == gem.id
    client.post(f"/final/{pid}/approve")
    assert status(factory, pid) == Status.DELIVERED
    assert (settings.delivered_dir / "Gemini_restored.png").is_file()
