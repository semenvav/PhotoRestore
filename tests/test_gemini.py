"""M4: исполнитель Gemini — промпт, пропорции, стоимость, бюджет, воркер, отказ, повторы."""

import asyncio
import base64
import io
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.genai import types
from imgutil import gradient
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from test_final import TOKEN, in_review, last_job, msg, photo, queue_photo, variants

from app import gemini, presets
from app.config import Settings
from app.gemini import GeminiAPIError, Reply, Request, Usage
from app.intake import Intake
from app.main import create_app
from app.models import Event, Job, JobStatus, JobType, Photo, Status, Variant, VariantSource

PRO = presets.Price(input=2.0, output_text=12.0, output_image=120.0)
USAGE = Usage(prompt=1520, image=1120, text=0, thoughts=250)
USAGE_USD = (1520 * 2 + 1120 * 120 + 250 * 12) / 1_000_000  # 0.14044


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.agent_token = SecretStr(TOKEN)
    settings.gemini_retry_base_s = 0.0
    return settings


def png_bytes(width: int, height: int, seed: int = 5) -> bytes:
    buf = io.BytesIO()
    gradient(width, height, seed=seed).save(buf, format="PNG")
    return buf.getvalue()


class Scripted:
    """Gemini понарошку: ответы по очереди (исключение — бросить), последний повторяется."""

    def __init__(self, *replies: Reply | Exception, delay: float = 0.0) -> None:
        self.replies = list(replies)
        self.delay = delay
        self.requests: list[Request] = []
        self.active = 0
        self.max_active = 0
        self.gate: asyncio.Event | None = None

    async def generate(self, request: Request) -> Reply:
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(self.delay)
            reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
            if isinstance(reply, Exception):
                raise reply
            return reply
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        return None


def ok(width: int = 2048, height: int = 1152, usage: Usage = USAGE) -> Reply:
    return Reply(image=png_bytes(width, height), finish_reason="STOP", usage=usage)


def use(app: FastAPI, backend: Any) -> Any:
    app.state.gemini.backend = backend
    app.state.gemini.poll_s = 0.05
    return backend


def wait_idle(factory: sessionmaker[Session], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with factory() as s:
            busy = s.scalar(
                select(func.count()).where(
                    Job.type == JobType.EDIT_GEMINI,
                    Job.status.in_((JobStatus.QUEUED, JobStatus.CLAIMED)),
                )
            )
        if not busy:
            return
        time.sleep(0.02)
    raise AssertionError("воркер Gemini не закончил вовремя")


def wait_for(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("не дождались")
        time.sleep(0.02)


def send(client: TestClient, pid: int, variant_id: int | None, **form: Any) -> Any:
    data = {"source": "variant" if variant_id else "prepared", "reference": "true", **form}
    if variant_id:
        data["variant_id"] = str(variant_id)
    return client.post(f"/review/{pid}/gemini", data=data, follow_redirects=False)


def gemini_jobs(factory: sessionmaker[Session]) -> list[Job]:
    with factory() as s:
        return list(s.scalars(select(Job).where(Job.type == JobType.EDIT_GEMINI).order_by(Job.id)))


# --- чистые функции ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((240, 160), "3:2"),
        ((1680, 944), "16:9"),
        ((1000, 1000), "1:1"),
        ((800, 1000), "4:5"),
        ((1000, 1800), "9:16"),
        ((3000, 1000), "21:9"),  # панорамы шире 21:9 — ближайшая, остальное обрежется
    ],
)
def test_closest_aspect(size: tuple[int, int], expected: str) -> None:
    assert gemini.closest_aspect(*size) == expected


def test_crop_to_aspect_centres_and_marks_big_cuts() -> None:
    img = gradient(2048, 2048)
    out, note = gemini.crop_to_aspect(img, 1680, 944)
    assert out.size == (2048, 1151)
    assert note == {"from": [2048, 2048], "to": [2048, 1151], "cut": 0.438}
    # обрезка по центру: верхняя строка результата — строка 448 исходника
    assert out.getpixel((10, 0)) == img.getpixel((10, 448))

    out, note = gemini.crop_to_aspect(gradient(2048, 1152), 1680, 944)
    assert out.size == (2048, 1151) and note is None  # 1 px — без отметки
    same, note = gemini.crop_to_aspect(gradient(300, 200), 600, 400)
    assert same.size == (300, 200) and note is None


def test_cost_from_usage() -> None:
    assert gemini.cost_usd(USAGE, PRO) == pytest.approx(USAGE_USD)
    assert gemini.cost_usd(Usage(), PRO) == 0


def test_build_prompt() -> None:
    two = gemini.build_prompt("BASE", "Remove the stain.\nKeep grain.", with_reference=True)
    assert two.startswith("Image 1 is the original scan. Image 2 is a restoration of it.\n")
    assert "Edit image 2: Remove the stain. Keep grain.\n" in two
    assert two.endswith("do not colorize. Return one image.")
    assert "Edit image 2: BASE\n" in gemini.build_prompt("BASE", "  ", with_reference=True)
    assert gemini.build_prompt("BASE", "Fix it.", with_reference=False) == "BASE\nFix it."
    assert gemini.build_prompt("BASE", "", with_reference=False) == "BASE"


def response(parts: list[dict[str, Any]], usage: dict[str, Any] | None = None, **cand: Any) -> Any:
    return types.GenerateContentResponse.model_validate(
        {
            "candidates": [{"content": {"role": "model", "parts": parts}, **cand}],
            "usage_metadata": usage,
        }
    )


def test_parse_response_takes_final_image_not_thought() -> None:
    thought = {"inline_data": {"mime_type": "image/png", "data": b"draft"}, "thought": True}
    final = {"inline_data": {"mime_type": "image/png", "data": b"final"}}
    usage = {
        "prompt_token_count": 1520,
        "candidates_token_count": 1130,
        "candidates_tokens_details": [
            {"modality": "IMAGE", "token_count": 1120},
            {"modality": "TEXT", "token_count": 10},
        ],
        "thoughts_token_count": 250,
    }
    reply = gemini.parse_response(
        response([thought, {"text": "Here you go."}, final], usage, finish_reason="STOP")
    )
    assert reply.image == b"final"
    assert reply.text == "Here you go."
    assert reply.finish_reason == "STOP"
    assert reply.usage == Usage(prompt=1520, image=1120, text=10, thoughts=250)


def test_parse_response_refusal_and_missing_details() -> None:
    reply = gemini.parse_response(
        response(
            [{"text": "I can't edit this photo."}],
            {"prompt_token_count": 1500, "candidates_token_count": 12},
            finish_reason="IMAGE_SAFETY",
        )
    )
    assert reply.image is None
    assert reply.usage == Usage(prompt=1500, image=0, text=12)  # без картинки — это текст
    assert gemini.refusal_text(reply) == (
        "нет картинки в ответе — фильтр безопасности (IMAGE_SAFETY): I can't edit this photo."
    )
    with_image = gemini.parse_response(
        response(
            [{"inline_data": {"mime_type": "image/png", "data": b"x"}}],
            {"prompt_token_count": 1, "candidates_token_count": 1300},
        )
    )
    assert with_image.usage.image == 1300  # разбивки нет — весь выход считается картинкой

    blocked = types.GenerateContentResponse.model_validate(
        {"prompt_feedback": {"block_reason": "PROHIBITED_CONTENT"}}
    )
    assert gemini.refusal_text(gemini.parse_response(blocked)) == (
        "запрос заблокирован (PROHIBITED_CONTENT)"
    )


def test_api_error_retryable() -> None:
    assert GeminiAPIError("x", 429).retryable
    assert GeminiAPIError("x", 503).retryable
    assert GeminiAPIError("сеть").retryable
    assert not GeminiAPIError("x", 400).retryable
    assert not GeminiAPIError("x", 403).retryable


# --- бюджет -----------------------------------------------------------------------------


def add_cost(
    factory: sessionmaker[Session],
    pid: int,
    cost: float,
    when: datetime | None = None,
    model: str = "gemini-3.1-flash-image",  # не та, что в настройках: оценку не трогает
    size: str = "2K",
) -> None:
    with factory() as s:
        s.add(
            Job(
                photo_id=pid,
                type=JobType.EDIT_GEMINI,
                status=JobStatus.DONE,
                cost_usd=cost,
                payload={"params": {"model": model, "image_size": size}},
                created_at=when or datetime.now(UTC),
            )
        )
        s.commit()


def test_month_spent_counts_this_month_in_display_tz(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    pid = queue_photo(client, intake, inbox, "a.png")
    tz = ZoneInfo("Asia/Tokyo")  # +9: 30.11 20:00 UTC — уже декабрь
    now = datetime(2026, 11, 30, 20, 0, tzinfo=UTC)
    add_cost(factory, pid, 0.5, datetime(2026, 11, 30, 14, 0, tzinfo=UTC))  # 23:00 30.11 Токио
    add_cost(factory, pid, 0.25, datetime(2026, 11, 30, 16, 0, tzinfo=UTC))  # 01:00 01.12
    with factory() as s:
        assert gemini.month_spent(s, tz, now) == 0.25
        assert gemini.month_spent(s, ZoneInfo("UTC"), now) == 0.75
        assert gemini.budget(s, tz, now).month == "декабрь"


def test_estimate_by_prices_then_by_history(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    pid = queue_photo(client, intake, inbox, "a.png")
    with factory() as s:
        prefs = presets.gemini_prefs(s)
        est = gemini.estimate(s, prefs, presets.gemini_prices(s))
    # 2 картинки и текст на входе, 1120 токенов картинки, 500 — мышление
    assert est == gemini.Estimate(round((1520 * 2 + 1120 * 120 + 500 * 12) / 1e6, 4), 0)
    assert est.basis == "по ценам из настроек"

    pro = "gemini-3-pro-image"
    add_cost(factory, pid, 0.10, model=pro)
    add_cost(factory, pid, 0.20, model=pro)
    add_cost(factory, pid, 9.99, model=pro, size="4K")  # другой размер — не в счёт
    add_cost(factory, pid, 5.00)  # другая модель — тоже
    with factory() as s:
        est = gemini.estimate(s, prefs, presets.gemini_prices(s))
        assert est == gemini.Estimate(0.15, 2)
        assert est.basis == "среднее по 2 прошлым запросам"
        assert gemini.Estimate(0.1, 21).basis == "среднее по 21 прошлому запросу"
        assert gemini.estimate(s, prefs.model_copy(update={"model": "x-image"}), {}) is None


def test_budget_check() -> None:
    est = gemini.Estimate(0.14, 3)
    b = gemini.Budget("ноябрь", spent=19.9, limit=20.0, inflight=0, estimate=est, model="m")
    assert "превысит бюджет Gemini за ноябрь" in (b.check(1) or "")
    assert gemini.Budget("ноябрь", 5.0, 20.0, 0, est, "m").check(10) is None
    busy = gemini.Budget("ноябрь", 5.0, 5.5, 3, est, "m")  # 3 × 0,14 в работе + 0,14 > 0,5
    assert "в работе ≈$0.42" in (busy.check(1) or "")
    spent = gemini.Budget("ноябрь", 20.0, 20.0, 0, est, "m")
    assert spent.exhausted
    assert spent.check(1) == (
        "Бюджет Gemini за ноябрь исчерпан: $20.00 из $20.00. Лимит можно поднять в настройках."
    )
    assert gemini.Budget("ноябрь", 0, 20, 0, None, "x-image").check(1) == (
        "Нет цен для модели x-image — задайте их в настройках."
    )


# --- воркер и экран проверки -----------------------------------------------------------


def test_gemini_variant_appears_in_feed_with_price(
    app: FastAPI,
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    """Критерий M4: вариант из Gemini появляется в ленте с ценой."""
    backend = use(app, Scripted(ok(2048, 2048)))
    pid = in_review(client, intake, inbox, "a.png")
    src = variants(factory, pid)[1]

    r = send(client, pid, src.id, text="Remove the white stain.", addons=[])
    assert r.status_code == 303
    assert "отправлено в Gemini (задача #" in msg(r)
    assert photo(factory, pid).status in (Status.GEMINI, Status.REVIEW)
    wait_idle(factory)

    p = photo(factory, pid)
    assert p.status == Status.REVIEW and not p.attention
    v = variants(factory, pid)[-1]
    assert v.source == VariantSource.GEMINI and v.parent_id == src.id and v.round == p.round
    assert v.cost_usd == pytest.approx(USAGE_USD)
    assert (v.width, v.height) == (2048, 1151)  # квадрат 1:1 → пропорции 1680×944
    assert v.params["cropped"]["cut"] == 0.438
    assert v.params["aspect_ratio"] == "16:9" and v.params["reference"] is True
    assert v.params["tokens"] == {"prompt": 1520, "image": 1120, "text": 0, "thoughts": 250}
    assert v.params["model"] == "gemini-3-pro-image" and v.params["image_size"] == "2K"
    assert "Edit image 2: Remove the white stain." in (v.prompt or "")
    path = settings.store_dir / f"R-{pid:04d}" / v.path
    assert path.is_file() and not path.stat().st_mode & 0o222

    req = backend.requests[0]
    assert len(req.images) == 2  # образец (prepared), затем источник
    with Image.open(io.BytesIO(req.images[0])) as ref, Image.open(io.BytesIO(req.images[1])) as s:
        assert ref.size == (240, 160) and s.size == (1680, 944) and s.format == "JPEG"
    assert (req.model, req.image_size, req.aspect_ratio) == ("gemini-3-pro-image", "2K", "16:9")

    job = last_job(factory, pid)
    assert job.status == JobStatus.DONE and job.cost_usd == pytest.approx(USAGE_USD)
    page = client.get(f"/review/{pid}?v={v.id}").text
    assert "Gemini · $0.14</span>" in page
    assert "$0.140" in page  # подсказка плитки
    card = client.get(f"/photo/{pid}").text
    assert "Gemini · $0.14" in card
    assert "В Gemini → Проверка · задача #" in card and "· $0.14" in card
    assert "Gemini $0.14 / $20.00" in client.get("/partials/header").text


def test_refusal_is_shown_as_text(
    app: FastAPI,
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
) -> None:
    """Критерий M4: отказ модели показан текстом — фото снова на проверке с отметкой."""
    refusal = Reply(
        image=None,
        text="I can't help with editing images of real people.",
        finish_reason="IMAGE_SAFETY",
        usage=Usage(prompt=1520, text=15, thoughts=100),
    )
    backend = use(app, Scripted(refusal))
    pid = in_review(client, intake, inbox, "a.png")
    send(client, pid, None)
    wait_idle(factory)

    assert len(backend.requests) == 1  # отказ не повторяется
    assert len(backend.requests[0].images) == 1  # источник prepared — образец не нужен
    p = photo(factory, pid)
    assert p.status == Status.REVIEW and p.attention
    expected = (
        "нет картинки в ответе — фильтр безопасности (IMAGE_SAFETY): "
        "I can't help with editing images of real people."
    )
    assert p.attention_reason == f"Gemini, задача #{last_job(factory, pid).id}: {expected}"
    job = last_job(factory, pid)
    assert job.status == JobStatus.FAILED and job.error == expected
    assert job.cost_usd == pytest.approx((1520 * 2 + 115 * 12) / 1e6)  # отказ тоже платный
    assert len(variants(factory, pid)) == 3
    page = client.get(f"/review/{pid}").text
    assert "I can&#39;t help with editing images of real people." in page
    assert "фильтр безопасности (IMAGE_SAFETY)" in page

    # следующая отправка снимает отметку
    use(app, Scripted(ok()))
    send(client, pid, None)
    assert not photo(factory, pid).attention
    wait_idle(factory)
    assert variants(factory, pid)[-1].source == VariantSource.GEMINI


def test_rate_limit_and_server_errors_are_retried(
    app: FastAPI,
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
) -> None:
    backend = use(
        app,
        Scripted(GeminiAPIError("Gemini API 429: quota", 429), GeminiAPIError("x", 503), ok()),
    )
    pid = in_review(client, intake, inbox, "a.png")
    send(client, pid, None)
    wait_idle(factory)
    assert len(backend.requests) == 3
    job = last_job(factory, pid)
    assert job.status == JobStatus.DONE and job.attempts == 3


def test_errors_after_three_retries_flag_the_photo(
    app: FastAPI,
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
) -> None:
    backend = use(app, Scripted(GeminiAPIError("Gemini API 500: internal", 500)))
    pid = in_review(client, intake, inbox, "a.png")
    send(client, pid, None)
    wait_idle(factory)
    assert len(backend.requests) == 4  # 1 + 3 повтора
    p = photo(factory, pid)
    assert p.status == Status.REVIEW and p.attention
    assert p.attention_reason and p.attention_reason.endswith("Gemini API 500: internal")
    job = last_job(factory, pid)
    assert job.status == JobStatus.FAILED and job.attempts == 4 and job.cost_usd is None


def test_client_errors_are_not_retried(
    app: FastAPI,
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
) -> None:
    backend = use(app, Scripted(GeminiAPIError("Gemini API 404: model not found", 404)))
    pid = in_review(client, intake, inbox, "a.png")
    send(client, pid, None)
    wait_idle(factory)
    assert len(backend.requests) == 1
    assert "404: model not found" in (photo(factory, pid).attention_reason or "")


def test_budget_exceeded_blocks_sending(
    app: FastAPI,
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
) -> None:
    """Критерий M4: превышение бюджета блокирует отправку."""
    use(app, Scripted(ok()))
    pid = in_review(client, intake, inbox, "a.png")
    other = queue_photo(client, intake, inbox, "b.png")
    add_cost(factory, other, 19.95)

    page = client.get(f"/review/{pid}").text
    assert "превысит бюджет Gemini за" in page
    assert "disabled>Отправить в Gemini · ≈$0.14</button>" in page  # кнопка выключена
    r = send(client, pid, None)
    assert "превысит бюджет Gemini" in msg(r) and "Лимит можно поднять в настройках" in msg(r)
    assert photo(factory, pid).status == Status.REVIEW
    assert gemini_jobs(factory)[-1].photo_id == other  # новой задачи нет

    add_cost(factory, other, 0.05)  # ровно 20 из 20
    r = send(client, pid, None)
    assert "Бюджет Gemini за" in msg(r) and "исчерпан: $20.00 из $20.00" in msg(r)
    assert "chip bad" in client.get("/partials/header").text

    # лимит подняли в настройках — снова можно
    client.post(
        "/settings/gemini",
        data={
            "model": "gemini-3-pro-image",
            "image_size": "2K",
            "monthly_budget_usd": "30",
            "max_concurrency": "2",
        },
    )
    r = send(client, pid, None)
    assert "отправлено в Gemini" in msg(r)
    wait_idle(factory)


def test_missing_prices_block_sending(
    app: FastAPI, client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    use(app, Scripted(ok()))
    pid = in_review(client, intake, inbox, "a.png")
    with factory() as s:
        presets.set_setting(
            s, "gemini", {**presets.gemini_prefs(s).model_dump(), "model": "x-image"}
        )
        s.commit()
    r = send(client, pid, None)
    assert msg(r) == f"/review/{pid}?msg=Нет цен для модели x-image — задайте их в настройках."
    assert photo(factory, pid).status == Status.REVIEW


def test_without_api_key_gemini_is_unavailable(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    pid = in_review(client, intake, inbox, "a.png")
    page = client.get(f"/review/{pid}").text
    assert "Gemini не настроен: нет GEMINI_API_KEY." in page
    assert "Gemini $" not in client.get("/partials/header").text
    r = send(client, pid, None)
    assert "Gemini не настроен" in msg(r)
    assert photo(factory, pid).status == Status.REVIEW


def test_bulk_send_one_job_per_photo_at_most_two_at_once(
    app: FastAPI,
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
) -> None:
    backend = use(app, Scripted(ok(), delay=0.1))
    ids = [in_review(client, intake, inbox, f"{n}.png") for n in "abcd"]
    prep = queue_photo(client, intake, inbox, "e.png")
    with factory() as s:  # у последнего фото последний вариант скрыт — берётся предпоследний
        v = s.get(Variant, variants(factory, ids[-1])[-1].id)
        assert v is not None
        v.hidden = True
        s.commit()

    r = client.post(
        "/bulk",
        data={
            "action": "gemini",
            "ids": [str(i) for i in [*ids, prep]],
            "text": "Fix the sky.",
            "source": "latest",
            "reference": "true",
        },
        follow_redirects=False,
    )
    assert msg(r).endswith(f"В Gemini: 4 (≈$0.57); не на проверке: R-{prep:04d}")
    wait_idle(factory)

    assert backend.max_active == 2
    jobs_ = gemini_jobs(factory)
    assert [j.photo_id for j in jobs_] == ids
    assert all(j.status == JobStatus.DONE for j in jobs_)
    assert jobs_[-1].payload["source_variant_id"] == variants(factory, ids[-1])[1].id
    assert all("Edit image 2: Fix the sky." in j.payload["prompt"] for j in jobs_)
    assert {photo(factory, i).status for i in ids} == {Status.REVIEW}


def test_bulk_send_refused_when_budget_is_short(
    app: FastAPI,
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
) -> None:
    use(app, Scripted(ok()))
    ids = [in_review(client, intake, inbox, f"{n}.png") for n in "abc"]
    add_cost(factory, ids[0], 19.70)  # остаётся 0,30 — на 3 × 0,14 не хватит
    r = client.post(
        "/bulk", data={"action": "gemini", "ids": [str(i) for i in ids]}, follow_redirects=False
    )
    assert "Отправка ≈$0.43 превысит бюджет" in msg(r)
    assert len(gemini_jobs(factory)) == 1
    assert {photo(factory, i).status for i in ids} == {Status.REVIEW}


def test_cancel_while_request_runs(
    app: FastAPI,
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
) -> None:
    backend = use(app, Scripted(ok()))
    loop = app.state.gemini._loop
    backend.gate = asyncio.Event()
    pid = in_review(client, intake, inbox, "a.png")
    send(client, pid, None)
    wait_for(lambda: last_job(factory, pid).status == JobStatus.CLAIMED)
    job_id = last_job(factory, pid).id

    client.post(f"/jobs/{job_id}/cancel")
    p = photo(factory, pid)
    assert p.status == Status.REVIEW and p.attention
    assert p.attention_reason == f"Задача #{job_id} отменена"

    loop.call_soon_threadsafe(backend.gate.set)
    wait_for(lambda: last_job(factory, pid).cost_usd is not None)
    assert last_job(factory, pid).status == JobStatus.CANCELLED
    assert len(variants(factory, pid)) == 3  # ответ выброшен, цена учтена
    assert last_job(factory, pid).cost_usd == pytest.approx(USAGE_USD)


def test_jobs_taken_before_restart_are_requeued(
    app: FastAPI, client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    pid = in_review(client, intake, inbox, "a.png")
    with factory() as s:
        p = s.get(Photo, pid)
        assert p is not None
        p.status = Status.GEMINI
        s.add(Job(photo_id=pid, type=JobType.EDIT_GEMINI, status=JobStatus.CLAIMED, payload={}))
        s.commit()
    [job_id] = app.state.gemini.recover()
    with factory() as s:
        job = s.get(Job, job_id)
        assert job is not None and job.status == JobStatus.QUEUED
        assert s.get(Photo, pid).status == Status.GEMINI  # type: ignore[union-attr]


def test_fake_gemini_marks_input_and_costs_nothing(settings: Settings) -> None:
    settings.fake_gemini = True
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.gemini.poll_s = 0.05
        intake = app.state.intake
        factory = app.state.factory
        pid = in_review(client, intake, settings.inbox_dir, "a.png")
        assert "FAKE_GEMINI" in client.get("/settings").text
        src = variants(factory, pid)[0]
        send(client, pid, src.id)
        wait_idle(factory)
        v = variants(factory, pid)[-1]
        assert v.source == VariantSource.GEMINI and v.cost_usd == 0
        assert (v.width, v.height) == (src.width, src.height)

        send(client, pid, src.id, text="FAKE_REFUSE please")
        wait_idle(factory)
        reason = photo(factory, pid).attention_reason or ""
        assert "фильтр безопасности (IMAGE_SAFETY): I can't help with editing this image." in reason


def test_events_describe_gemini(
    app: FastAPI, client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    use(app, Scripted(ok()))
    pid = in_review(client, intake, inbox, "a.png")
    src = variants(factory, pid)[2]
    send(client, pid, src.id)
    wait_idle(factory)
    with factory() as s:
        texts = [
            ev.data
            for ev in s.scalars(select(Event).where(Event.photo_id == pid, Event.kind == "status"))
        ]
    sent = texts[-2]
    assert sent["gemini"] == {
        "source": src.label,
        "reference": True,
        "model": "gemini-3-pro-image",
        "size": "2K",
    }
    card = client.get(f"/photo/{pid}").text
    assert f"Проверка → В Gemini · из {src.label} с образцом (gemini-3-pro-image, 2K)" in card


# --- google-genai против подменённого API ------------------------------------------------

API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-3-pro-image:generateContent"


def request(**kw: Any) -> Request:
    return Request(
        **{
            "model": "gemini-3-pro-image",
            "prompt": "Edit image 2: fix.",
            "images": [b"\xff\xd8ref", b"\xff\xd8src"],
            "aspect_ratio": "3:2",
            "image_size": "2K",
            **kw,
        }
    )


@respx.mock
async def test_sdk_request_and_response() -> None:
    image = png_bytes(30, 20)
    route = respx.post(API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "draft", "thought": True},
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": base64.b64encode(image).decode(),
                                    }
                                },
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1520,
                    "candidatesTokenCount": 1120,
                    "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1120}],
                    "thoughtsTokenCount": 250,
                    "totalTokenCount": 2890,
                },
            },
        )
    )
    backend = gemini.GenaiBackend("test-key", timeout_s=30)
    try:
        reply = await backend.generate(request())
    finally:
        await backend.aclose()

    assert reply.image == image and reply.finish_reason == "STOP" and reply.text == ""
    assert reply.usage == Usage(prompt=1520, image=1120, text=0, thoughts=250)
    sent = route.calls.last.request
    assert sent.headers["x-goog-api-key"] == "test-key"
    body = json.loads(sent.content)
    parts = body["contents"][0]["parts"]
    sent_images = [base64.urlsafe_b64decode(p["inlineData"]["data"]) for p in parts[:2]]
    assert sent_images == [b"\xff\xd8ref", b"\xff\xd8src"]  # образец, затем источник
    assert parts[0]["inlineData"]["mime_type"] == "image/jpeg"
    assert parts[2] == {"text": "Edit image 2: fix."}
    config = body["generationConfig"]
    assert config["responseModalities"] == ["TEXT", "IMAGE"]
    assert config["imageConfig"] == {"aspectRatio": "3:2", "imageSize": "2K"}


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (400, False)])
@respx.mock
async def test_sdk_errors(status: int, retryable: bool) -> None:
    respx.post(API_URL).mock(
        return_value=httpx.Response(
            status, json={"error": {"code": status, "message": "boom", "status": "X"}}
        )
    )
    backend = gemini.GenaiBackend("test-key", timeout_s=30)
    try:
        with pytest.raises(GeminiAPIError) as err:
            await backend.generate(request())
    finally:
        await backend.aclose()
    assert err.value.status == status and err.value.retryable is retryable
    assert str(err.value) == f"Gemini API {status}: boom"


@respx.mock
async def test_sdk_network_error_is_retryable() -> None:
    respx.post(API_URL).mock(side_effect=httpx.ConnectError("нет сети"))
    backend = gemini.GenaiBackend("test-key", timeout_s=30)
    try:
        with pytest.raises(GeminiAPIError) as err:
            await backend.generate(request())
    finally:
        await backend.aclose()
    assert err.value.status is None and err.value.retryable


def test_old_gemini_setting_gets_new_defaults(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    with factory() as s:  # так было сохранено в M2: без новых ключей
        presets.set_setting(s, "gemini", {"model": "gemini-3.1-flash-image", "image_size": "1K"})
        s.commit()
        prefs = presets.gemini_prefs(s)
    assert prefs.model == "gemini-3.1-flash-image" and prefs.image_size == "1K"
    assert prefs.monthly_budget_usd == 20.0 and prefs.max_concurrency == 2
