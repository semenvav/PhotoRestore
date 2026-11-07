"""Исполнитель Gemini (ТЗ §8): отправка по кнопке, бюджет, фоновый воркер.

Сервер вызывает Gemini сам, агент и `run` не участвуют. «Отправить в Gemini» переводит фото
`REVIEW → GEMINI` и ставит задачу `edit_gemini`; воркер (asyncio, не больше
`max_concurrency` запросов сразу) берёт её, отправляет вход и возвращает фото в `REVIEW`:
с новым вариантом или с `attention` и текстом отказа. 429, 5xx и обрыв связи повторяются
до 3 раз с паузой 2, 4, 8 с; отказ модели (нет картинки) не повторяется.

Стоимость считается по `usage_metadata` × цены из настроек и пишется в `job.cost_usd`
(и в `variant.cost_usd`); бюджет месяца — сумма `job.cost_usd` задач, созданных в этом
месяце (`DISPLAY_TZ`), включая отказы и отменённые.
"""

import asyncio
import contextlib
import io
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol
from zoneinfo import ZoneInfo

import httpx
from PIL import Image, ImageDraw
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app import imaging, jobs, presets, results, states
from app.config import Settings
from app.db import session_scope, utcnow
from app.models import Actor, Job, JobStatus, JobType, Photo, Status, Variant, VariantSource
from app.presets import GeminiPrefs, Price
from app.storage import Store
from app.views import usd

if TYPE_CHECKING:
    from google.genai import types as genai_types

log = logging.getLogger(__name__)

# Пропорции, которые принимают все image-модели Gemini (у Flash есть ещё 1:4, 4:1, 1:8, 8:1)
ASPECT_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")
CROP_NOTE = 0.02  # обрезано больше 2% по стороне — отметка в params
MAX_RETRIES = 3
INPUT_MAX_SIDE = 3072  # вход уменьшается: цена картинки на входе фиксирована (560 токенов)
INPUT_JPEG_QUALITY = 95

# Оценка до первых запросов (дальше — по истории): токены входа и выхода
INPUT_IMAGE_TOKENS = 560
PROMPT_TOKENS_GUESS = 400
THOUGHT_TOKENS_GUESS = 500
OUTPUT_IMAGE_TOKENS: dict[str, dict[str, int]] = {
    "gemini-3-pro-image": {"1K": 1120, "2K": 1120, "4K": 2000},
    "gemini-3.1-flash-image": {"1K": 1120, "2K": 1680, "4K": 2520},
    "gemini-nano-banana-2.1": {"1K": 1120, "2K": 1680, "4K": 3780},
}
DEFAULT_OUTPUT_TOKENS = {"1K": 1120, "2K": 1680, "4K": 3780}
HISTORY_SAMPLE = 20

TWO_IMAGES_TEMPLATE = (
    "Image 1 is the original scan. Image 2 is a restoration of it.\n"
    "Edit image 2: {user_prompt}\n"
    "Keep faces, facial features, expressions, clothing, composition and framing identical "
    "to image 1.\n"
    "Keep the original tonality, do not colorize. Return one image."
)

MONTHS = (
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
)

FINISH_HINTS = {
    "SAFETY": "фильтр безопасности",
    "IMAGE_SAFETY": "фильтр безопасности",
    "PROHIBITED_CONTENT": "запрещённое содержимое",
    "IMAGE_PROHIBITED_CONTENT": "запрещённое содержимое",
    "BLOCKLIST": "запрещённые слова",
    "SPII": "персональные данные",
    "RECITATION": "похоже на защищённый материал",
    "IMAGE_RECITATION": "похоже на защищённый материал",
    "NO_IMAGE": "модель не создала картинку",
    "STOP": "модель ответила только текстом",
    "MAX_TOKENS": "ответ оборвался на лимите токенов",
    "OTHER": "прочее",
    "IMAGE_OTHER": "прочее",
}


class GeminiError(Exception):
    pass


# --- промпт, пропорции, стоимость ------------------------------------------------------


def build_prompt(base: str, user_prompt: str, with_reference: bool) -> str:
    """С образцом — шаблон ТЗ §8 (образец, затем источник); с одной картинкой — базовый
    пресет + текст пользователя."""
    if with_reference:
        text = " ".join((user_prompt.strip() or base.strip()).split("\n"))
        return TWO_IMAGES_TEMPLATE.format(user_prompt=text)
    return presets.compose_prompt(base, user_prompt)


def _ratio(name: str) -> float:
    w, h = name.split(":")
    return int(w) / int(h)


def closest_aspect(width: int, height: int) -> str:
    target = width / height
    return min(ASPECT_RATIOS, key=lambda r: abs(math.log(_ratio(r) / target)))


def crop_to_aspect(img: Image.Image, width: int, height: int) -> tuple[Image.Image, dict | None]:
    """Центрированная обрезка под пропорции входа `width × height`. Вторым — отметка,
    если по какой-то стороне ушло больше `CROP_NOTE`."""
    w, h = img.size
    target = width / height
    if w / h > target:
        cw, ch = max(1, round(h * target)), h
    else:
        cw, ch = w, max(1, round(w / target))
    if (cw, ch) == (w, h):
        return img, None
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    cut = max(1 - cw / w, 1 - ch / h)
    note = {"from": [w, h], "to": [cw, ch], "cut": round(cut, 4)} if cut > CROP_NOTE else None
    return cropped, note


@dataclass(frozen=True)
class Usage:
    """Токены ответа: вход (текст и картинки), выход картинками и текстом, мышление."""

    prompt: int = 0
    image: int = 0
    text: int = 0
    thoughts: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt": self.prompt,
            "image": self.image,
            "text": self.text,
            "thoughts": self.thoughts,
        }


def cost_usd(usage: Usage, price: Price) -> float:
    total = (
        usage.prompt * price.input
        + usage.image * price.output_image
        + (usage.text + usage.thoughts) * price.output_text
    )
    return round(total / 1_000_000, 6)


@dataclass(frozen=True)
class Estimate:
    usd: float
    samples: int  # 0 — по ценам из настроек, иначе среднее по стольким прошлым запросам

    @property
    def basis(self) -> str:
        if self.samples:
            n = self.samples
            one = n % 10 == 1 and n % 100 != 11
            return f"среднее по {n} {'прошлому запросу' if one else 'прошлым запросам'}"
        return "по ценам из настроек"


def estimate(session: Session, prefs: GeminiPrefs, prices: dict[str, Price]) -> Estimate | None:
    """Оценка одного запроса: среднее по прошлым с той же моделью и размером, иначе по
    ценам. Нет цен для модели — None (бюджет не посчитать, отправка блокируется)."""
    recent = session.scalars(
        select(Job)
        .where(
            Job.type == JobType.EDIT_GEMINI,
            Job.status == JobStatus.DONE,
            Job.cost_usd.is_not(None),
        )
        .order_by(Job.id.desc())
        .limit(HISTORY_SAMPLE * 10)
    ).all()
    same = [
        j.cost_usd
        for j in recent
        if j.cost_usd is not None
        and (j.payload.get("params") or {}).get("model") == prefs.model
        and (j.payload.get("params") or {}).get("image_size") == prefs.image_size
    ][:HISTORY_SAMPLE]
    if same:
        return Estimate(round(sum(same) / len(same), 4), len(same))
    price = prices.get(prefs.model)
    if price is None:
        return None
    out = OUTPUT_IMAGE_TOKENS.get(prefs.model, DEFAULT_OUTPUT_TOKENS)[prefs.image_size]
    guess = Usage(
        prompt=2 * INPUT_IMAGE_TOKENS + PROMPT_TOKENS_GUESS,
        image=out,
        thoughts=THOUGHT_TOKENS_GUESS,
    )
    return Estimate(round(cost_usd(guess, price), 4), 0)


def month_start(tz: ZoneInfo, now: datetime | None = None) -> datetime:
    local = (now or utcnow()).astimezone(tz)
    start = local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(UTC)


def month_spent(session: Session, tz: ZoneInfo, now: datetime | None = None) -> float:
    total = session.scalar(
        select(func.coalesce(func.sum(Job.cost_usd), 0.0)).where(
            Job.type == JobType.EDIT_GEMINI, Job.created_at >= month_start(tz, now)
        )
    )
    return float(total or 0.0)


@dataclass
class Budget:
    month: str
    spent: float
    limit: float
    inflight: int  # задач Gemini в очереди и в работе (их цена ещё не известна)
    estimate: Estimate | None
    model: str

    @property
    def reserved(self) -> float:
        return self.inflight * self.estimate.usd if self.estimate else 0.0

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit

    def check(self, count: int = 1) -> str | None:
        """Причина, по которой нельзя отправить `count` запросов; None — можно."""
        if self.estimate is None:
            return f"Нет цен для модели {self.model} — задайте их в настройках."
        tail = "Лимит можно поднять в настройках."
        if self.exhausted:
            return (
                f"Бюджет Gemini за {self.month} исчерпан: {usd(self.spent)} из "
                f"{usd(self.limit)}. {tail}"
            )
        need = count * self.estimate.usd
        if self.spent + self.reserved + need > self.limit:
            busy = f", в работе ≈{usd(self.reserved)}" if self.inflight else ""
            return (
                f"Отправка ≈{usd(need)} превысит бюджет Gemini за {self.month}: потрачено "
                f"{usd(self.spent)}{busy} из {usd(self.limit)}. {tail}"
            )
        return None


def budget(session: Session, tz: ZoneInfo, now: datetime | None = None) -> Budget:
    prefs = presets.gemini_prefs(session)
    inflight = session.scalar(
        select(func.count()).where(
            Job.type == JobType.EDIT_GEMINI,
            Job.status.in_((JobStatus.QUEUED, JobStatus.CLAIMED)),
        )
    )
    local = (now or utcnow()).astimezone(tz)
    return Budget(
        month=MONTHS[local.month - 1],
        spent=month_spent(session, tz, now),
        limit=prefs.monthly_budget_usd,
        inflight=inflight or 0,
        estimate=estimate(session, prefs, presets.gemini_prices(session)),
        model=prefs.model,
    )


# --- отправка -----------------------------------------------------------------------------


def send(
    session: Session,
    photo: Photo,
    user_prompt: str,
    source_variant: Variant | None,
    reference: bool,
    actor: Actor = Actor.OWNER,
) -> Job:
    """«Отправить в Gemini»: `REVIEW → GEMINI` и задача `edit_gemini`. Бюджет проверяет
    вызывающий (`Budget.check`) — для массовой отправки сразу на все фото."""
    prefs = presets.gemini_prefs(session)
    with_reference = reference and source_variant is not None
    source = source_variant.label if source_variant else "prepared"
    states.transition(
        session,
        photo,
        Status.GEMINI,
        actor,
        {
            "gemini": {
                "source": source,
                "reference": with_reference,
                "model": prefs.model,
                "size": prefs.image_size,
            }
        },
    )
    jobs.clear_attention(photo)
    job = Job(
        photo_id=photo.id,
        type=JobType.EDIT_GEMINI,
        status=JobStatus.QUEUED,
        max_attempts=MAX_RETRIES + 1,
        payload={
            "prompt": build_prompt(presets.base_prompt(session), user_prompt, with_reference),
            "user_prompt": user_prompt,
            "source": "variant" if source_variant else "prepared",
            "source_variant_id": source_variant.id if source_variant else None,
            "reference": with_reference,
            "round": photo.round,
            "params": {"model": prefs.model, "image_size": prefs.image_size},
        },
    )
    session.add(job)
    session.flush()
    states.log_event(
        session, photo.id, actor, "job_created", {"job_id": job.id, "type": job.type.value}
    )
    return job


# --- обращение к модели -----------------------------------------------------------------


@dataclass
class Request:
    model: str
    prompt: str
    images: list[bytes]  # JPEG; порядок: образец, затем источник
    aspect_ratio: str
    image_size: str


@dataclass
class Reply:
    image: bytes | None
    text: str = ""
    finish_reason: str | None = None
    finish_message: str | None = None
    block_reason: str | None = None
    usage: Usage = field(default_factory=Usage)


class GeminiAPIError(Exception):
    """Ошибка вызова: `status` — HTTP-код, None — нет связи/таймаут."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status == 429 or self.status >= 500


class Backend(Protocol):
    async def generate(self, request: Request) -> Reply: ...

    async def aclose(self) -> None: ...


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(value.value if isinstance(value, Enum) else value)


def parse_response(response: "genai_types.GenerateContentResponse") -> Reply:
    """Первая inline-картинка (не из мышления), текст ответа, причина остановки, токены."""
    candidate = response.candidates[0] if response.candidates else None
    parts = (
        candidate.content.parts
        if candidate is not None and candidate.content and candidate.content.parts
        else []
    )
    image = next(
        (
            p.inline_data.data
            for p in parts
            if not p.thought
            and p.inline_data is not None
            and p.inline_data.data
            and (p.inline_data.mime_type or "image/").startswith("image/")
        ),
        None,
    )
    text = " ".join(p.text.strip() for p in parts if p.text and not p.thought).strip()

    usage = Usage()
    meta = response.usage_metadata
    if meta is not None:
        out = meta.candidates_token_count or 0
        details = meta.candidates_tokens_details or []
        if details:
            image_tokens = sum(
                d.token_count or 0 for d in details if _enum_name(d.modality) == "IMAGE"
            )
        else:  # без разбивки весь выход с картинкой считается картинкой — дороже
            image_tokens = out if image else 0
        usage = Usage(
            prompt=meta.prompt_token_count or 0,
            image=image_tokens,
            text=max(0, out - image_tokens),
            thoughts=meta.thoughts_token_count or 0,
        )
    feedback = response.prompt_feedback
    return Reply(
        image=image,
        text=text,
        finish_reason=_enum_name(candidate.finish_reason) if candidate else None,
        finish_message=candidate.finish_message if candidate else None,
        block_reason=_enum_name(feedback.block_reason) if feedback else None,
        usage=usage,
    )


class GenaiBackend:
    """Gemini Developer API через google-genai (`generateContent`)."""

    def __init__(
        self, api_key: str, timeout_s: float, http: httpx.AsyncClient | None = None
    ) -> None:
        from google import genai
        from google.genai import types

        # свой httpx-клиент: SDK по умолчанию берёт httpx2; таймаут на весь запрос
        self._http = http or httpx.AsyncClient(timeout=timeout_s)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000), httpx_async_client=self._http
            ),
        )

    async def generate(self, request: Request) -> Reply:
        from google.genai import errors, types

        contents: list[Any] = [
            types.Part.from_bytes(data=data, mime_type="image/jpeg") for data in request.images
        ]
        contents.append(request.prompt)
        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=request.aspect_ratio, image_size=request.image_size
            ),
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=request.model, contents=contents, config=config
            )
        except errors.APIError as e:
            detail = e.message or e.status or str(e)
            raise GeminiAPIError(f"Gemini API {e.code}: {detail}", e.code) from e
        except (httpx.HTTPError, OSError) as e:
            raise GeminiAPIError(f"нет связи с Gemini API: {e!r}") from e
        return parse_response(response)

    async def aclose(self) -> None:
        await self._http.aclose()


class FakeBackend:
    """`FAKE_GEMINI=1`: вход с пометкой, цена 0. Промпт с `FAKE_REFUSE` — отказ модели."""

    def __init__(self) -> None:
        self.requests: list[Request] = []

    async def generate(self, request: Request) -> Reply:
        self.requests.append(request)
        if "FAKE_REFUSE" in request.prompt:
            return Reply(
                image=None,
                text="I can't help with editing this image.",
                finish_reason="IMAGE_SAFETY",
            )
        with Image.open(io.BytesIO(request.images[-1])) as src:
            img = src.convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, min(img.width, 90), min(img.height, 16)), fill=(255, 255, 255))
        draw.text((3, 2), "FAKE GEMINI", fill=(200, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Reply(image=buf.getvalue(), finish_reason="STOP")

    async def aclose(self) -> None:
        return None


def make_backend(settings: Settings) -> Backend | None:
    if settings.fake_gemini:
        return FakeBackend()
    if settings.gemini_api_key is not None and settings.gemini_api_key.get_secret_value():
        return GenaiBackend(settings.gemini_api_key.get_secret_value(), settings.gemini_timeout_s)
    return None


def refusal_text(reply: Reply) -> str:
    if reply.block_reason:
        head = f"запрос заблокирован ({reply.block_reason})"
    else:
        code = reply.finish_reason or "нет причины"
        hint = FINISH_HINTS.get(code, "картинки нет")
        head = f"нет картинки в ответе — {hint} ({code})"
    extra = [t.strip() for t in (reply.finish_message, reply.text) if t and t.strip()]
    text = head + (": " + " · ".join(extra) if extra else "")
    return text if len(text) <= 600 else text[:599] + "…"


# --- воркер ---------------------------------------------------------------------------------


def encode_input(img: Image.Image) -> bytes:
    small = img.copy()
    small.thumbnail((INPUT_MAX_SIDE, INPUT_MAX_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=INPUT_JPEG_QUALITY)
    return buf.getvalue()


@dataclass
class Prepared:
    request: Request
    width: int  # размер источника: под его пропорции обрезается ответ
    height: int


class GeminiWorker:
    """Фоновый исполнитель задач `edit_gemini` (одна копия на процесс сервера)."""

    def __init__(
        self,
        factory: sessionmaker[Session],
        store: Store,
        settings: Settings,
        backend: Backend | None,
    ) -> None:
        self.factory = factory
        self.store = store
        self.backend = backend
        self.retry_base_s = settings.gemini_retry_base_s
        self.poll_s = 5.0
        self.tasks: set[asyncio.Task[None]] = set()
        self._event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def available(self) -> bool:
        return self.backend is not None

    @property
    def fake(self) -> bool:
        return isinstance(self.backend, FakeBackend)

    def wake(self) -> None:
        """Новая задача — разбудить воркер (можно звать из потока обработчика запроса)."""
        if self._loop is not None and self._event is not None:
            self._loop.call_soon_threadsafe(self._event.set)

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._event = asyncio.Event()
        await asyncio.to_thread(self.recover)
        while True:
            try:
                await self.fill()
            except Exception:
                log.exception("gemini: выдача задач упала")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._event.wait(), timeout=self.poll_s)
            self._event.clear()

    async def stop(self) -> None:
        for task in list(self.tasks):
            task.cancel()
        for task in list(self.tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self.backend is not None:
            await self.backend.aclose()

    def recover(self) -> list[int]:
        """Задачи, взятые до перезапуска сервера, — снова в очередь."""
        with session_scope(self.factory) as s:
            stuck = s.scalars(
                select(Job).where(Job.type == JobType.EDIT_GEMINI, Job.status == JobStatus.CLAIMED)
            ).all()
            for job in stuck:
                job.status = JobStatus.QUEUED
            if stuck:
                log.warning(
                    "gemini: задачи %s снова в очереди после перезапуска", [j.id for j in stuck]
                )
            return [j.id for j in stuck]

    def limit(self) -> int:
        with session_scope(self.factory) as s:
            return presets.gemini_prefs(s).max_concurrency

    async def fill(self) -> None:
        if self.backend is None:
            return
        limit = await asyncio.to_thread(self.limit)
        while len(self.tasks) < limit:
            job_id = await asyncio.to_thread(self.claim_next)
            if job_id is None:
                return
            task = asyncio.create_task(self.process(job_id))
            self.tasks.add(task)
            task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[None]) -> None:
        self.tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            log.error("gemini: задача упала", exc_info=exc)
        if self._event is not None:
            self._event.set()

    def claim_next(self) -> int | None:
        with session_scope(self.factory) as s:
            job = s.scalar(
                select(Job)
                .where(Job.type == JobType.EDIT_GEMINI, Job.status == JobStatus.QUEUED)
                .order_by(Job.priority.desc(), Job.id)
                .limit(1)
            )
            if job is None:
                return None
            job.status = JobStatus.CLAIMED
            job.started_at = utcnow()
            job.progress = {"done": 0, "total": 1, "stage": "gemini"}
            return job.id

    def is_active(self, job_id: int) -> bool:
        with session_scope(self.factory) as s:
            job = s.get(Job, job_id)
            return job is not None and job.status == JobStatus.CLAIMED

    def prepare(self, job_id: int) -> Prepared | None:
        with session_scope(self.factory) as s:
            job = s.get(Job, job_id)
            if job is None or job.status != JobStatus.CLAIMED:
                return None
            source = results.input_path(s, self.store, job, "source")
            reference = self.store.prepared(job.photo_id) if job.payload.get("reference") else None
            params = job.payload.get("params") or {}
            prompt = str(job.payload["prompt"])
        img = imaging.load_oriented(source)
        images = [encode_input(imaging.load_oriented(reference))] if reference else []
        images.append(encode_input(img))
        request = Request(
            model=str(params.get("model")),
            prompt=prompt,
            images=images,
            aspect_ratio=closest_aspect(img.width, img.height),
            image_size=str(params.get("image_size", "2K")),
        )
        return Prepared(request, img.width, img.height)

    async def process(self, job_id: int) -> None:
        assert self.backend is not None
        started = time.monotonic()
        try:
            prepared = await asyncio.to_thread(self.prepare, job_id)
        except Exception as e:
            log.exception("gemini: вход задачи #%s не прочитан", job_id)
            await asyncio.to_thread(
                self.finish, job_id, None, None, f"вход не прочитан: {e}", 0, started
            )
            return
        if prepared is None:
            return
        reply: Reply | None = None
        error: str | None = None
        attempts = 0
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                await asyncio.sleep(self.retry_base_s * 2 ** (attempt - 1))
                if not await asyncio.to_thread(self.is_active, job_id):
                    return  # отменили, пока ждали повтора
            attempts += 1
            try:
                reply = await self.backend.generate(prepared.request)
                error = None
                break
            except GeminiAPIError as e:
                error = str(e)
                log.warning("gemini: задача #%s, попытка %s: %s", job_id, attempts, e)
                if not e.retryable:
                    break
            except Exception as e:  # неожиданный ответ SDK — не повторяем
                log.exception("gemini: задача #%s", job_id)
                error = f"ошибка Gemini: {e!r}"
                break
        await asyncio.to_thread(self.finish, job_id, prepared, reply, error, attempts, started)

    def finish(
        self,
        job_id: int,
        prepared: Prepared | None,
        reply: Reply | None,
        error: str | None,
        attempts: int,
        started: float,
    ) -> None:
        with session_scope(self.factory) as s:
            job = s.get(Job, job_id)
            if job is None:
                return
            photo = s.get(Photo, job.photo_id)
            assert photo is not None
            params = job.payload.get("params") or {}
            usage = reply.usage if reply else Usage()
            price = presets.gemini_prices(s).get(str(params.get("model")))
            if reply is not None:
                job.cost_usd = cost_usd(usage, price) if price else None
            job.attempts = attempts
            if job.status != JobStatus.CLAIMED:
                return  # отменили, пока шёл запрос: результат не нужен, цена учтена
            variant = None
            if reply is not None and reply.image and prepared is not None:
                try:
                    variant = self._save(s, job, photo, prepared, reply, started)
                except (OSError, ValueError, Image.DecompressionBombError) as e:
                    error = f"Gemini вернул нечитаемую картинку: {e}"
            elif reply is not None and error is None:
                error = refusal_text(reply)
            self._close(s, job, photo, variant, error or "неизвестная ошибка")

    def _save(
        self, s: Session, job: Job, photo: Photo, prepared: Prepared, reply: Reply, started: float
    ) -> Variant:
        assert reply.image is not None
        with Image.open(io.BytesIO(reply.image)) as raw:
            raw.load()
            img = imaging.to_rgb8(raw)
        img, cropped = crop_to_aspect(img, prepared.width, prepared.height)
        params: dict[str, Any] = {
            **(job.payload.get("params") or {}),
            "aspect_ratio": prepared.request.aspect_ratio,
            "reference": bool(job.payload.get("reference")),
            "tokens": reply.usage.as_dict(),
        }
        if cropped:
            params["cropped"] = cropped
        return results.add_variant(
            s,
            self.store,
            photo,
            lambda tmp: imaging.save_png(img, tmp),
            image=img,
            parent_id=job.payload.get("source_variant_id"),
            source=VariantSource.GEMINI,
            job_id=job.id,
            round=int(job.payload.get("round", photo.round)),
            prompt=job.payload.get("prompt"),
            params=params,
            cost_usd=job.cost_usd,
            duration_s=round(time.monotonic() - started, 1),
        )

    def _close(
        self, s: Session, job: Job, photo: Photo, variant: Variant | None, error: str
    ) -> None:
        now = utcnow()
        job.finished_at = now
        if variant is not None:
            s.flush()
            job.status = JobStatus.DONE
            job.error = None
            job.progress = {"done": 1, "total": 1, "stage": "done"}
            data: dict[str, Any] = {"job_id": job.id, "variants": [variant.label]}
            if job.cost_usd is not None:
                data["cost"] = job.cost_usd
        else:
            job.status = JobStatus.FAILED
            job.error = error
            job.progress = None
            data = {"job_id": job.id, "error": error}
        if photo.status == Status.GEMINI:
            states.transition(s, photo, Status.REVIEW, Actor.SYSTEM, data)
        if variant is None:
            photo.attention = True
            photo.attention_reason = f"Gemini, задача #{job.id}: {error}"
        else:
            photo.attention = False
            photo.attention_reason = None
