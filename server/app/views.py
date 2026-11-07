"""Шаблоны и человекочитаемые тексты для UI."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi.templating import Jinja2Templates

from app.db import utcnow
from app.models import Actor, Event, JobStatus, JobType, Status, photo_code
from app.states import LABELS

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

ACTOR_LABELS = {
    Actor.OWNER: "владелец",
    Actor.FAMILY: "отец",
    Actor.AGENT: "агент",
    Actor.SYSTEM: "система",
}
JOB_TYPE_LABELS = {
    JobType.EDIT_LOCAL: "чистка на ПК",
    JobType.UPSCALE_LOCAL: "апскейл на ПК",
    JobType.EDIT_GEMINI: "Gemini",
}
JOB_STATUS_LABELS = {
    JobStatus.QUEUED: "в очереди",
    JobStatus.CLAIMED: "выполняется",
    JobStatus.DONE: "готово",
    JobStatus.FAILED: "ошибка",
    JobStatus.CANCELLED: "отменена",
}
AGENT_OFFLINE_AFTER = timedelta(seconds=60)


def status_label(value: str | None) -> str:
    return LABELS[Status(value)] if value else "—"


def describe_event(ev: Event) -> str:
    d: dict[str, Any] = ev.data or {}
    match ev.kind:
        case "status" if d.get("from") is None:
            text = f"Принято из «Входящих»: {d.get('inbox_path', '')}"
            if d.get("duplicate_of"):
                text += f" — дубль {photo_code(d['duplicate_of'])}"
            return text
        case "status":
            text = f"{status_label(d.get('from'))} → {status_label(d.get('to'))}"
            if prepared := d.get("prepared"):
                text += f" · подготовлено {prepared['width']}×{prepared['height']}"
            if d.get("job_id"):
                text += f" · задача #{d['job_id']}"
            if d.get("variants"):
                text += f" · варианты {', '.join(d['variants'])}"
            if d.get("source") and d.get("to") == "QUEUED_EDIT":
                text += f" · доработка из {d['source']}"
            if d.get("cancelled"):
                text += " · отменена"
            if d.get("error"):
                attempt = f" (попытка {d['attempt']})" if d.get("attempt") else ""
                text += f" · ошибка{attempt}: {d['error']}"
            return text
        case "similar":
            items = ", ".join(
                f"{photo_code(p['photo_id'])} (расстояние {p['distance']})" for p in d["photos"]
            )
            return f"Похоже на {items}"
        case "job_created":
            kind = JOB_TYPE_LABELS.get(JobType(d["type"]), d["type"])
            text = f"Создана задача #{d['job_id']}: {kind}"
            if d.get("retry_of"):
                text += f" (повтор #{d['retry_of']})"
            return text
        case "variant_hide":
            return f"Скрыт вариант {d['variant']}"
        case "variant_show":
            return f"Возвращён вариант {d['variant']}"
        case "run_started":
            scope = "всё" if d.get("scope") == "all" else "выбранные"
            return f"Запуск #{d['run_id']} ({scope}), задач: {d.get('jobs', '?')}"
        case "run_stopped":
            return f"Запуск #{d['run_id']} остановлен"
        case "run_completed":
            done, failed = d.get("done", 0), d.get("failed", 0)
            return f"Запуск #{d['run_id']} завершён: готово {done}, ошибок {failed}"
        case "jobs_cancelled":
            return "Отменены задачи: " + ", ".join(f"#{i}" for i in d["job_ids"])
        case "reprepared":
            text = f"Переподготовлено: {d['width']}×{d['height']}"
            if d.get("archived"):
                text += f", прежний файл → {d['archived']}"
            return text
        case "unsupported":
            return f"Файл «{d['path']}» не принят ({d['reason']}) → {d['moved_to']}"
        case "uploaded":
            return f"Загружено через интерфейс: {d['path']}"
        case "deleted":
            return f"Удалено {d['code']} «{d['original_name']}», файлы — в store/.deleted"
        case "promoted":
            return f"Стало основным: дубль удалённого {photo_code(d['deleted'])}"
        case _:
            return f"{ev.kind}: {d}"


def make_templates(display_tz: str, app_version: str = "dev") -> Jinja2Templates:
    tz = ZoneInfo(display_tz)
    templates = Jinja2Templates(directory=TEMPLATES_DIR)

    def localtime(value: datetime | None, fmt: str = "%d.%m.%Y %H:%M") -> str:
        return value.astimezone(tz).strftime(fmt) if value else "—"

    def ago(value: datetime | None) -> str:
        if value is None:
            return "никогда"
        seconds = int((utcnow() - value).total_seconds())
        if seconds < 60:
            return f"{seconds} с назад"
        if seconds < 3600:
            return f"{seconds // 60} мин назад"
        return localtime(value)

    env = templates.env
    env.filters["localtime"] = localtime
    env.filters["ago"] = ago
    env.filters["status_label"] = status_label
    env.filters["code"] = photo_code
    env.globals["describe_event"] = describe_event
    env.globals["ACTOR_LABELS"] = ACTOR_LABELS
    env.globals["JOB_TYPE_LABELS"] = JOB_TYPE_LABELS
    env.globals["JOB_STATUS_LABELS"] = JOB_STATUS_LABELS
    env.globals["APP_VERSION"] = app_version
    return templates
