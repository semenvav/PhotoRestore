"""Клиент API сервера (ТЗ §6). Агент — всегда инициатор.

Вызовы по задаче (входы, прогресс, выходы, complete, fail) при потере связи повторяются с
экспоненциальной паузой до успеха: готовые результаты ждут в workdir, пока не загрузятся.
"""

import json
import logging
import threading
from typing import Any

import httpx

log = logging.getLogger(__name__)

API = "/api/agent/v1"
MAX_BACKOFF_S = 60.0
UNAUTHORIZED = "сервер не принял токен (401) — проверьте token в config.toml"


class ServerError(Exception):
    pass


class Outdated(ServerError):
    """426: сервер не принимает эту версию агента."""


class JobGone(ServerError):
    """Задача больше не у агента: отменена, возвращена в очередь по аренде или удалена."""


class Stopped(Exception):
    """Агент останавливается — повторы прекращены."""


class ServerClient:
    def __init__(
        self, http: httpx.Client, stop: threading.Event | None = None, backoff_s: float = 1.0
    ) -> None:
        self.http = http
        self.stop = stop or threading.Event()
        self.backoff_s = backoff_s

    def _send(self, method: str, path: str, *, persistent: bool, **kw: Any) -> httpx.Response:
        """`persistent`: повторять до успеха — и при 401: посреди задачи сменили токен, а
        исправленный `config.toml` агент подхватит на лету (`self.http` подменяется)."""
        delay = self.backoff_s
        while True:
            try:
                r = self.http.request(
                    method, path if path.startswith("/") else f"{API}/{path}", **kw
                )
                if r.status_code < 500 and not (persistent and r.status_code == 401):
                    return r
                problem = UNAUTHORIZED if r.status_code == 401 else f"HTTP {r.status_code}"
            except httpx.TransportError as e:
                problem = f"{type(e).__name__}: {e}"
            if not persistent:
                raise ServerError(f"{method} {path}: {problem}")
            log.warning("%s %s: %s — повтор через %.0f с", method, path, problem, delay)
            if self.stop.wait(delay):
                raise Stopped
            delay = min(delay * 2, MAX_BACKOFF_S)

    @staticmethod
    def _check(r: httpx.Response, what: str) -> httpx.Response:
        if r.status_code == 426:
            raise Outdated(_error_text(r))
        if r.status_code in (404, 409):
            raise JobGone(f"{what}: {_error_text(r)}")
        if r.status_code == 401:
            raise ServerError(f"{what}: {UNAUTHORIZED}")
        if r.status_code >= 400:
            raise ServerError(f"{what}: HTTP {r.status_code} {_error_text(r)}")
        return r

    # --- пульс и выдача ----------------------------------------------------------------

    def hello(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._send("POST", "hello", persistent=False, json=payload)
        result: dict[str, Any] = self._check(r, "hello").json()
        return result

    def claim(self, agent_id: str, prefer_type: str | None) -> dict[str, Any] | None:
        r = self._send(
            "POST",
            "claim",
            persistent=False,
            json={"agent_id": agent_id, "prefer_type": prefer_type},
        )
        if r.status_code == 204:
            return None
        job: dict[str, Any] = self._check(r, "claim").json()
        return job

    # --- задача ------------------------------------------------------------------------

    def download(self, url: str) -> bytes:
        r = self._send("GET", url, persistent=True)
        return self._check(r, f"вход {url}").content

    def progress(self, job_id: int, done: int, total: int, stage: str) -> bool:
        """True — задачу надо прервать (отменена или аренда истекла)."""
        r = self._send(
            "POST",
            f"jobs/{job_id}/progress",
            persistent=True,
            json={"done": done, "total": total, "stage": stage},
        )
        return bool(self._check(r, "progress").json().get("cancel"))

    def upload_output(self, job_id: int, index: int, data: bytes, meta: dict[str, Any]) -> None:
        r = self._send(
            "POST",
            f"jobs/{job_id}/outputs/{index}",
            persistent=True,
            files={"file": (f"out_{index}.png", data, "image/png")},
            data={"meta": json.dumps(meta)},
        )
        self._check(r, f"выход {index}")

    def complete(self, job_id: int) -> None:
        self._check(
            self._send("POST", f"jobs/{job_id}/complete", persistent=True, json={}), "complete"
        )

    def fail(self, job_id: int, error: str, retryable: bool) -> None:
        r = self._send(
            "POST",
            f"jobs/{job_id}/fail",
            persistent=True,
            json={"error": error[:4000], "retryable": retryable},
        )
        self._check(r, "fail")


def _error_text(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return r.text[:500]
    if isinstance(body, dict):
        return str(body.get("error") or body.get("detail") or body)
    return str(body)
