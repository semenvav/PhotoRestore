"""Локальный ComfyUI через его HTTP API (ТЗ §7.2). Только 127.0.0.1, только запросы агента."""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)

UPLOAD_SUBFOLDER = "restore"
MODEL_EXT = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".sft")
QUEUE_CHECK_EVERY = 15  # опросов /history между проверками, что задача ещё в очереди ComfyUI


class ComfyError(Exception):
    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class Cancelled(Exception):
    """Сервер отменил задачу — выполнение прервано."""


@dataclass(frozen=True)
class Stats:
    gpu_name: str | None
    vram_free_mb: int | None


class Comfy(Protocol):
    """То, что агенту нужно от ComfyUI (настоящего или FAKE_COMFY)."""

    def system_stats(self) -> Stats | None: ...
    def check_models(self, workflow: dict[str, Any]) -> None: ...
    def upload_image(self, name: str, data: bytes) -> str: ...
    def run(
        self, workflow: dict[str, Any], output_node: str, on_tick: Callable[[], bool] | None = None
    ) -> bytes: ...
    def interrupt(self) -> None: ...
    def free(self) -> None: ...


def is_oom(text: str) -> bool:
    low = text.lower()
    return "out of memory" in low or "outofmemory" in low or "allocation on device" in low


def combo_values(spec: Any) -> list[Any] | None:
    """Допустимые значения поля из /object_info: `[[...], {...}]` или `["COMBO", {"options"}]`."""
    if not isinstance(spec, list) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get("options")
        return options if isinstance(options, list) else None
    return None


def missing_models(workflow: dict[str, Any], object_info: dict[str, Any]) -> list[str]:
    problems = []
    for node_id, node in workflow.items():
        cls = node.get("class_type")
        title = (node.get("_meta") or {}).get("title") or f"#{node_id}"
        info = object_info.get(cls)
        if info is None:
            problems.append(f"в ComfyUI нет узла {cls} ({title}) — не установлен custom node?")
            continue
        spec_all = {
            **info.get("input", {}).get("required", {}),
            **info.get("input", {}).get("optional", {}),
        }
        for field, value in node.get("inputs", {}).items():
            if not isinstance(value, str) or not value.lower().endswith(MODEL_EXT):
                continue
            allowed = combo_values(spec_all.get(field))
            if allowed is not None and value not in allowed:
                problems.append(f"нет файла {value} в models/… (узел {title}, поле {field})")
    return problems


class ComfyClient:
    def __init__(self, http: httpx.Client, poll_s: float = 2.0, tick_s: float = 10.0) -> None:
        self.http = http
        self.poll_s = poll_s
        self.tick_s = tick_s
        self.client_id = uuid.uuid4().hex

    def system_stats(self) -> Stats | None:
        try:
            r = self.http.get("/system_stats", timeout=5)
            r.raise_for_status()
            devices = r.json().get("devices") or [{}]
        except (httpx.HTTPError, ValueError):
            return None
        dev = devices[0]
        name = dev.get("name")
        if isinstance(name, str):  # "cuda:0 NVIDIA GeForce RTX 3090 Ti : cudaMallocAsync"
            name = name.split(" : ")[0].removeprefix("cuda:0 ").strip()
        free = dev.get("vram_free")
        return Stats(gpu_name=name, vram_free_mb=int(free) // 2**20 if free is not None else None)

    def check_models(self, workflow: dict[str, Any]) -> None:
        r = self._call("GET", "/object_info")
        problems = missing_models(workflow, r.json())
        if problems:
            raise ComfyError("; ".join(problems), retryable=False)

    def upload_image(self, name: str, data: bytes) -> str:
        r = self._call(
            "POST",
            "/upload/image",
            files={"image": (name, data, "image/png")},
            data={"subfolder": UPLOAD_SUBFOLDER, "type": "input", "overwrite": "true"},
        )
        body = r.json()
        sub = body.get("subfolder") or ""
        return f"{sub}/{body['name']}" if sub else str(body["name"])

    def run(
        self,
        workflow: dict[str, Any],
        output_node: str,
        on_tick: Callable[[], bool] | None = None,
    ) -> bytes:
        """Ставит граф в очередь, ждёт результата (опрос /history) и забирает картинку узла
        `output_node`. `on_tick` вызывается раз в `tick_s`; вернул True — прервать."""
        r = self.http.post(
            "/prompt", json={"prompt": workflow, "client_id": self.client_id}, timeout=60
        )
        if r.status_code == 400:
            raise ComfyError(f"ComfyUI отклонил граф: {_node_errors(r)}", retryable=False)
        r.raise_for_status()
        prompt_id = r.json()["prompt_id"]
        entry = self._wait(prompt_id, on_tick)
        images = (entry.get("outputs", {}).get(output_node) or {}).get("images") or []
        if not images:
            raise ComfyError(f"узел {output_node} не вернул картинку", retryable=False)
        img = images[0]
        view = self._call(
            "GET",
            "/view",
            params={
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            },
        )
        return view.content

    def _wait(self, prompt_id: str, on_tick: Callable[[], bool] | None) -> dict[str, Any]:
        last_tick = time.monotonic()
        polls = 0
        while True:
            history = self._call("GET", f"/history/{prompt_id}").json()
            entry = history.get(prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise _execution_error(status)
                if status.get("completed") or entry.get("outputs"):
                    return dict(entry)
            polls += 1
            if polls % QUEUE_CHECK_EVERY == 0 and entry is None and not self._queued(prompt_id):
                raise ComfyError("ComfyUI потерял задачу (перезапущен?)")
            if on_tick and time.monotonic() - last_tick >= self.tick_s:
                last_tick = time.monotonic()
                if on_tick():
                    self.interrupt()
                    raise Cancelled
            time.sleep(self.poll_s)

    def _queued(self, prompt_id: str) -> bool:
        q = self._call("GET", "/queue").json()
        items = (q.get("queue_running") or []) + (q.get("queue_pending") or [])
        return any(len(item) > 1 and item[1] == prompt_id for item in items)

    def interrupt(self) -> None:
        try:
            self.http.post("/interrupt", json={}, timeout=10)
        except httpx.HTTPError as e:
            log.warning("interrupt: %s", e)

    def free(self) -> None:
        self._call("POST", "/free", json={"unload_models": True, "free_memory": True})

    def _call(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            r = self.http.request(method, path, timeout=kw.pop("timeout", 120), **kw)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise ComfyError(f"ComfyUI {method} {path}: {e}") from e
        return r


def _node_errors(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return r.text[:1000]
    parts = []
    if err := body.get("error"):
        parts.append(str(err.get("message", err)) if isinstance(err, dict) else str(err))
    for node_id, info in (body.get("node_errors") or {}).items():
        cls = info.get("class_type", node_id)
        for e in info.get("errors", []):
            parts.append(f"{cls}: {e.get('message')} {e.get('details', '')}".strip())
    return "; ".join(parts) or r.text[:1000]


def _execution_error(status: dict[str, Any]) -> Exception:
    for kind, data in status.get("messages") or []:
        if kind == "execution_interrupted":
            return Cancelled()
        if kind == "execution_error":
            text = (
                f"{data.get('node_type', '?')}: {data.get('exception_type', '')} "
                f"{data.get('exception_message', '')}"
            ).strip()
            return ComfyError(text, retryable=not is_oom(text))
    return ComfyError("ComfyUI: ошибка выполнения без подробностей")
