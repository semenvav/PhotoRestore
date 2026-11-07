"""Цикл работы агента (ТЗ §7.2).

1. `hello` каждые 15 с с состоянием ComfyUI.
2. Есть активный `run` и ComfyUI доступен — `claim`. Без `run` агент GPU не трогает.
3. Задача: входы → ComfyUI, по каждому выходу — граф с именем входа и seed → результат
   в workdir → загрузка на сервер → `progress`; в конце `complete`.
4. `claim` вернул 204 или запуск остановлен — один раз `POST /free`, чтобы освободить VRAM.

Пульс заодно следит за `config.toml`: изменённый файл применяется без перезапуска.
"""

import copy
import dataclasses
import functools
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from restore_agent import __version__, wiring
from restore_agent.client import JobGone, Outdated, ServerClient, ServerError, Stopped
from restore_agent.comfy import Cancelled, Comfy, ComfyError
from restore_agent.config import Config, ConfigError, load

log = logging.getLogger(__name__)

HELLO_INTERVAL_S = 15.0
COMFY_START_TIMEOUT_S = 180.0
COMFY_RELAUNCH_PAUSE_S = 60.0  # не чаще: команда запуска может падать сразу
STALE_JOB_DIR_S = 3 * 24 * 3600  # недоделанные задачи старше — убираем при старте


class Agent:
    def __init__(
        self,
        cfg: Config,
        server: ServerClient,
        comfy: Comfy,
        stop: threading.Event | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.server = server
        self.comfy = comfy
        self.stop = stop or server.stop
        self.wake = threading.Event()
        self.run_active = False
        self.poll_interval = HELLO_INTERVAL_S
        self.outdated: str | None = None
        self.comfy_ok = False
        self.worked_since_free = False
        self.last_type: str | None = None
        self.checked_workflows: set[str] = set()
        self.comfy_proc: subprocess.Popen[bytes] | None = None
        self.comfy_launched_at: float | None = None
        self.jobs_dir = cfg.workdir / "jobs"
        # config.toml на лету: пульс замечает изменение, клиент ComfyUI меняется между задачами
        self.config_path = config_path
        self.config_stamp = _stamp(config_path)
        self.config_bad_stamp: tuple[int, int] | None = None
        self.pending_comfy: Comfy | None = None
        self.lock = threading.Lock()

    # --- пульс --------------------------------------------------------------------------

    def heartbeat(self) -> None:
        stats = self.comfy.system_stats()
        self.comfy_ok = stats is not None
        payload = {
            "agent_id": self.cfg.agent_id,
            "version": __version__,
            "name": self.cfg.name,
            "comfy_ok": self.comfy_ok,
            "gpu_name": stats.gpu_name if stats else None,
            "vram_free_mb": stats.vram_free_mb if stats else None,
        }
        was_active = self.run_active
        try:
            reply = self.server.hello(payload)
        except Outdated as e:
            if self.outdated != str(e):
                log.error("сервер не принимает агента %s: %s", __version__, e)
            self.outdated = str(e)
            self.run_active = False
            return
        except ServerError as e:
            log.warning("%s", e)  # в тексте ошибки уже есть вызов
            return
        self.outdated = None
        self.run_active = bool(reply.get("run_active"))
        self.poll_interval = float(reply.get("poll_interval_s") or HELLO_INTERVAL_S)
        if self.run_active and not was_active:
            log.info("запуск активен — беру задачи")
            self.wake.set()

    # --- один шаг основного цикла -------------------------------------------------------

    def step(self) -> float:
        """Один проход; возвращает, сколько ждать до следующего (0 — сразу)."""
        self.apply_pending_comfy()
        if self.outdated or not self.run_active:
            if self.worked_since_free:  # «Стоп» или запуск кончился, пока шла задача
                self.release_vram()
            return HELLO_INTERVAL_S
        if not self.comfy_ok and not self.ensure_comfy():
            return self.poll_interval
        try:
            job = self.server.claim(self.cfg.agent_id, self.last_type)
        except ServerError as e:
            log.warning("%s", e)
            return self.poll_interval
        if job is None:
            if self.worked_since_free:
                self.release_vram()
            return self.poll_interval
        self.worked_since_free = True
        self.last_type = job["type"]
        self.execute(job)
        return 0.0

    def release_vram(self) -> None:
        try:
            self.comfy.free()
            self.worked_since_free = False
            log.info("работы нет — модели выгружены, VRAM освобождена")
        except ComfyError as e:
            if self.comfy.system_stats() is None:  # ComfyUI не работает — и VRAM не держит
                self.comfy_ok = False
                self.worked_since_free = False
            else:
                log.warning("free: %s", e)

    def ensure_comfy(self) -> bool:
        """ComfyUI недоступен: запускаем по `comfy_launch_cmd` (если задан) и ждём до 3 мин.

        Упал или его закрыли — запускаем снова, но не чаще раза в минуту. Пока запущенный нами
        процесс жив (стартует или завис), второй не запускаем.
        """
        if self.comfy.system_stats() is not None:
            self.comfy_ok = True
            return True
        cmd = self.cfg.comfy_launch_cmd
        if not cmd:
            return False
        if self.comfy_proc is not None and self.comfy_proc.poll() is None:
            return False
        now = time.monotonic()
        if (
            self.comfy_launched_at is not None
            and now - self.comfy_launched_at < COMFY_RELAUNCH_PAUSE_S
        ):
            return False
        if self.comfy_proc is not None:
            log.warning("ComfyUI завершился (код %s) — запускаю снова", _code(self.comfy_proc))
        log.info("ComfyUI не отвечает — запускаю: %s", cmd)
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW  # без окна консоли
        proc = subprocess.Popen(cmd, shell=True, creationflags=flags)
        self.comfy_proc, self.comfy_launched_at = proc, now
        deadline = now + COMFY_START_TIMEOUT_S
        while time.monotonic() < deadline and not self.stop.is_set():
            if self.comfy.system_stats() is not None:
                self.comfy_ok = True
                log.info("ComfyUI запущен")
                return True
            if proc.poll() is not None:
                log.error(
                    "ComfyUI завершился при запуске (код %s) — см. его лог "
                    "ComfyUI\\user\\comfyui_<порт>.log",
                    _code(proc),
                )
                return False
            self.stop.wait(3)
        if not self.stop.is_set():
            log.error("ComfyUI не ответил за %.0f с", COMFY_START_TIMEOUT_S)
        return False

    # --- выполнение задачи --------------------------------------------------------------

    def execute(self, job: dict[str, Any]) -> None:
        job_id = int(job["id"])
        job_dir = self.jobs_dir / str(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "задача %s (%s, фото %s): выходов %s",
            job_id,
            job["type"],
            job["photo_id"],
            len(job["outputs"]),
        )
        try:
            self._execute(job, job_dir)
            self.server.complete(job_id)
            log.info("задача %s выполнена", job_id)
        except Cancelled:
            log.info("задача %s отменена сервером — прерываю", job_id)
        except JobGone as e:
            log.info("задача %s больше не наша: %s", job_id, e)
        except ComfyError as e:
            log.error("задача %s: %s", job_id, e)
            if self.comfy.system_stats() is None:
                self.comfy_ok = False  # ComfyUI упал: новых задач не берём, пока не поднимем
            self._fail(job_id, str(e), e.retryable)
        except Stopped:
            log.info("остановка агента посреди задачи %s — сервер вернёт её по аренде", job_id)
            return  # результаты остаются в workdir
        except ServerError as e:  # сервер отверг вызов (4xx): результаты остаются в workdir
            log.error("задача %s: %s", job_id, e)
            self._fail(job_id, f"агент: {e}", retryable=True)
            return
        except Exception as e:  # агент не должен падать из-за одной задачи
            log.exception("задача %s: непредвиденная ошибка", job_id)
            self._fail(job_id, f"агент: {type(e).__name__}: {e}", retryable=True)
        shutil.rmtree(job_dir, ignore_errors=True)

    def _fail(self, job_id: int, error: str, retryable: bool) -> None:
        try:
            self.server.fail(job_id, error, retryable)
        except JobGone:
            pass
        except ServerError as e:
            log.error("fail задачи %s не принят: %s — сервер вернёт её по аренде", job_id, e)

    def _execute(self, job: dict[str, Any], job_dir: Path) -> None:
        job_id = int(job["id"])
        workflow: dict[str, Any] = job["workflow"]
        bindings = job["bindings"]
        node = _titles(workflow)

        digest = hashlib.sha256(json.dumps(workflow, sort_keys=True).encode()).hexdigest()
        if digest not in self.checked_workflows:
            self.comfy.check_models(workflow)
            self.checked_workflows.add(digest)

        uploaded: dict[str, str] = {}
        for item in job["inputs"]:
            data = self.server.download(item["url"])
            uploaded[item["name"]] = self.comfy.upload_image(f"{item['name']}.png", data)

        outputs = job["outputs"]
        total = len(outputs)
        for item in outputs:
            index, seed = int(item["index"]), item.get("seed")
            result = job_dir / f"out_{index}.png"
            meta_path = job_dir / f"out_{index}.json"
            stage = f"вариант {index + 1} из {total}"
            if not _reusable(meta_path, seed):
                if self.server.progress(job_id, index, total, stage):
                    raise Cancelled
                graph = copy.deepcopy(workflow)
                image = bindings["image"]
                graph[node(image["node"])]["inputs"][image["field"]] = uploaded["source"]
                if (seed_at := bindings.get("seed")) and seed is not None:
                    graph[node(seed_at["node"])]["inputs"][seed_at["field"]] = seed
                started = time.monotonic()
                data = self.comfy.run(
                    graph,
                    node(bindings["output"]["node"]),
                    on_tick=functools.partial(self.server.progress, job_id, index, total, stage),
                )
                meta = {"seed": seed, "duration_s": round(time.monotonic() - started, 1)}
                _write_atomic(result, data)
                _write_atomic(meta_path, json.dumps(meta).encode())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.server.upload_output(job_id, index, result.read_bytes(), meta)
            if self.server.progress(job_id, index + 1, total, f"загружен {index + 1} из {total}"):
                raise Cancelled

    # --- config.toml на лету ------------------------------------------------------------

    def check_config(self) -> None:
        """Файл изменился — перечитать и применить; с ошибкой — работать со старым."""
        stamp = _stamp(self.config_path)
        if self.config_path is None or stamp is None:
            return
        if stamp in (self.config_stamp, self.config_bad_stamp):
            return
        try:
            cfg = load(self.config_path)
        except ConfigError as e:
            self.config_bad_stamp = stamp  # до следующего изменения файла
            log.error("config.toml не применён, работаю со старым: %s", e)
            return
        self.config_stamp = stamp
        self.apply_config(cfg)

    def apply_config(self, cfg: Config) -> None:
        old = self.cfg
        if cfg.workdir != old.workdir:
            log.warning(
                "workdir меняется только перезапуском агента, пока остаётся %s", old.workdir
            )
            cfg = dataclasses.replace(cfg, workdir=old.workdir)
        changed = [
            f.name
            for f in dataclasses.fields(Config)
            if getattr(cfg, f.name) != getattr(old, f.name)
        ]
        if not changed:
            return
        log.info("config.toml изменён (%s) — применяю", ", ".join(changed))
        if cfg.server_url != old.server_url or cfg.token != old.token:
            # вызовы, которые сейчас повторяются, со следующей попытки пойдут через новый клиент
            self.server.http = wiring.server_http(cfg)
        if cfg.comfy_url != old.comfy_url or cfg.fake_comfy != old.fake_comfy:
            with self.lock:
                self.pending_comfy = wiring.comfy_client(cfg)  # между задачами, см. step()
        if cfg.comfy_launch_cmd != old.comfy_launch_cmd:
            self.comfy_launched_at = None
        self.cfg = cfg
        self.wake.set()

    def apply_pending_comfy(self) -> None:
        with self.lock:
            comfy, self.pending_comfy = self.pending_comfy, None
        if comfy is None:
            return
        if self.worked_since_free:
            self.release_vram()  # прежний ComfyUI держит модели
        self.comfy = comfy
        self.comfy_ok = comfy.system_stats() is not None
        self.comfy_proc = None
        self.comfy_launched_at = None
        self.worked_since_free = False
        self.checked_workflows.clear()
        log.info("ComfyUI теперь %s", "FAKE" if self.cfg.fake_comfy else self.cfg.comfy_url)

    # --- жизнь процесса -----------------------------------------------------------------

    def cleanup_stale(self) -> None:
        """Каталоги задач, брошенных при прошлых запусках (только наши, только в workdir)."""
        if not self.jobs_dir.is_dir():
            return
        cutoff = time.time() - STALE_JOB_DIR_S
        for path in self.jobs_dir.iterdir():
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)

    def run_forever(self) -> None:
        self.cleanup_stale()
        log.info(
            "агент %s (%s) запущен, сервер %s, ComfyUI %s",
            self.cfg.agent_id,
            __version__,
            self.cfg.server_url,
            "FAKE" if self.cfg.fake_comfy else self.cfg.comfy_url,
        )
        beat = threading.Thread(target=self._heartbeat_loop, name="hello", daemon=True)
        beat.start()
        while not self.stop.is_set():
            try:
                delay = self.step()
            except Stopped:
                break
            except Exception:  # сбой одного прохода не должен останавливать агента
                log.exception("сбой в цикле агента — продолжаю")
                delay = self.poll_interval
            if delay > 0:
                self.wake.wait(delay)
                self.wake.clear()
        log.info("агент остановлен")

    def _heartbeat_loop(self) -> None:
        while not self.stop.is_set():
            try:
                self.check_config()
            except Exception:
                log.exception("проверка config.toml упала")
            try:
                self.heartbeat()
            except Exception:
                log.exception("hello упал")
            self.stop.wait(HELLO_INTERVAL_S)

    def shutdown(self) -> None:
        self.stop.set()
        self.wake.set()


def _titles(workflow: dict[str, Any]) -> Any:
    by_title = {(n.get("_meta") or {}).get("title"): node_id for node_id, n in workflow.items()}

    def node(title: str) -> str:
        if title not in by_title:
            raise ComfyError(f"в графе нет узла с заголовком {title}", retryable=False)
        return by_title[title]

    return node


def _code(proc: "subprocess.Popen[bytes]") -> int | None:
    """Код выхода; убитый процесс в Windows даёт 0xFFFFFFFF — показываем как -1."""
    code = proc.returncode
    return code - 2**32 if code is not None and code >= 2**31 else code


def _stamp(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _reusable(meta_path: Path, seed: Any) -> bool:
    """Результат этого выхода уже есть в workdir (агент перезапускался посреди задачи)."""
    if not meta_path.is_file() or not meta_path.with_suffix(".png").is_file():
        return False
    try:
        return bool(json.loads(meta_path.read_text(encoding="utf-8")).get("seed") == seed)
    except ValueError:
        return False


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
