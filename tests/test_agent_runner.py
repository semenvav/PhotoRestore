"""Агент без сервера: подъём ComfyUI, config.toml на лету, 401, живучесть цикла."""

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from restore_agent import runner, wiring
from restore_agent.client import ServerClient, ServerError
from restore_agent.comfy import ComfyClient, ComfyError, Stats
from restore_agent.config import Config, load
from restore_agent.fake import FakeComfy
from restore_agent.runner import Agent

S = "http://server.test"


class Proc:
    """Процесс ComfyUI для подменённого subprocess.Popen."""

    def __init__(self) -> None:
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


class Comfy(FakeComfy):
    """ComfyUI, который можно «уронить»."""

    def __init__(self) -> None:
        super().__init__()
        self.up = False

    def system_stats(self) -> Stats | None:
        return Stats(gpu_name="GPU", vram_free_mb=1) if self.up else None

    def free(self) -> None:
        if not self.up:
            raise ComfyError("ComfyUI недоступен")
        super().free()


class Ticker(threading.Event):
    """Вместо паузы в ожидании ComfyUI — действие теста («ComfyUI поднялся», «упал»)."""

    def __init__(self, on_wait: Callable[[], object]) -> None:
        super().__init__()
        self.on_wait = on_wait

    def wait(self, timeout: float | None = None) -> bool:
        self.on_wait()
        return self.is_set()


def make_agent(
    tmp_path: Path, comfy: Any = None, stop: threading.Event | None = None, **cfg: Any
) -> Agent:
    config = Config(server_url=S, token="t", agent_id="pc", workdir=tmp_path / "w", **cfg)
    server = ServerClient(httpx.Client(base_url=S), backoff_s=0.001)
    return Agent(config, server, comfy or Comfy(), stop)


@pytest.fixture
def popen(monkeypatch: pytest.MonkeyPatch) -> list[Proc]:
    launched: list[Proc] = []

    def fake_popen(cmd: str, **kw: Any) -> Proc:
        launched.append(Proc())
        return launched[-1]

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner, "COMFY_START_TIMEOUT_S", 2.0)
    return launched


def test_comfy_is_launched_again_after_crash(
    tmp_path: Path, popen: list[Proc], monkeypatch: pytest.MonkeyPatch
) -> None:
    comfy = Comfy()
    stop = Ticker(lambda: setattr(comfy, "up", True))  # пока ждём — ComfyUI поднялся
    ag = make_agent(tmp_path, comfy, stop, comfy_launch_cmd="comfy.exe")
    assert ag.ensure_comfy() and len(popen) == 1

    # ComfyUI упал: снова запускаем, но не чаще раза в минуту
    comfy.up, popen[0].returncode = False, 1
    assert not ag.ensure_comfy() and len(popen) == 1
    monkeypatch.setattr(runner, "COMFY_RELAUNCH_PAUSE_S", 0.0)
    assert ag.ensure_comfy() and len(popen) == 2

    # наш процесс жив, но не отвечает (стартует или завис) — второй не запускаем
    comfy.up = False
    assert not ag.ensure_comfy() and len(popen) == 2


def test_comfy_that_exits_at_start_is_not_awaited(
    tmp_path: Path, popen: list[Proc], monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    stop = Ticker(lambda: setattr(popen[-1], "returncode", 1))  # пока ждём — процесс вышел
    ag = make_agent(tmp_path, None, stop, comfy_launch_cmd="broken.exe")
    with caplog.at_level(logging.ERROR):
        assert not ag.ensure_comfy()
    assert "завершился при запуске (код 1)" in caplog.text


def test_without_launch_cmd_comfy_is_not_started(tmp_path: Path, popen: list[Proc]) -> None:
    assert not make_agent(tmp_path).ensure_comfy() and popen == []


def test_stopped_run_frees_vram_once(tmp_path: Path) -> None:
    comfy = Comfy()
    comfy.up = True
    ag = make_agent(tmp_path, comfy)
    ag.worked_since_free = True  # задача кончилась, а запуск уже остановлен
    ag.step()
    ag.step()
    assert comfy.frees == 1


def test_nothing_to_free_when_comfy_is_down(tmp_path: Path, caplog: Any) -> None:
    ag = make_agent(tmp_path)
    ag.worked_since_free = True
    with caplog.at_level(logging.WARNING):
        ag.step()
        ag.step()
    assert not ag.worked_since_free and "free" not in caplog.text


def test_loop_survives_unexpected_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    ag = make_agent(tmp_path)
    calls: list[int] = []

    def step() -> float:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("диск переполнен")
        ag.shutdown()
        return 0.0

    monkeypatch.setattr(ag, "step", step)
    monkeypatch.setattr(ag, "_heartbeat_loop", lambda: None)
    ag.poll_interval = 0.01
    with caplog.at_level(logging.ERROR):
        ag.run_forever()
    assert len(calls) == 2 and "диск переполнен" in caplog.text


# --- 401 и config.toml на лету ------------------------------------------------------------


@respx.mock
def test_job_call_waits_out_401_until_token_is_fixed() -> None:
    server = ServerClient(
        httpx.Client(base_url=S, headers={"Authorization": "Bearer old"}), backoff_s=0.001
    )
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        if request.headers["Authorization"] == "Bearer old":
            # так поступает пульс, заметив исправленный config.toml
            server.http = httpx.Client(base_url=S, headers={"Authorization": "Bearer new"})
            return httpx.Response(401)
        return httpx.Response(200, json={"cancel": False})

    respx.post(f"{S}/api/agent/v1/jobs/1/progress").mock(side_effect=handler)
    assert server.progress(1, 0, 3, "вариант 1 из 3") is False
    assert seen == ["Bearer old", "Bearer new"]


@respx.mock
def test_hello_with_wrong_token_says_so() -> None:
    respx.post(f"{S}/api/agent/v1/hello").respond(401, json={"detail": "bad token"})
    server = ServerClient(httpx.Client(base_url=S))
    with pytest.raises(ServerError, match=r"проверьте token в config.toml"):
        server.hello({})


def write_config(path: Path, **values: Any) -> None:
    lines = [
        f"{k} = {v!r}".replace("True", "true").replace("False", "false") for k, v in values.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    st = path.stat()  # время изменения — заведомо новое, даже если запись была в тот же тик
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9 * (1 + len(values))))


@respx.mock
def test_config_is_reloaded_on_the_fly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    monkeypatch.delenv("FAKE_COMFY", raising=False)
    path = tmp_path / "config.toml"
    base = {"agent_id": "pc", "workdir": str(tmp_path / "w"), "fake_comfy": True}
    write_config(path, server_url="http://old.test", token="t1", **base)
    cfg = load(path)
    ag = Agent(
        cfg, ServerClient(wiring.server_http(cfg)), wiring.comfy_client(cfg), config_path=path
    )
    first_http = ag.server.http
    ag.check_config()
    assert ag.server.http is first_http  # файл не менялся

    # новый адрес и токен — новый клиент сервера сразу
    write_config(path, server_url="http://new.test", token="t2", name="Рабочий ПК", **base)
    with caplog.at_level(logging.INFO):
        ag.check_config()
    assert ag.cfg.server_url == "http://new.test" and ag.cfg.name == "Рабочий ПК"
    assert ag.server.http is not first_http
    assert str(ag.server.http.base_url) == "http://new.test"
    assert ag.server.http.headers["Authorization"] == "Bearer t2"
    assert "t2" not in caplog.text  # токен в лог не попадает

    # файл с ошибкой: работаем со старым, об ошибке — один раз
    path.write_text("server_url = \n", encoding="utf-8")
    os.utime(path, ns=(0, path.stat().st_mtime_ns + 10**10))
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        ag.check_config()
        ag.check_config()
    assert ag.cfg.server_url == "http://new.test"
    assert caplog.text.count("config.toml не применён") == 1

    # другой ComfyUI — подменяется между задачами; workdir — только перезапуском
    respx.get("http://127.0.0.1:9999/system_stats").mock(side_effect=httpx.ConnectError("нет"))
    write_config(
        path,
        server_url="http://new.test",
        token="t2",
        agent_id="pc",
        workdir=str(tmp_path / "other"),
        comfy_url="http://127.0.0.1:9999",
    )
    ag.check_config()
    assert isinstance(ag.comfy, FakeComfy) and ag.pending_comfy is not None
    ag.step()
    assert isinstance(ag.comfy, ComfyClient) and not ag.comfy_ok
    assert ag.cfg.workdir == tmp_path / "w"
