"""Обновление агента до сборки, которая работает на сервере (`agent\\update.ps1`).

    python -m restore_agent.update [--config config.toml] [--version 0.2.1]

1. Номер сборки — `--version` или `GET {server_url}/healthz`. Номер есть только у сборок ветки
   по умолчанию (`0.2`, `0.2.1`…); у них же есть git-тег `v<номер>` (см. CI).
2. Архив этого тега с GitHub распаковывается во временную папку, `pip install …/agent` ставит
   агента в тот venv, из которого запущен этот модуль. Рабочая копия репозитория не участвует.
3. На время установки задача Планировщика останавливается, потом запускается снова. Взятая
   задача вернётся в очередь по аренде, а готовые результаты агент найдёт в workdir.
"""

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

from restore_agent.config import ConfigError, default_config_path, load

REPO = "https://github.com/semenvav/PhotoRestore"
TASK = "photo-restore agent"
RELEASE = re.compile(r"^\d+\.\d+(\.\d+)*$")

Run = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


class UpdateError(Exception):
    pass


def server_version(http: httpx.Client) -> str:
    try:
        r = http.get("/healthz")
        r.raise_for_status()
        version = str(r.json().get("version") or "")
    except (httpx.HTTPError, ValueError) as e:
        raise UpdateError(f"не удалось узнать версию сервера: {e}") from e
    if not RELEASE.match(version):
        raise UpdateError(
            f"сервер работает на сборке «{version}» без номера (не из ветки по умолчанию) — "
            "укажите версию явно: update.ps1 -Version 0.2.1"
        )
    return version


def archive_url(repo: str, version: str) -> str:
    return f"{repo.rstrip('/')}/archive/refs/tags/v{version}.zip"


def download(http: httpx.Client, url: str, dest: Path) -> Path:
    try:
        with http.stream("GET", url) as r:
            if r.status_code == 404:
                raise UpdateError(f"на GitHub нет такой версии: {url}")
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    except httpx.HTTPError as e:
        raise UpdateError(f"не удалось скачать {url}: {e}") from e
    return dest


def extract_agent(archive: Path, dest: Path) -> Path:
    """Распаковать архив тега и вернуть его папку `agent/`."""
    root = dest.resolve()
    try:
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise UpdateError(f"путь вне папки в архиве: {name}")
            z.extractall(root)
    except zipfile.BadZipFile as e:
        raise UpdateError(f"архив повреждён: {e}") from e
    found = sorted(root.glob("*/agent/pyproject.toml"))
    if len(found) != 1:
        raise UpdateError("в архиве нет agent/pyproject.toml — в этой версии нет агента")
    return found[0].parent


def run(cmd: list[str]) -> "subprocess.CompletedProcess[str]":
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW
    return subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)


def install(agent_dir: Path, task: str | None, run: Run = run) -> None:
    """pip install при остановленной задаче Планировщика; задачу — запустить снова."""
    has_task = bool(task) and run(["schtasks", "/Query", "/TN", str(task)]).returncode == 0
    if has_task:
        print(f"Останавливаю задачу «{task}»", flush=True)
        run(["schtasks", "/End", "/TN", str(task)])
    try:
        pip = run([sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", str(agent_dir)])
        if pip.returncode != 0:
            raise UpdateError(f"pip install не удался, агент остался прежним:\n{pip.stderr}")
    finally:
        if has_task:
            run(["schtasks", "/Run", "/TN", str(task)])
            print(f"Задача «{task}» запущена", flush=True)
    if not has_task:
        print("Задачи Планировщика нет — перезапустите агента вручную", flush=True)


def main(argv: list[str] | None = None, run: Run = run) -> int:
    parser = argparse.ArgumentParser(prog="restore_agent.update", description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--version", default=None, help="номер сборки, например 0.2.1")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--task", default=TASK, help="задача Планировщика; пусто — без неё")
    args = parser.parse_args(argv)
    try:
        version = args.version
        if version is None:
            cfg = load(args.config or default_config_path())
            with httpx.Client(base_url=cfg.server_url, timeout=30, follow_redirects=True) as http:
                version = server_version(http)
            print(f"Сервер работает на сборке {version}", flush=True)
        elif not RELEASE.match(version):
            raise UpdateError(f"версия «{version}» — ожидается номер вида 0.2.1")
        with (
            tempfile.TemporaryDirectory(prefix="restore-agent-") as tmp,
            httpx.Client(timeout=120, follow_redirects=True) as http,
        ):
            url = archive_url(args.repo, version)
            print(f"Скачиваю {url}", flush=True)
            agent_dir = extract_agent(download(http, url, Path(tmp) / "src.zip"), Path(tmp))
            install(agent_dir, args.task or None, run)
    except (UpdateError, ConfigError) as e:
        print(f"Обновление не выполнено: {e}", file=sys.stderr, flush=True)
        return 1
    installed = run([sys.executable, "-m", "restore_agent", "--version"]).stdout.strip()
    print(f"Готово: агент {installed} из сборки {version}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
