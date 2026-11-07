"""Обновление агента до сборки сервера (`agent\\update.ps1` → `restore_agent.update`)."""

import io
import subprocess
import zipfile
from pathlib import Path

import httpx
import pytest
import respx

from restore_agent import update
from restore_agent.update import UpdateError

S = "http://server.test"
REPO = "https://github.com/owner/repo"


def archive(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


TAG_ZIP = archive(
    {
        "repo-0.2.1/agent/pyproject.toml": "[project]\nname = 'restore-agent'\n",
        "repo-0.2.1/agent/restore_agent/__init__.py": "__version__ = '0.2.0'\n",
        "repo-0.2.1/server/pyproject.toml": "",
    }
)


class Commands:
    """Подменённый subprocess.run: запоминает команды, коды возврата — по первому слову."""

    def __init__(self, codes: dict[str, int] | None = None) -> None:
        self.seen: list[list[str]] = []
        self.codes = codes or {}

    def __call__(self, cmd: list[str]) -> "subprocess.CompletedProcess[str]":
        self.seen.append(cmd)
        key = cmd[1] if cmd[0] == "schtasks" else " ".join(cmd[1:4])
        out = "0.2.0\n" if "--version" in cmd else ""
        return subprocess.CompletedProcess(cmd, self.codes.get(key, 0), out, "ошибка pip")

    def kinds(self) -> list[str]:
        return [c[1] if c[0] == "schtasks" else c[3] if c[2] == "pip" else c[2] for c in self.seen]


@pytest.mark.parametrize("version", ["0.2", "0.2.1", "1.10.3"])
@respx.mock
def test_server_version(version: str) -> None:
    respx.get(f"{S}/healthz").respond(json={"ok": True, "version": version})
    assert update.server_version(httpx.Client(base_url=S)) == version


@pytest.mark.parametrize("version", ["dev", "master@abc1234", "feature@1234567", ""])
@respx.mock
def test_server_without_release_number(version: str) -> None:
    respx.get(f"{S}/healthz").respond(json={"ok": True, "version": version})
    with pytest.raises(UpdateError, match="-Version"):
        update.server_version(httpx.Client(base_url=S))


@respx.mock
def test_server_unreachable() -> None:
    respx.get(f"{S}/healthz").mock(side_effect=httpx.ConnectError("нет сети"))
    with pytest.raises(UpdateError, match="версию сервера"):
        update.server_version(httpx.Client(base_url=S))


def test_archive_url() -> None:
    assert update.archive_url(REPO + "/", "0.2.1") == f"{REPO}/archive/refs/tags/v0.2.1.zip"


def test_extract_agent(tmp_path: Path) -> None:
    src = tmp_path / "src.zip"
    src.write_bytes(TAG_ZIP)
    agent_dir = update.extract_agent(src, tmp_path / "out")
    assert agent_dir == (tmp_path / "out" / "repo-0.2.1" / "agent").resolve()
    assert (agent_dir / "restore_agent" / "__init__.py").is_file()


@pytest.mark.parametrize(
    ("files", "error"),
    [
        ({"repo-0.1/server/pyproject.toml": ""}, "нет агента"),
        ({"../evil/agent/pyproject.toml": ""}, "вне папки"),
    ],
)
def test_extract_rejects_bad_archives(tmp_path: Path, files: dict[str, str], error: str) -> None:
    src = tmp_path / "src.zip"
    src.write_bytes(archive(files))
    with pytest.raises(UpdateError, match=error):
        update.extract_agent(src, tmp_path / "out")
    assert not (tmp_path / "evil").exists()


def test_install_stops_and_restarts_task(tmp_path: Path) -> None:
    run = Commands()
    update.install(tmp_path, "photo-restore agent", run)
    assert run.kinds() == ["/Query", "/End", "install", "/Run"]
    assert run.seen[2][-1] == str(tmp_path)


def test_failed_pip_still_restarts_task(tmp_path: Path) -> None:
    run = Commands({"-m pip install": 1})
    with pytest.raises(UpdateError, match="агент остался прежним"):
        update.install(tmp_path, "photo-restore agent", run)
    assert run.kinds() == ["/Query", "/End", "install", "/Run"]


def test_install_without_task(tmp_path: Path) -> None:
    run = Commands({"/Query": 1})
    update.install(tmp_path, "photo-restore agent", run)
    assert run.kinds() == ["/Query", "install"]


@respx.mock
def test_main_installs_server_build(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.toml"
    config.write_text(f"server_url = '{S}'\ntoken = 't'\nagent_id = 'pc'\n", encoding="utf-8")
    respx.get(f"{S}/healthz").respond(json={"ok": True, "version": "0.2.1"})
    respx.get(f"{REPO}/archive/refs/tags/v0.2.1.zip").respond(content=TAG_ZIP)
    run = Commands()
    assert update.main(["--config", str(config), "--repo", REPO], run) == 0
    assert run.kinds() == ["/Query", "/End", "install", "/Run", "restore_agent"]
    assert Path(run.seen[2][-1]).parts[-2:] == ("repo-0.2.1", "agent")
    out = capsys.readouterr().out
    assert "сборке 0.2.1" in out and "Готово: агент 0.2.0 из сборки 0.2.1" in out


@respx.mock
def test_main_with_unknown_version(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    respx.get(f"{REPO}/archive/refs/tags/v9.9.zip").respond(404)
    run = Commands()
    assert update.main(["--version", "9.9", "--repo", REPO], run) == 1
    assert run.seen == []  # агента не трогали
    assert "нет такой версии" in capsys.readouterr().err


def test_main_rejects_malformed_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert update.main(["--version", "v0.2; rm -rf"], Commands()) == 1
    assert "ожидается номер" in capsys.readouterr().err
