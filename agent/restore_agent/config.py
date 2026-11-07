"""`config.toml` агента (ТЗ §7.1)."""

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(Exception):
    pass


def default_workdir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "restore-agent" if base else Path.home() / ".restore-agent"


def default_config_path() -> Path:
    """`agent\\config.toml`: рядом с venv установки (agent\\.venv) или с исходниками пакета."""
    candidates = [
        Path(sys.prefix).parent / "config.toml",
        Path(__file__).parents[1] / "config.toml",
    ]
    return next((p for p in candidates if p.is_file()), candidates[-1])


@dataclass(frozen=True)
class Config:
    server_url: str  # https://<хост>; путь /api/agent/v1 агент добавляет сам
    token: str
    agent_id: str
    comfy_url: str = "http://127.0.0.1:8188"
    workdir: Path = field(default_factory=default_workdir)
    comfy_launch_cmd: str | None = None
    name: str | None = None
    fake_comfy: bool = False  # FAKE_COMFY=1: вместо ComfyUI — вход с напечатанным seed


def load(path: Path) -> Config:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{path}: {e}") from e
    missing = [k for k in ("server_url", "token", "agent_id") if not raw.get(k)]
    if missing:
        raise ConfigError(f"{path}: не заданы {', '.join(missing)}")
    workdir = Path(raw["workdir"]).expanduser() if raw.get("workdir") else default_workdir()
    return Config(
        server_url=str(raw["server_url"]).rstrip("/"),
        token=str(raw["token"]),
        agent_id=str(raw["agent_id"]),
        comfy_url=str(raw.get("comfy_url") or Config.comfy_url).rstrip("/"),
        workdir=workdir,
        comfy_launch_cmd=raw.get("comfy_launch_cmd") or None,
        name=raw.get("name") or None,
        fake_comfy=os.environ.get("FAKE_COMFY") == "1" or bool(raw.get("fake_comfy")),
    )
