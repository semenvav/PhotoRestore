"""Клиенты сервера и ComfyUI по конфигу: при старте и при изменении `config.toml` на лету."""

import httpx

from restore_agent import __version__
from restore_agent.comfy import Comfy, ComfyClient
from restore_agent.config import Config
from restore_agent.fake import FakeComfy


def server_http(cfg: Config) -> httpx.Client:
    return httpx.Client(
        base_url=cfg.server_url,
        headers={
            "Authorization": f"Bearer {cfg.token}",
            "User-Agent": f"restore-agent/{__version__}",
        },
        timeout=httpx.Timeout(120, connect=10),
    )


def comfy_client(cfg: Config) -> Comfy:
    if cfg.fake_comfy:
        return FakeComfy()
    return ComfyClient(httpx.Client(base_url=cfg.comfy_url, timeout=120))
