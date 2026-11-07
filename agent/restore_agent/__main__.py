"""`python -m restore_agent [--config config.toml]`."""

import argparse
import logging
import logging.handlers
import signal
import sys
import threading
from pathlib import Path

from restore_agent import __version__, wiring
from restore_agent.client import ServerClient
from restore_agent.config import Config, ConfigError, default_config_path, load
from restore_agent.runner import Agent

log = logging.getLogger("restore_agent")


def setup_logging(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    file = logging.handlers.RotatingFileHandler(
        workdir / "agent.log", maxBytes=5 * 2**20, backupCount=5, encoding="utf-8"
    )
    file.setFormatter(fmt)
    root.addHandler(file)
    if sys.stderr is not None:  # под pythonw консоли нет
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # под pythonw stderr нет: без этого необработанная ошибка исчезла бы бесследно
    sys.excepthook = lambda t, v, tb: log.critical("агент упал", exc_info=(t, v, tb))
    threading.excepthook = _thread_crash


def _thread_crash(a: threading.ExceptHookArgs) -> None:
    name = a.thread.name if a.thread else "?"
    if a.exc_value is not None:
        log.critical("поток %s упал", name, exc_info=a.exc_value)


def build(cfg: Config, stop: threading.Event, config_path: Path | None = None) -> Agent:
    server = ServerClient(wiring.server_http(cfg), stop)
    return Agent(cfg, server, wiring.comfy_client(cfg), stop, config_path=config_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="restore_agent", description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    config_path: Path = args.config or default_config_path()
    try:
        cfg = load(config_path)
    except ConfigError as e:
        print(f"restore_agent: {e}", file=sys.stderr)
        return 2
    setup_logging(cfg.workdir)
    stop = threading.Event()
    agent = build(cfg, stop, config_path)
    signal.signal(signal.SIGINT, lambda *_: agent.shutdown())
    signal.signal(signal.SIGTERM, lambda *_: agent.shutdown())
    agent.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
