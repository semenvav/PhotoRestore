from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# workflows/ лежит в корне репозитория; в образе — /srv/workflows
DEFAULT_WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / "workflows"


class Settings(BaseSettings):
    """Параметры процесса из окружения (и `.env` при локальной разработке).

    Пользовательские настройки (пресеты, параметры edit/upscale, бюджет) живут в таблице
    `setting`, а не здесь.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("/data")
    db_path: Path = Path("/db/restore.db")

    intake_enabled: bool = True
    intake_poll_s: float = 30.0
    max_image_pixels: int = 300_000_000

    display_tz: str = "Europe/Moscow"
    log_level: str = "INFO"

    # номер сборки: CI передаёт его в образ (`APP_VERSION`), вне контейнера — dev
    app_version: str = "dev"

    # агент (ТЗ §6): без токена API агента отвечает 503
    agent_token: SecretStr | None = None
    lease_s: float = 600.0  # аренда задачи: claim и каждый progress продлевают её
    sweep_interval_s: float = 60.0  # как часто возвращать просроченные задачи в очередь
    workflows_dir: Path = DEFAULT_WORKFLOWS_DIR

    # Gemini (ТЗ §8): без ключа отправка недоступна; FAKE_GEMINI — вход с пометкой, цена 0
    gemini_api_key: SecretStr | None = None
    fake_gemini: bool = False
    gemini_timeout_s: float = 300.0
    gemini_retry_base_s: float = 2.0  # паузы перед повторами 429/5xx: 2, 4, 8 с

    # отец (ТЗ §5.6): запросы на этот хост (через запятую — несколько) видят только /family
    family_host: str = ""

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @property
    def unsupported_dir(self) -> Path:
        return self.inbox_dir / "_unsupported"

    @property
    def delivered_dir(self) -> Path:
        return self.data_dir / "delivered"

    @property
    def store_dir(self) -> Path:
        return self.data_dir / "store"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
