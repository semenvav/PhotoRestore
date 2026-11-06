from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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
