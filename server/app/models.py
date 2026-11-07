from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    ForeignKey,
    Index,
    String,
    Text,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base, UTCDateTime, utcnow


class Status(StrEnum):
    PREP = "PREP"
    QUEUED_EDIT = "QUEUED_EDIT"
    EDITING = "EDITING"
    REVIEW = "REVIEW"
    GEMINI = "GEMINI"
    QUEUED_UPSCALE = "QUEUED_UPSCALE"
    UPSCALING = "UPSCALING"
    FINAL_REVIEW = "FINAL_REVIEW"
    DELIVERED = "DELIVERED"
    ACCEPTED = "ACCEPTED"
    DUPLICATE = "DUPLICATE"
    ON_HOLD = "ON_HOLD"


class Actor(StrEnum):
    OWNER = "owner"
    FAMILY = "family"
    AGENT = "agent"
    SYSTEM = "system"


class VariantSource(StrEnum):
    LOCAL_EDIT = "local_edit"
    GEMINI = "gemini"
    MANUAL = "manual"
    UPSCALE = "upscale"


class JobType(StrEnum):
    EDIT_LOCAL = "edit_local"
    UPSCALE_LOCAL = "upscale_local"
    EDIT_GEMINI = "edit_gemini"


class JobStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunState(StrEnum):
    ACTIVE = "active"
    STOPPED = "stopped"
    COMPLETED = "completed"


class RunScope(StrEnum):
    ALL = "all"
    SELECTED = "selected"


class PresetKind(StrEnum):
    BASE = "base"
    ADDON = "addon"


def _enum(cls: type[StrEnum], length: int = 20) -> SAEnum:
    # VARCHAR без CHECK: новые значения не требуют миграции, проверяет Python
    return SAEnum(
        cls,
        native_enum=False,
        create_constraint=False,
        length=length,
        values_callable=lambda e: [m.value for m in e],
    )


def photo_code(photo_id: int) -> str:
    return f"R-{photo_id:04d}"


class Photo(Base):
    __tablename__ = "photo"
    __table_args__ = (
        # sha256 уникален среди «первых» фото; у дубля тот же хеш и ссылка duplicate_of
        Index(
            "uq_photo_sha256_primary",
            "sha256",
            unique=True,
            sqlite_where=text("duplicate_of IS NULL"),
        ),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[Status] = mapped_column(_enum(Status), index=True)
    attention: Mapped[bool] = mapped_column(default=False)
    attention_reason: Mapped[str | None] = mapped_column(Text)
    original_name: Mapped[str] = mapped_column(String(255))
    album: Mapped[str] = mapped_column(String(255), default="", index=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    phash: Mapped[str | None] = mapped_column(String(16))
    width: Mapped[int]
    height: Mapped[int]
    prep_params: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    accepted_variant_id: Mapped[int | None] = mapped_column(
        ForeignKey("variant.id", use_alter=True)
    )
    final_variant_id: Mapped[int | None] = mapped_column(ForeignKey("variant.id", use_alter=True))
    duplicate_of: Mapped[int | None] = mapped_column(ForeignKey("photo.id"))
    round: Mapped[int] = mapped_column(default=1)
    family_comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    @property
    def code(self) -> str:
        return photo_code(self.id)


class Variant(Base):
    __tablename__ = "variant"

    id: Mapped[int] = mapped_column(primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id"), index=True)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("variant.id"))
    source: Mapped[VariantSource] = mapped_column(_enum(VariantSource))
    job_id: Mapped[int | None] = mapped_column(ForeignKey("job.id"))
    round: Mapped[int] = mapped_column(default=1)
    prompt: Mapped[str | None] = mapped_column(Text)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    path: Mapped[str] = mapped_column(String(255))  # относительно store/R-xxxx/
    width: Mapped[int]
    height: Mapped[int]
    cost_usd: Mapped[float | None]
    duration_s: Mapped[float | None]
    hidden: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    @property
    def number(self) -> int:
        """Номер из имени файла: `variants/v007.png` → 7."""
        return int(self.path.rsplit("/", 1)[-1].removeprefix("v").removesuffix(".png"))

    @property
    def label(self) -> str:
        return f"v{self.number:03d}"


class Job(Base):
    __tablename__ = "job"

    id: Mapped[int] = mapped_column(primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id"), index=True)
    type: Mapped[JobType] = mapped_column(_enum(JobType))
    status: Mapped[JobStatus] = mapped_column(_enum(JobStatus), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    priority: Mapped[int] = mapped_column(default=0)
    attempts: Mapped[int] = mapped_column(default=0)
    max_attempts: Mapped[int] = mapped_column(default=3)
    agent_id: Mapped[str | None] = mapped_column(String(64))
    # запуск, в рамках которого задачу взяли последний раз — для прогресса запуска
    run_id: Mapped[int | None] = mapped_column(ForeignKey("run.id"), index=True)
    lease_until: Mapped[datetime | None] = mapped_column(UTCDateTime)
    progress: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class Run(Base):
    __tablename__ = "run"

    id: Mapped[int] = mapped_column(primary_key=True)
    state: Mapped[RunState] = mapped_column(_enum(RunState), index=True)
    scope: Mapped[RunScope] = mapped_column(_enum(RunScope))
    job_ids: Mapped[list[int] | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class Event(Base):
    __tablename__ = "event"

    id: Mapped[int] = mapped_column(primary_key=True)
    # NULL — системное событие без фото (например, файл убран в _unsupported)
    photo_id: Mapped[int | None] = mapped_column(ForeignKey("photo.id"), index=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    actor: Mapped[Actor] = mapped_column(_enum(Actor))
    kind: Mapped[str] = mapped_column(String(40))
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Preset(Base):
    __tablename__ = "preset"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[PresetKind] = mapped_column(_enum(PresetKind))
    name: Mapped[str] = mapped_column(String(100))
    text: Mapped[str] = mapped_column(Text)
    sort: Mapped[int] = mapped_column(default=0)


class Agent(Base):
    __tablename__ = "agent"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(100))
    version: Mapped[str | None] = mapped_column(String(40))
    last_seen: Mapped[datetime | None] = mapped_column(UTCDateTime)
    comfy_ok: Mapped[bool] = mapped_column(default=False)
    gpu_name: Mapped[str | None] = mapped_column(String(100))
    vram_free_mb: Mapped[int | None]


class Setting(Base):
    __tablename__ = "setting"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)


class InboxFile(Base):
    """Файл во «Входящих», который уже принят. Файлы остаются на месте, поэтому
    сервер помнит (путь, размер, mtime) и не принимает их повторно."""

    __tablename__ = "inbox_file"

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(1024), unique=True)  # относительно inbox/, posix
    size: Mapped[int] = mapped_column(BigInteger)
    mtime_ns: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    photo_id: Mapped[int | None] = mapped_column(ForeignKey("photo.id"))
    seen_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
