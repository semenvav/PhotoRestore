"""Приём файлов из `inbox/` (SPEC §5.1).

Опрос каталога (inotify по NFS не видит записи других клиентов). Файл берётся, когда его
размер и mtime совпали на двух опросах подряд. Принятые файлы остаются во «Входящих»:
таблица `inbox_file` помнит (путь, размер, mtime), чтобы не принимать их повторно.
"""

import hashlib
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import imaging, states
from app.config import Settings
from app.db import session_scope, utcnow
from app.models import Actor, InboxFile, Photo, Status
from app.storage import ALLOWED_EXT, Store

log = logging.getLogger(__name__)

IGNORED_NAMES = frozenset({"thumbs.db", "desktop.ini", ".ds_store"})
COPY_CHUNK = 1 << 20

Signature = tuple[int, int]  # (size, mtime_ns)


@dataclass
class PollResult:
    registered: list[int] = field(default_factory=list)
    duplicates: list[int] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    pending: int = 0


class Intake:
    def __init__(self, settings: Settings, factory: sessionmaker[Session], store: Store) -> None:
        self.settings = settings
        self.factory = factory
        self.store = store
        self._pending: dict[str, Signature] = {}
        self._trusted: set[str] = set()
        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()

    @property
    def inbox(self) -> Path:
        return self.settings.inbox_dir

    def ensure_dirs(self) -> None:
        for d in (
            self.inbox,
            self.settings.unsupported_dir,
            self.settings.delivered_dir,
            self.store.root,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def trust(self, rel: str) -> None:
        """Файл записан сервером целиком (загрузка из UI) — принять без ожидания."""
        with self._lock:
            self._trusted.add(rel)

    # --- опрос ---------------------------------------------------------------------------

    def poll(self) -> PollResult:
        with self._poll_lock:
            return self._poll()

    def _poll(self) -> PollResult:
        result = PollResult()
        with session_scope(self.factory) as s:
            known: dict[str, Signature] = {
                row.path: (row.size, row.mtime_ns) for row in s.scalars(select(InboxFile))
            }

        seen: dict[str, Signature] = {}
        ready: list[tuple[int, str, Path, Signature]] = []
        for path in self._walk():
            rel = path.relative_to(self.inbox).as_posix()
            try:
                st = path.stat()
            except FileNotFoundError:
                continue
            sig = (st.st_size, st.st_mtime_ns)
            if st.st_size == 0 or known.get(rel) == sig:
                continue
            with self._lock:
                trusted = rel in self._trusted
            if trusted or self._pending.get(rel) == sig:
                ready.append((st.st_mtime_ns, rel, path, sig))
            else:
                seen[rel] = sig

        for _, rel, path, sig in sorted(ready):
            try:
                if (changed := self._ingest(path, rel, sig, result)) is not None:
                    seen[rel] = changed
            except Exception:
                log.exception("intake: не удалось принять %s, повторю на следующем опросе", rel)
                seen[rel] = sig
            finally:
                with self._lock:
                    self._trusted.discard(rel)

        self._pending = seen
        result.pending = len(seen)
        return result

    def _walk(self) -> list[Path]:
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.inbox):
            here = Path(dirpath)
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".") and not (here == self.inbox and d == "_unsupported")
            ]
            for name in filenames:
                low = name.lower()
                if name.startswith((".", "~$")) or low in IGNORED_NAMES:
                    continue
                files.append(here / name)
        return files

    # --- приём одного файла ----------------------------------------------------------------

    def _ingest(self, path: Path, rel: str, sig: Signature, result: PollResult) -> Signature | None:
        """Принимает файл. Возвращает новую сигнатуру, если файл менялся во время
        копирования и его нужно подождать ещё."""
        ext = path.suffix.lower().lstrip(".")
        if ext not in ALLOWED_EXT:
            self._reject(path, rel, "формат не поддерживается")
            result.unsupported.append(rel)
            return None

        staged, sha256 = self._stage(path, ext)
        try:
            st = path.stat()
            if (st.st_size, st.st_mtime_ns) != sig:
                log.info("intake: %s изменился во время копирования, жду", rel)
                return st.st_size, st.st_mtime_ns
            try:
                inspected = imaging.inspect(staged)
            except imaging.UnsupportedImage as e:
                self._reject(path, rel, f"не открывается: {e}")
                result.unsupported.append(rel)
                return None
            self._register(path, rel, sig, staged, sha256, inspected, result)
            return None
        finally:
            if staged.exists():
                staged.unlink()

    def _stage(self, path: Path, ext: str) -> tuple[Path, str]:
        """Копирует файл в `store/.staging/`, считая sha256 по записанным байтам."""
        self.store.staging_dir.mkdir(parents=True, exist_ok=True)
        staged = self.store.staging_dir / f"{uuid.uuid4().hex}.{ext}"
        digest = hashlib.sha256()
        with path.open("rb") as src, staged.open("xb") as dst:
            while chunk := src.read(COPY_CHUNK):
                digest.update(chunk)
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        return staged, digest.hexdigest()

    def _register(
        self,
        path: Path,
        rel: str,
        sig: Signature,
        staged: Path,
        sha256: str,
        inspected: imaging.Inspected,
        result: PollResult,
    ) -> None:
        album = Path(rel).parent.as_posix()
        album = "" if album == "." else album
        img = inspected.image

        with session_scope(self.factory) as s:
            primary = s.scalar(
                select(Photo).where(Photo.sha256 == sha256, Photo.duplicate_of.is_(None))
            )
            photo = Photo(
                original_name=path.name,
                album=album,
                sha256=sha256,
                phash=inspected.phash,
                width=img.width,
                height=img.height,
                duplicate_of=primary.id if primary else None,
            )
            data: dict[str, object] = {"inbox_path": rel, "album": album}
            if primary:
                data["duplicate_of"] = primary.id
            states.transition(
                s, photo, Status.DUPLICATE if primary else Status.PREP, Actor.SYSTEM, data
            )

            if (aside := self.store.set_aside_orphan(photo.id)) is not None:
                log.warning("intake: каталог %s уже был, перенесён в %s", photo.code, aside)
            original = self.store.original(photo.id, photo.original_name)
            original.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, original)
            os.chmod(original, 0o444)

            imaging.save_thumb(img, self.store.thumb(photo.id))
            imaging.save_preview(img, self.store.preview(photo.id, "original"))

            if primary is None:
                self._note_similar(s, photo)

            row = s.scalar(select(InboxFile).where(InboxFile.path == rel))
            if row is None:
                row = InboxFile(path=rel)
                s.add(row)
            row.size, row.mtime_ns = sig
            row.sha256 = sha256
            row.photo_id = photo.id
            row.seen_at = utcnow()

            (result.duplicates if primary else result.registered).append(photo.id)
            log.info("intake: %s → %s (%s)", rel, photo.code, photo.status)

    def _note_similar(self, s: Session, photo: Photo) -> None:
        if not photo.phash:
            return
        others = s.execute(
            select(Photo.id, Photo.phash).where(
                Photo.id != photo.id, Photo.duplicate_of.is_(None), Photo.phash.is_not(None)
            )
        ).all()
        similar = []
        for other_id, other_hash in others:
            if other_hash is None:
                continue
            distance = imaging.phash_distance(photo.phash, other_hash)
            if distance <= imaging.SIMILAR_MAX_DISTANCE:
                similar.append({"photo_id": other_id, "distance": distance})
        if similar:
            similar.sort(key=lambda d: d["distance"])
            states.log_event(s, photo.id, Actor.SYSTEM, "similar", {"photos": similar})

    def _reject(self, path: Path, rel: str, reason: str) -> None:
        dest = self.settings.unsupported_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        n = 1
        while dest.exists():
            dest = dest.with_name(f"{Path(rel).stem}_{n}{path.suffix}")
            n += 1
        shutil.move(path, dest)
        moved_to = dest.relative_to(self.inbox).as_posix()
        with session_scope(self.factory) as s:
            states.log_event(
                s,
                None,
                Actor.SYSTEM,
                "unsupported",
                {"path": rel, "moved_to": moved_to, "reason": reason},
            )
        log.warning("intake: %s → %s (%s)", rel, moved_to, reason)
