"""Результаты задач агента: входы, загрузка выходов, `complete` (ТЗ §6).

Выходы ключуются `(job_id, index)` и до `complete` лежат в `store/R-xxxx/jobs/<job_id>/`;
повторная загрузка того же индекса перезаписывает файл. Варианты появляются в БД только при
`complete`, поэтому половинчатая задача не оставляет мусора в ленте.
"""

import io
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy.orm import Session

from app import imaging, states
from app.db import utcnow
from app.jobs import STAGES, outputs_count
from app.models import Actor, Job, JobStatus, JobType, Photo, Variant, VariantSource
from app.storage import Store, variant_label
from app.workflows import Registry

SOURCES = {
    JobType.EDIT_LOCAL: VariantSource.LOCAL_EDIT,
    JobType.UPSCALE_LOCAL: VariantSource.UPSCALE,
}


class ResultError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def input_path(session: Session, store: Store, job: Job, name: str) -> Path:
    if name != "source":
        raise ResultError(404, f"у задачи нет входа {name!r}")
    variant_id = job.payload.get("source_variant_id")
    if variant_id:
        variant = session.get(Variant, variant_id)
        if variant is None or variant.photo_id != job.photo_id:
            raise ResultError(404, f"вариант {variant_id} не найден")
        return store.photo_dir(job.photo_id) / variant.path
    return store.prepared(job.photo_id)


def _meta_path(output: Path) -> Path:
    return output.with_suffix(".json")


def save_output(store: Store, job: Job, index: int, data: bytes, meta: dict[str, Any]) -> None:
    if not 0 <= index < outputs_count(job):
        raise ResultError(400, f"индекс {index} вне диапазона 0…{outputs_count(job) - 1}")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format != "PNG":
                raise ResultError(400, f"ожидается PNG, получен {img.format}")
            img.load()
            size = img.size
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        raise ResultError(400, f"файл не открывается: {e}") from e

    dest = store.job_output(job.photo_id, job.id, index)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "width": size[0], "height": size[1]}
    for path, content in ((dest, data), (_meta_path(dest), json.dumps(meta).encode())):
        tmp = path.with_name(f".{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)


def complete(session: Session, store: Store, registry: Registry, job: Job) -> list[Variant]:
    """Создаёт варианты из выходов и двигает статус фото. Вызывающий фиксирует транзакцию
    и затем убирает `jobs/<id>/` (`cleanup`): до commit выходы остаются на месте, и
    повторный `complete` после сбоя пройдёт заново."""
    n = outputs_count(job)
    outputs = [store.job_output(job.photo_id, job.id, i) for i in range(n)]
    missing = [str(i) for i, p in enumerate(outputs) if not p.is_file()]
    if missing:
        raise ResultError(409, f"не загружены результаты: {', '.join(missing)}")

    photo = session.get(Photo, job.photo_id)
    assert photo is not None
    payload = job.payload
    seeds = payload.get("seeds") or [None] * n
    workflow = payload.get("workflow")
    if workflow is None and (wf := registry.for_type(job.type)) is not None:
        workflow = wf.describe()

    variants = []
    for i, src in enumerate(outputs):
        meta = json.loads(_meta_path(src).read_text(encoding="utf-8"))
        number = store.next_variant_number(photo.id)
        dest = store.variant_file(photo.id, number)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{uuid.uuid4().hex}.tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        os.chmod(dest, 0o444)  # варианты неизменяемы

        img = imaging.load_oriented(dest)
        imaging.save_preview(img, store.preview(photo.id, variant_label(number)))
        imaging.save_thumb(img, store.variant_thumb(photo.id, number))

        variant = Variant(
            photo_id=photo.id,
            parent_id=payload.get("source_variant_id"),
            source=SOURCES[job.type],
            job_id=job.id,
            round=int(payload.get("round", photo.round)),
            prompt=payload.get("prompt"),
            params={
                "seed": meta.get("seed", seeds[i]),
                **(workflow or {}),
                **(payload.get("params") or {}),
            },
            path=dest.relative_to(store.photo_dir(photo.id)).as_posix(),
            width=img.width,
            height=img.height,
            duration_s=meta.get("duration_s"),
        )
        session.add(variant)
        variants.append(variant)
    session.flush()

    now = utcnow()
    job.status = JobStatus.DONE
    job.finished_at = now
    job.lease_until = None
    job.error = None
    job.progress = {"done": n, "total": n, "stage": "done"}
    stage = STAGES[job.type]
    states.transition(
        session,
        photo,
        stage.done,
        Actor.AGENT,
        {"job_id": job.id, "variants": [v.label for v in variants]},
    )
    photo.attention = False
    photo.attention_reason = None
    return variants


def cleanup(store: Store, job: Job) -> None:
    shutil.rmtree(store.job_dir(job.photo_id, job.id), ignore_errors=True)
