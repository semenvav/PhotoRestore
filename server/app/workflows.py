"""Реестр workflow ComfyUI (ТЗ §7.3).

`workflows/registry.yaml` описывает каждый workflow: тип задачи, версию, параметры и их
привязку к узлам по заголовку (`_meta.title`). При старте сервер проверяет все привязки и
без них не запускается. Агенту уходит граф с уже подставленными параметрами и привязки,
по которым он сам ставит имя входного файла и seed каждого выхода.

У параметра может быть `choices`: короткий ключ (его видят настройки и UI) → значение для
графа, например `7b_fp16` → `seedvr2_7b_fp16.safetensors`.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.models import JobType

REGISTRY_FILE = "registry.yaml"
# без них сервер не стартует: чистка и апскейл на ПК
REQUIRED_TYPES = (JobType.EDIT_LOCAL, JobType.UPSCALE_LOCAL)


class WorkflowError(Exception):
    pass


@dataclass(frozen=True)
class Binding:
    node: str  # заголовок узла
    field: str | None = None
    default: Any = None
    choices: dict[str, Any] | None = None  # ключ → значение в графе

    def resolve(self, value: Any) -> Any:
        return self.choices.get(value, value) if self.choices else value


@dataclass(frozen=True)
class Workflow:
    name: str
    job_type: JobType
    version: int
    model: str | None
    params: dict[str, Binding]
    image: Binding
    output: str
    graph: dict[str, Any] = field(repr=False)
    node_ids: dict[str, str] = field(repr=False)  # заголовок → id узла

    def defaults(self) -> dict[str, Any]:
        return {k: b.default for k, b in self.params.items() if b.default is not None}

    def build(self, values: dict[str, Any]) -> dict[str, Any]:
        """Копия графа с подставленными параметрами; не заданные и без default — как есть."""
        graph = copy.deepcopy(self.graph)
        for name, binding in self.params.items():
            value = values.get(name, binding.default)
            if value is not None:
                inputs = graph[self.node_ids[binding.node]]["inputs"]
                inputs[binding.field] = binding.resolve(value)
        return graph

    def agent_bindings(self) -> dict[str, Any]:
        seed = self.params.get("seed")
        return {
            "image": {"node": self.image.node, "field": self.image.field},
            "seed": {"node": seed.node, "field": seed.field} if seed else None,
            "output": {"node": self.output},
        }

    def choices(self, param: str) -> dict[str, Any]:
        binding = self.params.get(param)
        return dict(binding.choices) if binding and binding.choices else {}

    def describe(self) -> dict[str, Any]:
        """Что записать в `variant.params`, чтобы результат можно было воспроизвести."""
        return {"workflow": self.name, "workflow_version": self.version, "model": self.model}


@dataclass
class Registry:
    workflows: dict[str, Workflow]

    def for_type(self, job_type: JobType) -> Workflow | None:
        for wf in self.workflows.values():
            if wf.job_type == job_type:
                return wf
        return None

    def job_types(self) -> set[JobType]:
        return {wf.job_type for wf in self.workflows.values()}


def load_registry(
    directory: Path, required: tuple[JobType, ...] = (JobType.EDIT_LOCAL,)
) -> Registry:
    path = directory / REGISTRY_FILE
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise WorkflowError(f"{path}: {e}") from e
    if not isinstance(raw, dict):
        raise WorkflowError(f"{path}: ожидается словарь workflow")

    workflows: dict[str, Workflow] = {}
    for name, entry in raw.items():
        wf = _load_one(directory, str(name), entry)
        if other := next((w for w in workflows.values() if w.job_type == wf.job_type), None):
            raise WorkflowError(f"{name}: тип {wf.job_type} уже занят workflow {other.name}")
        workflows[wf.name] = wf

    registry = Registry(workflows)
    missing = [t.value for t in required if registry.for_type(t) is None]
    if missing:
        raise WorkflowError(f"{path}: нет workflow для {', '.join(missing)}")
    return registry


def _load_one(directory: Path, name: str, entry: Any) -> Workflow:
    def fail(msg: str) -> WorkflowError:
        return WorkflowError(f"workflow {name}: {msg}")

    if not isinstance(entry, dict):
        raise fail("ожидается словарь")
    try:
        job_type = JobType(entry["job_type"])
        file = str(entry["file"])
        version = int(entry["version"])
        image_raw = entry["image"]
        output = str(entry["output"])
    except (KeyError, ValueError, TypeError) as e:
        raise fail(f"нет или неверно поле {e}") from e

    try:
        graph = json.loads((directory / file).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise fail(f"{file}: {e}") from e
    if not isinstance(graph, dict) or not all(
        isinstance(n, dict) and "class_type" in n and "inputs" in n for n in graph.values()
    ):
        raise fail(f"{file}: ожидается API-формат ComfyUI (Workflow → Export (API))")

    titles: dict[str, list[str]] = {}
    for node_id, node in graph.items():
        title = (node.get("_meta") or {}).get("title")
        if title:
            titles.setdefault(title, []).append(node_id)

    def node_for(title: str) -> str:
        ids = titles.get(title, [])
        if len(ids) != 1:
            found = "не найден" if not ids else f"найден {len(ids)} раза"
            raise fail(f"узел с заголовком {title!r} {found}")
        return ids[0]

    def binding(raw: Any, what: str) -> Binding:
        if not isinstance(raw, dict) or "node" not in raw or "field" not in raw:
            raise fail(f"{what}: нужны node и field")
        choices = raw.get("choices")
        if choices is not None and (not isinstance(choices, dict) or not choices):
            raise fail(f"{what}: choices — непустой словарь «ключ: значение»")
        b = Binding(
            node=str(raw["node"]),
            field=str(raw["field"]),
            default=raw.get("default"),
            choices={str(k): v for k, v in choices.items()} if choices else None,
        )
        if b.choices and b.default is not None and b.default not in b.choices:
            raise fail(f"{what}: default {b.default!r} не из choices")
        inputs = graph[node_for(b.node)]["inputs"]
        if b.field not in inputs:
            raise fail(f"{what}: у узла {b.node} нет поля {b.field!r}")
        if isinstance(inputs[b.field], list):
            raise fail(f"{what}: поле {b.node}.{b.field} подключено к другому узлу")
        return b

    params = {str(k): binding(v, f"параметр {k}") for k, v in (entry.get("params") or {}).items()}
    image = binding(image_raw, "image")
    node_for(output)

    bound = [image.node, output, *(b.node for b in params.values())]
    return Workflow(
        name=name,
        job_type=job_type,
        version=version,
        model=entry.get("model"),
        params=params,
        image=image,
        output=output,
        graph=graph,
        node_ids={t: node_for(t) for t in bound},
    )
