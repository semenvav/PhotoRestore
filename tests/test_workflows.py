import json
from pathlib import Path
from typing import Any

import pytest

from app.config import DEFAULT_WORKFLOWS_DIR
from app.models import JobType
from app.workflows import REQUIRED_TYPES, WorkflowError, load_registry

GRAPH: dict[str, Any] = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}, "_meta": {"title": "IN_IMAGE"}},
    "2": {
        "class_type": "Text",
        "inputs": {"prompt": "", "clip": ["9", 0]},
        "_meta": {"title": "IN_PROMPT"},
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {"seed": 0, "steps": ["4", 0]},
        "_meta": {"title": "IN_SEED"},
    },
    "4": {"class_type": "PrimitiveInt", "inputs": {"value": 8}, "_meta": {"title": "IN_STEPS"}},
    "5": {
        "class_type": "SaveImage",
        "inputs": {"images": ["3", 0]},
        "_meta": {"title": "OUT_IMAGE"},
    },
}

REGISTRY = """
edit:
  job_type: edit_local
  file: edit.json
  version: 3
  model: test-model
  params:
    prompt: {node: IN_PROMPT, field: prompt}
    seed:   {node: IN_SEED,   field: seed}
    steps:  {node: IN_STEPS,  field: value, default: 8}
  image: {node: IN_IMAGE, field: image}
  output: OUT_IMAGE
"""


def make(tmp_path: Path, graph: dict[str, Any] = GRAPH, registry: str = REGISTRY) -> Path:
    (tmp_path / "edit.json").write_text(json.dumps(graph), encoding="utf-8")
    (tmp_path / "registry.yaml").write_text(registry, encoding="utf-8")
    return tmp_path


def test_repository_workflows_are_valid() -> None:
    """Реестр из репозитория — тот, с которым стартует сервер."""
    registry = load_registry(DEFAULT_WORKFLOWS_DIR)
    wf = registry.for_type(JobType.EDIT_LOCAL)
    assert wf is not None
    assert wf.defaults() == {"steps": 8, "megapixels": 1.5}
    graph = wf.build({"prompt": "restore", "steps": 4, "megapixels": 1.0})
    by_title = {n["_meta"]["title"]: n for n in graph.values()}
    assert by_title["IN_PROMPT"]["inputs"]["prompt"] == "restore"
    assert by_title["IN_STEPS"]["inputs"]["value"] == 4
    assert by_title["IN_MEGAPIXELS"]["inputs"]["megapixels"] == 1.0
    assert by_title["OUT_IMAGE"]["class_type"] == "SaveImage"


def test_build_substitutes_and_keeps_original(tmp_path: Path) -> None:
    wf = load_registry(make(tmp_path)).for_type(JobType.EDIT_LOCAL)
    assert wf is not None
    graph = wf.build({"prompt": "hello"})
    assert graph["2"]["inputs"]["prompt"] == "hello"
    assert graph["4"]["inputs"]["value"] == 8  # default
    assert graph["3"]["inputs"]["seed"] == 0  # seed ставит агент
    assert graph["2"]["inputs"]["clip"] == ["9", 0]
    assert wf.graph["2"]["inputs"]["prompt"] == ""  # исходник не тронут
    assert wf.agent_bindings() == {
        "image": {"node": "IN_IMAGE", "field": "image"},
        "seed": {"node": "IN_SEED", "field": "seed"},
        "output": {"node": "OUT_IMAGE"},
    }
    assert wf.describe() == {"workflow": "edit", "workflow_version": 3, "model": "test-model"}


def retitle(old: str, new: str | None) -> dict[str, Any]:
    graph = json.loads(json.dumps(GRAPH))
    for node in graph.values():
        if node["_meta"]["title"] == old:
            node["_meta"]["title"] = new
    return graph


@pytest.mark.parametrize(
    ("graph", "registry", "message"),
    [
        (retitle("IN_SEED", "seed"), REGISTRY, "'IN_SEED' не найден"),
        (retitle("IN_STEPS", "IN_SEED"), REGISTRY, "'IN_SEED' найден 2 раза"),
        (retitle("OUT_IMAGE", None), REGISTRY, "'OUT_IMAGE' не найден"),
        (GRAPH, REGISTRY.replace("field: value", "field: steps"), "нет поля 'steps'"),
        (
            GRAPH,
            REGISTRY.replace("{node: IN_STEPS,  field: value", "{node: IN_SEED, field: steps"),
            "подключено к другому узлу",
        ),
        (GRAPH, REGISTRY.replace("edit_local", "upscale_local"), "нет workflow для edit_local"),
        (GRAPH, REGISTRY.replace("job_type: edit_local", "job_type: nope"), "неверно поле"),
        ({"nodes": [], "links": []}, REGISTRY, "API-формат"),
    ],
)
def test_bad_bindings_stop_startup(
    tmp_path: Path, graph: dict[str, Any], registry: str, message: str
) -> None:
    with pytest.raises(WorkflowError, match=message):
        load_registry(make(tmp_path, graph, registry))


def test_one_workflow_per_job_type(tmp_path: Path) -> None:
    doubled = REGISTRY + REGISTRY.replace("edit:", "edit2:", 1)
    with pytest.raises(WorkflowError, match="уже занят"):
        load_registry(make(tmp_path, registry=doubled))


def test_repository_upscale_workflow() -> None:
    registry = load_registry(DEFAULT_WORKFLOWS_DIR, REQUIRED_TYPES)
    wf = registry.for_type(JobType.UPSCALE_LOCAL)
    assert wf is not None and wf.name == "upscale_seedvr2"
    assert wf.choices("model") == {
        "7b_fp16": "seedvr2_7b_fp16.safetensors",
        "3b_fp16": "seedvr2_3b_fp16.safetensors",
    }
    graph = wf.build({"model": "3b_fp16", "scale": 1.59})
    by_title = {n["_meta"]["title"]: n for n in graph.values()}
    assert by_title["IN_UPSCALE_MODEL"]["inputs"]["unet_name"] == "seedvr2_3b_fp16.safetensors"
    assert by_title["IN_UPSCALE"]["inputs"]["resize_type.multiplier"] == 1.59
    defaults = {n["_meta"]["title"]: n for n in wf.build({}).values()}
    assert defaults["IN_UPSCALE_MODEL"]["inputs"]["unet_name"] == "seedvr2_7b_fp16.safetensors"
    assert wf.agent_bindings()["seed"] == {"node": "IN_SEED", "field": "seed"}


def test_registry_without_upscale_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="upscale_local"):
        load_registry(make(tmp_path), REQUIRED_TYPES)


@pytest.mark.parametrize(
    ("choices", "error"),
    [("{a: x.png}", "default 8 не из choices"), ("[x.png]", "choices — непустой словарь")],
)
def test_choices_validation(tmp_path: Path, choices: str, error: str) -> None:
    registry = REGISTRY.replace(
        "steps:  {node: IN_STEPS,  field: value, default: 8}",
        f"steps:  {{node: IN_STEPS,  field: value, default: 8, choices: {choices}}}",
    )
    with pytest.raises(WorkflowError, match=error):
        load_registry(make(tmp_path, registry=registry))
