"""Клиент ComfyUI агента против подменённого HTTP API (respx)."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from restore_agent.comfy import Cancelled, ComfyClient, ComfyError, missing_models
from restore_agent.config import ConfigError, load

C = "http://127.0.0.1:8188"
WORKFLOW = {
    "1": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "qwen.safetensors"},
        "_meta": {"title": "Load"},
    },
    "2": {
        "class_type": "LoadImage",
        "inputs": {"image": "restore/source.png"},
        "_meta": {"title": "IN_IMAGE"},
    },
    "3": {
        "class_type": "SaveImage",
        "inputs": {"images": ["1", 0]},
        "_meta": {"title": "OUT_IMAGE"},
    },
}
OBJECT_INFO = {
    "UNETLoader": {"input": {"required": {"unet_name": [["qwen.safetensors", "other.gguf"]]}}},
    "LoadImage": {"input": {"required": {"image": [["restore/source.png"], {}]}}},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"]}}},
}


@pytest.fixture
def comfy() -> ComfyClient:
    return ComfyClient(httpx.Client(base_url=C), poll_s=0, tick_s=0)


def entry(
    status: str = "success", messages: list | None = None, outputs: dict | None = None
) -> dict:
    return {
        "status": {
            "status_str": status,
            "completed": status == "success",
            "messages": messages or [],
        },
        "outputs": outputs or {},
    }


@respx.mock
def test_system_stats(comfy: ComfyClient) -> None:
    respx.get(f"{C}/system_stats").respond(
        json={
            "devices": [
                {
                    "name": "cuda:0 NVIDIA GeForce RTX 3090 Ti : cudaMallocAsync",
                    "vram_free": 23 * 2**30,
                }
            ]
        }
    )
    stats = comfy.system_stats()
    assert stats is not None
    assert stats.gpu_name == "NVIDIA GeForce RTX 3090 Ti"
    assert stats.vram_free_mb == 23 * 1024


@respx.mock
def test_system_stats_when_comfy_is_down(comfy: ComfyClient) -> None:
    respx.get(f"{C}/system_stats").mock(side_effect=httpx.ConnectError("refused"))
    assert comfy.system_stats() is None


@respx.mock
def test_upload_into_restore_subfolder(comfy: ComfyClient) -> None:
    route = respx.post(f"{C}/upload/image").respond(
        json={"name": "source.png", "subfolder": "restore", "type": "input"}
    )
    assert comfy.upload_image("source.png", b"png") == "restore/source.png"
    body = route.calls[0].request.content
    assert b'name="subfolder"\r\n\r\nrestore' in body
    assert b'name="overwrite"\r\n\r\ntrue' in body


@respx.mock
def test_run_waits_for_history_and_fetches_output(comfy: ComfyClient) -> None:
    prompt = respx.post(f"{C}/prompt").respond(json={"prompt_id": "p1"})
    respx.get(f"{C}/history/p1").mock(
        side_effect=[
            httpx.Response(200, json={}),
            httpx.Response(
                200,
                json={
                    "p1": entry(
                        outputs={
                            "3": {
                                "images": [
                                    {
                                        "filename": "out_00001_.png",
                                        "subfolder": "restore",
                                        "type": "output",
                                    }
                                ]
                            }
                        }
                    )
                },
            ),
        ]
    )
    view = respx.get(f"{C}/view").respond(content=b"PNGDATA")
    ticks: list[int] = []
    assert comfy.run(WORKFLOW, "3", on_tick=lambda: ticks.append(1) and False) == b"PNGDATA"
    assert json.loads(prompt.calls[0].request.content)["prompt"] == WORKFLOW
    assert view.calls[0].request.url.params["filename"] == "out_00001_.png"
    assert ticks  # прогресс отправлялся, пока ждали


@respx.mock
def test_rejected_graph_is_not_retryable(comfy: ComfyClient) -> None:
    respx.post(f"{C}/prompt").respond(
        400,
        json={
            "error": {"message": "Prompt outputs failed validation"},
            "node_errors": {
                "1": {
                    "class_type": "UNETLoader",
                    "errors": [
                        {"message": "Value not in list", "details": "unet_name: 'x' not in [...]"}
                    ],
                }
            },
        },
    )
    with pytest.raises(ComfyError) as e:
        comfy.run(WORKFLOW, "3")
    assert not e.value.retryable
    assert "UNETLoader: Value not in list" in str(e.value)


@pytest.mark.parametrize(
    ("exception_type", "message", "retryable"),
    [
        ("torch.OutOfMemoryError", "Allocation on device", False),
        ("RuntimeError", "CUDA out of memory. Tried to allocate 2.00 GiB", False),
        ("ValueError", "something odd", True),
    ],
)
@respx.mock
def test_execution_errors(
    comfy: ComfyClient, exception_type: str, message: str, retryable: bool
) -> None:
    respx.post(f"{C}/prompt").respond(json={"prompt_id": "p1"})
    respx.get(f"{C}/history/p1").respond(
        json={
            "p1": entry(
                "error",
                messages=[
                    ["execution_start", {}],
                    [
                        "execution_error",
                        {
                            "node_type": "KSampler",
                            "exception_type": exception_type,
                            "exception_message": message,
                        },
                    ],
                ],
            )
        }
    )
    with pytest.raises(ComfyError) as e:
        comfy.run(WORKFLOW, "3")
    assert e.value.retryable is retryable
    assert "KSampler" in str(e.value) and message in str(e.value)


@respx.mock
def test_cancel_interrupts_comfy(comfy: ComfyClient) -> None:
    respx.post(f"{C}/prompt").respond(json={"prompt_id": "p1"})
    respx.get(f"{C}/history/p1").respond(json={})
    interrupt = respx.post(f"{C}/interrupt").respond(200)
    with pytest.raises(Cancelled):
        comfy.run(WORKFLOW, "3", on_tick=lambda: True)
    assert interrupt.called


@respx.mock
def test_lost_prompt_is_retryable_error(comfy: ComfyClient) -> None:
    respx.post(f"{C}/prompt").respond(json={"prompt_id": "p1"})
    respx.get(f"{C}/history/p1").respond(json={})
    respx.get(f"{C}/queue").respond(json={"queue_running": [], "queue_pending": []})
    with pytest.raises(ComfyError, match="потерял") as e:
        comfy.run(WORKFLOW, "3")
    assert e.value.retryable


@respx.mock
def test_free_unloads_models(comfy: ComfyClient) -> None:
    route = respx.post(f"{C}/free").respond(200)
    comfy.free()
    assert json.loads(route.calls[0].request.content) == {
        "unload_models": True,
        "free_memory": True,
    }


def test_missing_model_files_are_reported() -> None:
    assert missing_models(WORKFLOW, OBJECT_INFO) == []
    wf = json.loads(json.dumps(WORKFLOW))
    wf["1"]["inputs"]["unet_name"] = "qwen_image_edit_2511_fp8mixed.safetensors"
    [problem] = missing_models(wf, OBJECT_INFO)
    assert problem.startswith("нет файла qwen_image_edit_2511_fp8mixed.safetensors в models/")
    # новый формат COMBO из ComfyUI
    info = {
        **OBJECT_INFO,
        "UNETLoader": {
            "input": {"required": {"unet_name": ["COMBO", {"options": ["qwen.safetensors"]}]}}
        },
    }
    assert len(missing_models(wf, info)) == 1
    wf["9"] = {"class_type": "SeedVR2Loader", "inputs": {}, "_meta": {"title": "X"}}
    assert any("нет узла SeedVR2Loader" in p for p in missing_models(wf, info))


@respx.mock
def test_check_models_fails_without_retry(comfy: ComfyClient) -> None:
    respx.get(f"{C}/object_info").respond(
        json={**OBJECT_INFO, "UNETLoader": {"input": {"required": {"unet_name": [["other.gguf"]]}}}}
    )
    with pytest.raises(ComfyError, match=r"нет файла qwen\.safetensors") as e:
        comfy.check_models(WORKFLOW)
    assert not e.value.retryable


def test_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'server_url = "https://restore.example.org/"\ntoken = "t"\n'
        'agent_id = "pc"\nworkdir = "' + (tmp_path / "w").as_posix() + '"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("FAKE_COMFY", "1")
    cfg = load(path)
    assert cfg.server_url == "https://restore.example.org"
    assert cfg.comfy_url == "http://127.0.0.1:8188"
    assert cfg.workdir == tmp_path / "w"
    assert cfg.fake_comfy
    path.write_text('server_url = "x"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="token, agent_id"):
        load(path)
