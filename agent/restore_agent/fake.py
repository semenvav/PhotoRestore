"""FAKE_COMFY=1: ComfyUI без GPU для сквозных тестов — возвращает вход с напечатанным seed
(для графа апскейла — увеличенный на его множитель)."""

import io
import threading
from collections.abc import Callable
from typing import Any

from PIL import Image, ImageDraw

from restore_agent.comfy import Cancelled, Stats


class FakeComfy:
    def __init__(self) -> None:
        self.inputs: dict[str, bytes] = {}
        self.runs: list[dict[str, Any]] = []
        self.frees = 0
        self.interrupts = 0
        # для тестов: вызывается перед выполнением графа (например, чтобы отменить задачу)
        self.before_run: Callable[[], None] | None = None
        self._lock = threading.Lock()

    def system_stats(self) -> Stats | None:
        return Stats(gpu_name="FAKE_COMFY", vram_free_mb=24000)

    def check_models(self, workflow: dict[str, Any]) -> None:
        return None

    def upload_image(self, name: str, data: bytes) -> str:
        key = f"restore/{name}"
        with self._lock:
            self.inputs[key] = data
        return key

    def run(
        self,
        workflow: dict[str, Any],
        output_node: str,
        on_tick: Callable[[], bool] | None = None,
    ) -> bytes:
        if self.before_run:
            self.before_run()
        if on_tick and on_tick():
            self.interrupt()
            raise Cancelled
        self.runs.append(workflow)
        source = next(
            self.inputs[v]
            for node in workflow.values()
            for v in node.get("inputs", {}).values()
            if isinstance(v, str) and v in self.inputs
        )
        seed = next(
            (
                node["inputs"]["seed"]
                for node in workflow.values()
                if isinstance(node.get("inputs", {}).get("seed"), int)
            ),
            None,
        )
        img = Image.open(io.BytesIO(source)).convert("RGB")
        scale = next(
            (
                float(node["inputs"]["resize_type.multiplier"])
                for node in workflow.values()
                if "resize_type.multiplier" in node.get("inputs", {})
            ),
            1.0,
        )
        if scale != 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width, 14), fill=(0, 0, 0))
        draw.text((2, 1), f"seed {seed}", fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def interrupt(self) -> None:
        self.interrupts += 1

    def free(self) -> None:
        self.frees += 1
