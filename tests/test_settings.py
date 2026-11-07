"""M4: страница настроек (edit, апскейл, Gemini, цены, пресеты) и ручная правка."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from imgutil import EXIF_ORIENTATION, encode, gradient
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from test_final import TOKEN, in_review, last_job, msg, photo, queue_photo, variants

from app import presets
from app.config import Settings
from app.intake import Intake
from app.models import Event, Preset, PresetKind, Status, VariantSource


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.agent_token = SecretStr(TOKEN)
    return settings


def stored(factory: sessionmaker[Session], key: str) -> object:
    with factory() as s:
        return presets.get_setting(s, key)


GEMINI_FORM = {
    "model": "gemini-3-pro-image",
    "image_size": "2K",
    "monthly_budget_usd": "20",
    "max_concurrency": "2",
}


def test_settings_page_shows_current_values(client: TestClient) -> None:
    page = client.get("/settings").text
    for anchor in ('id="edit"', 'id="upscale"', 'id="gemini"', 'id="prices"', 'id="presets"'):
        assert anchor in page
    assert 'name="variants" min="1" max="6" step="1" value="3"' in page
    assert '<option value="7b_fp16" selected>SeedVR2 7B fp16</option>' in page
    assert '<option value="3b_fp16">SeedVR2 3B fp16</option>' in page
    assert 'value="gemini-3-pro-image" required' in page
    assert "нет GEMINI_API_KEY — отправка недоступна" in page
    assert "<code>gemini-3.1-flash-image</code>" in page
    assert "Restore this old photograph." in page
    assert 'value="Сохранить зерно"' in page
    assert 'href="/settings" aria-current="page">Настройки' in page


def test_upscale_defaults_drive_new_upscale_jobs(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    r = client.post(
        "/settings/upscale",
        data={"model": "3b_fp16", "target_mp": "2,5", "max_scale": "1.5"},
        follow_redirects=False,
    )
    assert msg(r) == "/settings?msg=Сохранено#upscale"
    assert stored(factory, "upscale") == {"model": "3b_fp16", "target_mp": 2.5, "max_scale": 1.5}

    pid = in_review(client, intake, inbox, "a.png")  # варианты 1680×944
    client.post(f"/review/{pid}/accept", data={"variant_id": variants(factory, pid)[0].id})
    job = last_job(factory, pid)
    assert job.payload["params"] == {"model": "3b_fp16", "scale": 1.26}  # √(2,5 / 1,586)
    page = client.get("/settings").text
    assert '<option value="3b_fp16" selected>' in page


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({"model": "13b", "target_mp": "4", "max_scale": "2"}, "нет модели '13b' в workflow"),
        ({"model": "7b_fp16", "target_mp": "0.5", "max_scale": "2"}, "цель, Мп: не меньше 1.0"),
        (
            {"model": "7b_fp16", "target_mp": "4", "max_scale": "5"},
            "предел масштаба: не больше 4.0",
        ),
        ({"model": "7b_fp16", "target_mp": "много", "max_scale": "2"}, "цель, Мп: нужно число"),
    ],
)
def test_upscale_settings_validation(
    client: TestClient, factory: sessionmaker[Session], data: dict[str, str], error: str
) -> None:
    r = client.post("/settings/upscale", data=data, follow_redirects=False)
    assert msg(r).startswith("/settings?msg=Не сохранено") and error in msg(r)
    assert stored(factory, "upscale") == presets.UpscalePrefs().model_dump()


def test_edit_defaults_drive_new_edit_jobs(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    r = client.post(
        "/settings/edit",
        data={"variants": "4", "steps": "6", "megapixels": "1,2"},
        follow_redirects=False,
    )
    assert msg(r).endswith("msg=Сохранено#edit")
    pid = queue_photo(client, intake, inbox, "a.png")
    job = last_job(factory, pid)
    assert len(job.payload["seeds"]) == 4
    assert job.payload["params"] == {"steps": 6, "megapixels": 1.2}
    r = client.post(
        "/settings/edit",
        data={"variants": "9", "steps": "6", "megapixels": "1"},
        follow_redirects=False,
    )
    assert "вариантов: не больше 6" in msg(r)


def test_gemini_settings_and_missing_prices_warning(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    r = client.post(
        "/settings/gemini",
        data={
            **GEMINI_FORM,
            "model": "gemini-3.1-flash-image",
            "image_size": "4K",
            "monthly_budget_usd": "7,5",
            "max_concurrency": "1",
        },
        follow_redirects=False,
    )
    assert msg(r).endswith("msg=Сохранено#gemini")
    assert stored(factory, "gemini") == {
        "model": "gemini-3.1-flash-image",
        "image_size": "4K",
        "monthly_budget_usd": 7.5,
        "max_concurrency": 1,
    }
    r = client.post(
        "/settings/gemini", data={**GEMINI_FORM, "model": "gemini-9-image"}, follow_redirects=False
    )
    assert "Задайте цены для gemini-9-image — без них отправка заблокирована" in msg(r)
    page = client.get("/settings").text
    assert "Для gemini-9-image нет цен" in page
    assert '<code>gemini-9-image</code> <span class="chip small">выбрана</span>' in page

    for bad, error in [
        ({"model": "Gemini 3"}, "модель: недопустимые символы"),
        ({"image_size": "8K"}, "размер: недопустимое значение"),
        ({"max_concurrency": "5"}, "одновременных запросов: не больше 2"),
        ({"monthly_budget_usd": "-1"}, "бюджет: не меньше 0"),
    ]:
        r = client.post("/settings/gemini", data={**GEMINI_FORM, **bad}, follow_redirects=False)
        assert error in msg(r), bad


def test_prices_edit_add_delete(client: TestClient, factory: sessionmaker[Session]) -> None:
    rows = {
        "model": ["gemini-3-pro-image", "gemini-3.1-flash-image", "gemini-nano-banana-2.1"],
        "input": ["2", "0,5", "1.5"],
        "output_text": ["12", "3", "7.5"],
        "output_image": ["130", "60", "30"],
    }
    r = client.post(
        "/settings/prices",
        data={
            **rows,
            "delete": ["gemini-nano-banana-2.1"],
            "new_model": "gemini-4-image",
            "new_input": "1",
            "new_output_text": "2",
            "new_output_image": "3",
        },
        follow_redirects=False,
    )
    assert msg(r).endswith("msg=Цены сохранены#prices")
    assert stored(factory, "gemini_prices") == {
        "gemini-3-pro-image": {"input": 2.0, "output_text": 12.0, "output_image": 130.0},
        "gemini-3.1-flash-image": {"input": 0.5, "output_text": 3.0, "output_image": 60.0},
        "gemini-4-image": {"input": 1.0, "output_text": 2.0, "output_image": 3.0},
    }

    r = client.post(
        "/settings/prices", data={**rows, "delete": ["gemini-3-pro-image"]}, follow_redirects=False
    )
    assert "gemini-3-pro-image выбрана в настройках Gemini" in msg(r)
    r = client.post(
        "/settings/prices",
        data={
            **rows,
            "new_model": "bad name",
            "new_input": "1",
            "new_output_text": "1",
            "new_output_image": "1",
        },
        follow_redirects=False,
    )
    assert "недопустимое имя модели 'bad name'" in msg(r)
    r = client.post(
        "/settings/prices", data={**rows, "output_image": ["", "60", "30"]}, follow_redirects=False
    )
    assert "картинка: нужно число (gemini-3-pro-image)" in msg(r)
    assert "gemini-4-image" in stored(factory, "gemini_prices")  # type: ignore[operator]


def test_base_prompt_edit_and_reset(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    r = client.post(
        "/settings/presets/base", data={"text": "Clean it gently."}, follow_redirects=False
    )
    assert msg(r).endswith("Базовый промпт сохранён#presets")
    pid = queue_photo(client, intake, inbox, "a.png")
    assert last_job(factory, pid).payload["prompt"] == "Clean it gently."

    r = client.post("/settings/presets/base", data={"text": "   "}, follow_redirects=False)
    assert "текст: не заполнено" in msg(r)
    client.post("/settings/presets/base", data={"reset": "true"})
    with factory() as s:
        assert presets.base_prompt(s) == presets.BASE_PROMPT


def test_addons_crud_and_order(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    def addons() -> list[tuple[int, str, str]]:
        with factory() as s:
            return [(p.id, p.name, p.text) for p in presets.addons(s)]

    first, second = addons()[0], addons()[1]
    r = client.post(
        "/settings/presets/addons",
        data={"name": "Небо", "text": "Clean up the sky."},
        follow_redirects=False,
    )
    assert msg(r).endswith("Добавка «Небо» добавлена#presets")
    new_id = addons()[-1][0]
    assert addons()[-1][1:] == ("Небо", "Clean up the sky.")

    client.post(f"/settings/presets/{new_id}", data={"name": "Небо+", "text": "Sky."})
    assert addons()[-1][1:] == ("Небо+", "Sky.")
    client.post(f"/settings/presets/{second[0]}/up")
    assert [a[0] for a in addons()][:2] == [second[0], first[0]]
    client.post(f"/settings/presets/{second[0]}/up")  # уже первая — без изменений
    assert addons()[0][0] == second[0]
    client.post(f"/settings/presets/{first[0]}/delete")
    assert first[0] not in [a[0] for a in addons()]

    # базовый пресет через маршруты добавок не меняется и не удаляется
    with factory() as s:
        base_id = s.scalar(select(Preset.id).where(Preset.kind == PresetKind.BASE))
    r = client.post(f"/settings/presets/{base_id}/delete", follow_redirects=False)
    assert "Добавка не найдена" in msg(r)

    # новая добавка — чип на проверке, и её текст уходит в промпт
    pid = in_review(client, intake, inbox, "a.png")
    page = client.get(f"/review/{pid}").text
    assert f'name="addons" value="{new_id}"> Небо+' in page
    client.post(f"/review/{pid}/refine", data={"addons": [str(new_id)], "text": "More."})
    assert last_job(factory, pid).payload["prompt"].endswith("\nSky.\nMore.")


# --- ручная правка -----------------------------------------------------------------------


def jpeg_rotated(width: int, height: int) -> bytes:
    """JPEG с EXIF-ориентацией 6: на экране — повёрнутый на 90°."""
    img = gradient(width, height, seed=3, noise=False)
    exif = img.getexif()
    exif[EXIF_ORIENTATION] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, exif=exif)
    return buf.getvalue()


def test_manual_edit_becomes_variant_of_selected(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    pid = in_review(client, intake, inbox, "a.png")
    parent = variants(factory, pid)[1]
    r = client.post(
        f"/review/{pid}/manual",
        data={"variant_id": str(parent.id)},
        files={"file": ("правка.jpg", jpeg_rotated(300, 200), "image/jpeg")},
        follow_redirects=False,
    )
    v = variants(factory, pid)[-1]
    assert msg(r) == f"/review/{pid}?v={v.id}&msg=R-{pid:04d}: правка загружена как {v.label}"
    assert v.source == VariantSource.MANUAL and v.parent_id == parent.id
    assert (v.width, v.height) == (200, 300)  # EXIF применён
    assert v.params == {"file": "правка.jpg"} and v.round == photo(factory, pid).round
    path = settings.store_dir / f"R-{pid:04d}" / v.path
    with Image.open(path) as img:
        assert img.format == "PNG" and img.size == (200, 300)
    assert not path.stat().st_mode & 0o222
    assert photo(factory, pid).status == Status.REVIEW

    page = client.get(f"/review/{pid}?v={v.id}").text
    assert '<span class="badge src-manual">Вручную</span>' in page
    assert "файл правка.jpg" in page
    card = client.get(f"/photo/{pid}").text
    assert f"Проверка → Проверка · загружена правка {v.label} (от {parent.label})" in card
    with factory() as s:
        ev = s.scalars(select(Event).where(Event.photo_id == pid).order_by(Event.id.desc())).first()
        assert ev is not None and ev.data["manual"] == v.label


def test_manual_edit_rejected(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    pid = in_review(client, intake, inbox, "a.png")
    r = client.post(
        f"/review/{pid}/manual",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
        follow_redirects=False,
    )
    assert "Файл не принят" in msg(r)
    gif = io.BytesIO()
    gradient(20, 20).save(gif, format="GIF")
    r = client.post(
        f"/review/{pid}/manual", files={"file": ("a.gif", gif.getvalue())}, follow_redirects=False
    )
    assert "формат GIF не поддерживается" in msg(r)
    assert len(variants(factory, pid)) == 3

    queued = queue_photo(client, intake, inbox, "b.png")  # в очереди на ПК, не на проверке
    r = client.post(
        f"/review/{queued}/manual",
        files={"file": ("p.png", encode(gradient(20, 20), "PNG"))},
        follow_redirects=False,
    )
    assert "фото не на проверке" in msg(r)
    assert variants(factory, queued) == []
