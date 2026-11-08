"""M5: экран отца `/family`, роль по хосту, «Принять» и «Вернуть» с комментарием."""

import io
import json
from pathlib import Path
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient
from imgutil import gradient
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from test_final import TOKEN, in_review, msg, photo, variants

from app import auth
from app.api import family as family_api
from app.config import Settings
from app.intake import Intake
from app.main import create_app
from app.models import Actor, Event, Photo, Status

FAMILY_HOST = "family.test"
FAMILY = {"host": FAMILY_HOST}


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.agent_token = SecretStr(TOKEN)
    settings.family_host = f"{FAMILY_HOST}, other.family.test"
    return settings


def deliver(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    name: str,
    album: str = "",
) -> int:
    pid = in_review(client, intake, inbox, name, album)
    v = variants(factory, pid)[0]
    client.post(f"/review/{pid}/accept", data={"variant_id": v.id, "upscale": "false"})
    client.post(f"/final/{pid}/approve")
    assert photo(factory, pid).status == Status.DELIVERED
    return pid


def family_events(factory: sessionmaker[Session], pid: int) -> list[Event]:
    with factory() as s:
        return list(
            s.scalars(
                select(Event)
                .where(Event.photo_id == pid, Event.actor == Actor.FAMILY)
                .order_by(Event.id)
            )
        )


# --- роль по хосту ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Family.Example:443", "family.example"),
        ("family.example.", "family.example"),
        ("[::1]:8000", "[::1]"),
        ("10.0.0.5", "10.0.0.5"),
        (" ", ""),
    ],
)
def test_host_name(value: str, expected: str) -> None:
    assert auth.host_name(value) == expected


def test_parse_hosts_and_allowed_paths() -> None:
    assert auth.parse_hosts("a.example, B.example:80,,") == {"a.example", "b.example"}
    assert auth.parse_hosts("") == frozenset()
    assert auth.role_for("A.example:8000", {"a.example"}) is auth.Role.FAMILY
    assert auth.role_for("owner.example", {"a.example"}) is auth.Role.OWNER
    for path in ("/family", "/family/3", "/family/3/after.jpg", "/static/app.css", "/healthz"):
        assert auth.family_allowed(path), path
    for path in (
        "/",
        "/familyx",
        "/review/1",
        "/settings",
        "/api/agent/v1/hello",
        "/files/1/thumb",
    ):
        assert not auth.family_allowed(path), path


def test_family_host_sees_only_family(client: TestClient) -> None:
    """Критерий M5: пользователь «отец» видит только /family."""
    for path in ("/", "/review", "/review/1", "/final", "/pc", "/settings", "/photo/1", "/prep"):
        r = client.get(path, headers=FAMILY, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/family", path
    for path in ("/files/1/thumb", "/partials/header", "/partials/columns"):
        r = client.get(path, headers=FAMILY, follow_redirects=False)
        assert r.headers["location"] == "/family", path
    for path in ("/bulk", "/settings/edit", "/runs/start", "/photo/1/hold", "/api/agent/v1/hello"):
        assert client.post(path, headers=FAMILY).status_code == 403, path
    # второй хост из списка, порт и регистр не важны
    r = client.get("/settings", headers={"host": "Other.Family.Test:8443"}, follow_redirects=False)
    assert r.headers["location"] == "/family"

    assert client.get("/static/app.css", headers=FAMILY).status_code == 200
    assert client.get("/healthz", headers=FAMILY).json()["ok"] is True
    page = client.get("/family", headers=FAMILY).text
    assert "Готовые фотографии" in page
    assert 'href="/settings"' not in page and 'href="/pc"' not in page and "К доске" not in page

    # владелец — как раньше
    assert client.get("/settings").status_code == 200
    assert client.get("/", follow_redirects=False).status_code == 200


# --- список и страница фото -------------------------------------------------------------


def test_family_lists_only_delivered_and_accepted(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    on_review = in_review(client, intake, inbox, "На проверке.png")
    first = deliver(client, intake, inbox, factory, "Бабушка 1952.png", "Семья")
    second = deliver(client, intake, inbox, factory, "Дом.jpg")
    client.post(f"/family/{second}/accept", headers=FAMILY)

    page = client.get("/family", headers=FAMILY).text
    waiting, _, accepted = page.partition("Принятые")
    assert "Бабушка 1952" in waiting and "Семья" in waiting and f"/family/{first}" in waiting
    assert "Дом" in accepted and f"/family/{second}" in accepted
    assert "На проверке" not in page and f"/family/{on_review}" not in page
    # без технических деталей
    for word in ("R-0", "v00", "Gemini", "seed", "раунд", ".png"):
        assert word not in page, word

    r = client.get(f"/family/{on_review}", headers=FAMILY, follow_redirects=False)
    assert r.status_code == 303 and "Этого фото здесь уже нет" in msg(r)
    for name in ("before.jpg", "after.jpg", "after.png", "thumb.webp"):
        assert client.get(f"/family/{on_review}/{name}", headers=FAMILY).status_code == 404


def test_family_photo_page_shows_before_and_after(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    pid = deliver(client, intake, inbox, factory, "Бабушка 1952.png", "Семья")
    final_id = photo(factory, pid).final_variant_id
    page = client.get(f"/family/{pid}", headers=FAMILY).text
    assert "<h1>Бабушка 1952</h1>" in page
    assert f'action="/family/{pid}/accept"' in page and f'action="/family/{pid}/return"' in page
    assert "disabled" not in page.split('class="family-answer"')[1].split('class="viewer')[0]
    config = json.loads(page.split('id="review-config">')[1].split("</script>")[0])
    assert [i["label"] for i in config["images"]] == ["Было", "Стало"]
    assert config["right"] == "after"
    assert f"/family/{pid}/after.png?v={final_id}" in page
    for word in ("R-0", "v00", "seed", "Gemini"):
        assert word not in page, word

    before = client.get(f"/family/{pid}/before.jpg", headers=FAMILY)
    assert before.headers["content-type"] == "image/jpeg"
    after = client.get(f"/family/{pid}/after.jpg", headers=FAMILY)
    assert after.status_code == 200 and after.headers["content-type"] == "image/jpeg"
    assert client.get(f"/family/{pid}/before.png", headers=FAMILY).status_code == 200
    assert client.get(f"/family/{pid}/thumb.webp", headers=FAMILY).status_code == 200
    full = client.get(f"/family/{pid}/after.png?download=1", headers=FAMILY)
    assert unquote(full.headers["content-disposition"]).endswith("Бабушка 1952_restored.png")
    with Image.open(io.BytesIO(full.content)) as img:
        assert img.format == "PNG"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("портрет_у_окна.jpg", "портрет у окна"), ("Дом  1950.png", "Дом 1950"), ("___.png", "___")],
)
def test_title_for_family(name: str, expected: str) -> None:
    assert family_api.title(Photo(original_name=name)) == expected


def test_owner_sees_preview_without_answer(
    client: TestClient, intake: Intake, inbox: Path, factory: sessionmaker[Session]
) -> None:
    pid = deliver(client, intake, inbox, factory, "a.png")
    page = client.get(f"/family/{pid}").text
    assert "Так этот экран видит отец" in page and "family.test, other.family.test" in page
    answer = page.split('class="family-answer"')[1].split('class="viewer')[0]
    assert answer.count("disabled") == 3
    assert "Экран отца" in client.get(f"/photo/{pid}").text

    for action, data in (("accept", {}), ("return", {"comment": "переделать"})):
        r = client.post(f"/family/{pid}/{action}", data=data, follow_redirects=False)
        assert "только отец" in msg(r)
    assert photo(factory, pid).status == Status.DELIVERED
    assert family_events(factory, pid) == []


def test_without_family_host_nobody_can_answer(settings: Settings, client: TestClient) -> None:
    settings.family_host = ""
    with TestClient(create_app(settings)) as c:
        page = c.get("/family", headers=FAMILY).text
        assert "не настроен" in page and "FAMILY_HOST" in page
        assert c.get("/settings", headers=FAMILY).status_code == 200


# --- «Принять» ----------------------------------------------------------------------------


def test_accept_by_family(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    first = deliver(client, intake, inbox, factory, "first.png")
    second = deliver(client, intake, inbox, factory, "second.png")
    r = client.post(f"/family/{first}/accept", headers=FAMILY, follow_redirects=False)
    assert r.headers["location"].startswith(f"/family/{second}?")
    assert "Спасибо! «first» принято." in msg(r)
    p = photo(factory, first)
    assert p.status == Status.ACCEPTED
    assert (settings.delivered_dir / "first_restored.png").is_file()  # файл остаётся
    [ev] = family_events(factory, first)
    assert ev.data == {"from": "DELIVERED", "to": "ACCEPTED", "variant": "v001"}

    # повторное нажатие (двойной щелчок) — без ошибки
    r = client.post(f"/family/{first}/accept", headers=FAMILY, follow_redirects=False)
    assert "уже принято" in msg(r)
    page = client.get(f"/family/{first}", headers=FAMILY).text
    assert "Вы приняли это фото" in page and f'action="/family/{first}/accept"' not in page

    r = client.post(f"/family/{second}/accept", headers=FAMILY, follow_redirects=False)
    assert r.headers["location"].startswith("/family?")  # ждущих больше нет


# --- «Вернуть» ----------------------------------------------------------------------------


def test_return_requires_comment(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    pid = deliver(client, intake, inbox, factory, "a.png")
    for comment in ("", "   \n  "):
        r = client.post(
            f"/family/{pid}/return",
            data={"comment": comment},
            headers=FAMILY,
            follow_redirects=False,
        )
        assert r.headers["location"].startswith(f"/family/{pid}?return=1")
        assert "Напишите, что нужно исправить" in msg(r)
    r = client.post(
        f"/family/{pid}/return",
        data={"comment": "x" * 2001},
        headers=FAMILY,
        follow_redirects=False,
    )
    assert "не больше 2000" in msg(r)
    assert photo(factory, pid).status == Status.DELIVERED
    assert (settings.delivered_dir / "a_restored.png").is_file()
    # форма возврата открыта сразу
    assert "back: true" in client.get(f"/family/{pid}?return=1", headers=FAMILY).text


def test_return_moves_photo_to_review_with_comment(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    """Критерий M5: возврат переводит фото в REVIEW с комментарием."""
    pid = deliver(client, intake, inbox, factory, "Бабушка 1952.png", "Семья")
    out = settings.delivered_dir / "Семья" / "Бабушка 1952_restored.png"
    assert out.is_file()
    before = photo(factory, pid)
    comment = "Лицо слева не похоже.\nВерните морщины."

    r = client.post(
        f"/family/{pid}/return",
        data={"comment": f"  {comment}  "},
        headers=FAMILY,
        follow_redirects=False,
    )
    assert r.headers["location"].startswith("/family?")
    assert "«Бабушка 1952» вернули на доработку" in msg(r)

    p = photo(factory, pid)
    assert p.status == Status.REVIEW and p.round == before.round + 1
    assert p.family_comment == comment
    assert p.delivered_path is None and p.final_variant_id is None
    assert p.accepted_variant_id is None
    assert not out.exists()  # копия из «Готово» удалена
    assert len(variants(factory, pid)) == 3  # финал остаётся в store/
    assert all(Path(settings.store_dir / p.code / v.path).is_file() for v in variants(factory, pid))
    [ev] = family_events(factory, pid)
    assert ev.data == {
        "from": "DELIVERED",
        "to": "REVIEW",
        "comment": comment,
        "variant": "v001",
        "delivered_removed": "Семья/Бабушка 1952_restored.png",
    }

    review = client.get(f"/review/{pid}").text
    assert "Отец вернул v001" in review and "Лицо слева не похоже." in review
    assert "↩ Отец: Лицо слева не похоже." in client.get("/").text
    card = client.get(f"/photo/{pid}").text
    assert "вернул v001: «Лицо слева не похоже." in card and "копия убрана из «Готово»" in card
    # у отца фото пропало
    assert "Бабушка 1952" not in client.get("/family", headers=FAMILY).text
    r = client.post(
        f"/family/{pid}/return", data={"comment": "ещё"}, headers=FAMILY, follow_redirects=False
    )
    assert r.status_code == 404


def test_return_keeps_foreign_or_missing_file(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    replaced = deliver(client, intake, inbox, factory, "replaced.png")
    gone = deliver(client, intake, inbox, factory, "gone.png")
    foreign = settings.delivered_dir / "replaced_restored.png"
    gradient(50, 50).save(foreign, format="PNG")  # отец пересохранил — метаданных нет
    (settings.delivered_dir / "gone_restored.png").unlink()

    for pid in (replaced, gone):
        client.post(f"/family/{pid}/return", data={"comment": "ещё раз"}, headers=FAMILY)
        assert photo(factory, pid).status == Status.REVIEW
    assert foreign.is_file()
    [ev] = family_events(factory, replaced)
    assert ev.data["delivered_kept"] == "replaced_restored.png"
    assert ev.data["reason"] == "файл заменён"
    [ev] = family_events(factory, gone)
    assert ev.data["delivered_missing"] == "gone_restored.png"
    card = client.get(f"/photo/{replaced}").text
    assert "файл во «Готово» оставлен: файл заменён" in card
    assert "в «Готово» файла уже не было" in client.get(f"/photo/{gone}").text


def test_redelivery_after_return(
    client: TestClient,
    intake: Intake,
    inbox: Path,
    factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    pid = deliver(client, intake, inbox, factory, "a.png")
    client.post(f"/family/{pid}/return", data={"comment": "темновато"}, headers=FAMILY)
    v2 = variants(factory, pid)[1]
    client.post(f"/review/{pid}/accept", data={"variant_id": v2.id, "upscale": "false"})
    client.post(f"/final/{pid}/approve")
    p = photo(factory, pid)
    assert p.status == Status.DELIVERED and p.delivered_path == "a_restored.png"
    assert "↩ Отец" not in client.get("/").text  # отдано снова — метка снята

    listing = client.get("/family", headers=FAMILY).text
    assert "Переделано по вашей просьбе" in listing
    page = client.get(f"/family/{pid}", headers=FAMILY).text
    assert "Вы просили: «темновато»" in page
    assert f"after.png?v={v2.id}" in page

    client.post(f"/family/{pid}/accept", headers=FAMILY)
    assert photo(factory, pid).status == Status.ACCEPTED
    assert "Отец вернул" not in client.get(f"/review/{pid}").text
    assert "Вы просили" not in client.get(f"/family/{pid}", headers=FAMILY).text
    assert (settings.delivered_dir / "a_restored.png").is_file()
