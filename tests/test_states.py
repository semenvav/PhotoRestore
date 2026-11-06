import itertools
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Actor, Event, Photo, Status
from app.states import TRANSITIONS, TransitionError, can_transition, transition

S, A = Status, Actor

# Таблица переходов SPEC §3, выписанная независимо от states.py
SPEC_TABLE: set[tuple[Status | None, Status, Actor]] = {
    (None, S.PREP, A.SYSTEM),
    (None, S.DUPLICATE, A.SYSTEM),
    (S.PREP, S.QUEUED_EDIT, A.OWNER),
    (S.QUEUED_EDIT, S.EDITING, A.AGENT),
    (S.EDITING, S.REVIEW, A.AGENT),
    (S.EDITING, S.QUEUED_EDIT, A.SYSTEM),
    (S.REVIEW, S.QUEUED_EDIT, A.OWNER),
    (S.REVIEW, S.GEMINI, A.OWNER),
    (S.GEMINI, S.REVIEW, A.SYSTEM),
    (S.REVIEW, S.REVIEW, A.OWNER),
    (S.REVIEW, S.QUEUED_UPSCALE, A.OWNER),
    (S.REVIEW, S.FINAL_REVIEW, A.OWNER),
    (S.QUEUED_UPSCALE, S.UPSCALING, A.AGENT),
    (S.UPSCALING, S.FINAL_REVIEW, A.AGENT),
    (S.UPSCALING, S.QUEUED_UPSCALE, A.SYSTEM),
    (S.FINAL_REVIEW, S.DELIVERED, A.OWNER),
    (S.FINAL_REVIEW, S.QUEUED_UPSCALE, A.OWNER),
    (S.FINAL_REVIEW, S.REVIEW, A.OWNER),
    (S.DELIVERED, S.ACCEPTED, A.FAMILY),
    (S.DELIVERED, S.REVIEW, A.FAMILY),
    # «Отложить»: любой, кроме DELIVERED, ACCEPTED, EDITING, UPSCALING, GEMINI
    (S.PREP, S.ON_HOLD, A.OWNER),
    (S.QUEUED_EDIT, S.ON_HOLD, A.OWNER),
    (S.REVIEW, S.ON_HOLD, A.OWNER),
    (S.QUEUED_UPSCALE, S.ON_HOLD, A.OWNER),
    (S.FINAL_REVIEW, S.ON_HOLD, A.OWNER),
    (S.DUPLICATE, S.ON_HOLD, A.OWNER),
    (S.ON_HOLD, S.PREP, A.OWNER),
    (S.DUPLICATE, S.PREP, A.OWNER),
}

ALL_SOURCES: list[Status | None] = [None, *Status]
ALL_COMBOS = list(itertools.product(ALL_SOURCES, list(Status), list(Actor)))


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def make_photo(session: Session, status: Status | None) -> Photo:
    photo = Photo(original_name="x.jpg", album="", sha256="0" * 64, width=10, height=10)
    if status is not None:
        photo.status = status  # прямая установка — только как подготовка теста
        session.add(photo)
        session.flush()
    return photo


def test_table_matches_spec() -> None:
    implemented = {(src, dst, a) for (src, dst), actors in TRANSITIONS.items() for a in actors}
    assert implemented == SPEC_TABLE


@pytest.mark.parametrize(("src", "dst", "actor"), sorted(SPEC_TABLE, key=str))
def test_allowed_transition(
    session: Session, src: Status | None, dst: Status, actor: Actor
) -> None:
    photo = make_photo(session, src)
    event = transition(session, photo, dst, actor, {"note": "t"})
    session.flush()
    assert photo.status == dst
    assert event.photo_id == photo.id
    assert event.kind == "status"
    assert event.actor == actor
    assert event.data == {"from": src.value if src else None, "to": dst.value, "note": "t"}


@pytest.mark.parametrize(
    ("src", "dst", "actor"),
    [c for c in ALL_COMBOS if c not in SPEC_TABLE],
    ids=str,
)
def test_forbidden_transition(
    session: Session, src: Status | None, dst: Status, actor: Actor
) -> None:
    photo = make_photo(session, src)
    events_before = session.scalars(select(Event)).all()
    assert not can_transition(src, dst, actor)
    with pytest.raises(TransitionError):
        transition(session, photo, dst, actor)
    if src is not None:
        assert photo.status == src
    assert session.scalars(select(Event)).all() == events_before


@pytest.mark.parametrize(
    ("src", "dst", "actor", "bumps"),
    [
        (S.REVIEW, S.QUEUED_EDIT, A.OWNER, True),
        (S.DELIVERED, S.REVIEW, A.FAMILY, True),
        (S.PREP, S.QUEUED_EDIT, A.OWNER, False),
        (S.EDITING, S.QUEUED_EDIT, A.SYSTEM, False),
        (S.FINAL_REVIEW, S.REVIEW, A.OWNER, False),
        (S.REVIEW, S.REVIEW, A.OWNER, False),
    ],
)
def test_round_counter(
    session: Session, src: Status, dst: Status, actor: Actor, bumps: bool
) -> None:
    photo = make_photo(session, src)
    before = photo.round
    transition(session, photo, dst, actor)
    assert photo.round == before + (1 if bumps else 0)
