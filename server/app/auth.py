"""Роли (ТЗ §1, §9): владелец и отец.

Входа нет: всё на доверии к домашней сети. Отец отличается адресом — запросы на
`FAMILY_HOST` получают роль «отец» и видят только `/family` (остальное — переадресация туда,
POST — 403). Любой другой хост — владелец.
"""

from collections.abc import Iterable
from enum import StrEnum

from fastapi import Request
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send


class Role(StrEnum):
    OWNER = "owner"
    FAMILY = "family"


FAMILY_HOME = "/family"


def host_name(value: str) -> str:
    """Имя из `Host` без порта, в нижнем регистре: `Family.Example:443` → `family.example`."""
    host = value.strip().lower()
    if host.startswith("["):  # IPv6: [::1]:8000
        return host.split("]", 1)[0] + "]"
    return host.split(":", 1)[0].rstrip(".")


def parse_hosts(value: str) -> frozenset[str]:
    """`FAMILY_HOST`: один хост или несколько через запятую."""
    return frozenset(h for part in value.split(",") if (h := host_name(part)))


def role_for(host: str, family_hosts: Iterable[str]) -> Role:
    return Role.FAMILY if host_name(host) in family_hosts else Role.OWNER


def family_allowed(path: str) -> bool:
    return (
        path == FAMILY_HOME
        or path.startswith((f"{FAMILY_HOME}/", "/static/"))
        or path == "/healthz"
    )


def role(request: Request) -> Role:
    value: Role = getattr(request.state, "role", Role.OWNER)
    return value


class RoleMiddleware:
    """Ставит `request.state.role` и не пускает отца дальше `/family`."""

    def __init__(self, app: ASGIApp, family_hosts: frozenset[str]) -> None:
        self.app = app
        self.family_hosts = family_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        current = role_for(Headers(scope=scope).get("host", ""), self.family_hosts)
        scope.setdefault("state", {})["role"] = current
        if current is Role.FAMILY and not family_allowed(scope["path"]):
            response: Response
            if scope["method"] in ("GET", "HEAD"):
                response = RedirectResponse(FAMILY_HOME, status_code=303)
            else:
                response = PlainTextResponse("Недоступно", status_code=403)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
