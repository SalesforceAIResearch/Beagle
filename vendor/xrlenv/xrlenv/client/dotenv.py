""".env helpers for in-container secrets — harbor-agnostic.

Two shapes for getting per-rollout secrets into a container:

1. **As env vars at acquire time.** Parse ``.env`` on the operator
   host, pass the resulting ``dict[str, str]`` to
   ``Client.acquire_container(environment=...)``. The Docker create
   call sets those variables in the container's process env; any
   in-container agent that reads from ``os.environ`` / ``$VAR`` sees
   them. Best for the **outside-container agent pattern** and for
   tools that already read from env.

2. **As an .env file copied into the container.** Tar the file,
   ``put_archive`` it under a known target directory; let
   in-container code read it via whatever dotenv-loader it already
   uses (python-dotenv, direnv, ``set -a; source .env; set +a``,
   etc.). Best for the **inside-container agent pattern** when the
   harness or agent expects a file rather than env vars.

Both shapes are platform-level primitives; they don't know anything
about harbor / SWE-bench / specific agents. Operators pick the shape
their workload's in-container code already wants.

The helpers here are stdlib-only and import cheap so the public
:mod:`xrlenv.client` surface stays light. ``upload_dotenv`` runs an
``mkdir -p`` against the target directory before
:meth:`ClusterContainerSession.put_archive` because Docker's
``put_archive`` refuses to create the target dir itself —
operator-visible failure mode without the mkdir, surfaces as a
404 ``NotFound`` from the docker API.
"""

from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

# Re-export the auto-load helpers from the private top-level
# module. They live there (not here) so ``xrlenv/__init__.py``'s
# import-time call to ``_maybe_auto_load_dotenv`` doesn't fan out
# through ``xrlenv.client.__init__`` → Client → docker / gRPC /
# prometheus — which would break the in-sandbox stub's slim image
# contract (test_import_cycles.py).
from xrlenv._dotenv_autoload import (
    load_dotenv,
    parse_dotenv,
)

if TYPE_CHECKING:
    from xrlenv.client.container_session import ClusterContainerSession


LOGGER = logging.getLogger(__name__)


__all__ = ["load_dotenv", "parse_dotenv", "upload_dotenv"]


# ``parse_dotenv`` + ``load_dotenv`` are re-exported above from
# ``xrlenv._dotenv_autoload``. Their full docstrings + contracts live
# there; ``__all__`` advertises them as part of this module's public
# surface so ``from xrlenv.client.dotenv import parse_dotenv`` keeps
# working.


# ──────────────────────────────────────────────────────────────────────────────
# upload_dotenv
# ──────────────────────────────────────────────────────────────────────────────


async def upload_dotenv(
    session: ClusterContainerSession,
    *,
    source: str | Path,
    target_dir: str = "/workspace",
    arcname: str = ".env",
    mkdir: bool = True,
    exec_timeout_s: float = 30.0,
) -> str:
    """Copy a local ``.env`` file into a remote container.

    The file lands at ``f"{target_dir}/{arcname}"`` inside the
    container. Default target is ``/workspace/.env`` since most
    benchmark / agent images use ``/workspace`` or ``/app`` as their
    working directory; pass ``target_dir="/app"`` (or wherever your
    workload's in-container code looks) when the convention differs.

    ``mkdir=True`` (default) runs ``mkdir -p <target_dir>`` via
    :meth:`session.exec` before the put_archive call. Docker's
    ``put_archive`` returns a 404 if the target directory doesn't
    exist; the pre-mkdir closes that gotcha so operators don't have
    to know about it.

    Returns the in-container path the file landed at, so the caller
    can log it or thread it into a follow-on ``exec``.

    Raises ``FileNotFoundError`` if ``source`` doesn't exist locally.
    """
    src = Path(source)
    if not src.is_file():
        raise FileNotFoundError(
            f"upload_dotenv: source {src} is not a file",
        )
    if mkdir:
        await session.exec(
            ["mkdir", "-p", target_dir],
            timeout_s=exec_timeout_s,
        )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        tf.add(str(src), arcname=arcname)
    await session.put_archive(target_dir=target_dir, tarball=buf.getvalue())
    return f"{target_dir.rstrip('/')}/{arcname}"
