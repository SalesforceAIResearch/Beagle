"""Meta-agent (proposer) dispatch — route the proposer stages to an editor backend.

Two modes, resolved per call, so this one file works both standalone and hosted:

**Standalone** (no host): env-gated dispatch via ``META_AGENT`` (default ``cursor``)::

    cursor       -> cursor_agent.run         (reasoning effort lives in the model slug)
    monet_code   -> monet_code_agent.run     (--effort: none|low|medium|high|max)
    claude_code  -> claude_code_agent.run    (--effort: low|medium|high|xhigh|max)

``monet`` / ``claude`` are accepted as aliases. With the default ``cursor`` this module
forwards **verbatim** to ``cursor_agent.run``, so behaviour is byte-identical to the
pre-dispatch pipeline.

**Hosted** (an embedder injected an editor): every call routes to that editor instead,
and ``META_AGENT`` is inert. This is the proposer/evolver *seam*: the host decides what
the proposer is, and the ~40 ``meta_agent.run(...)`` call sites elsewhere in the driver
do not change.

WHY THE SEAM LIVES HERE RATHER THAN IN THE HOST. An embedder previously had to *replace*
this file to redirect the proposer, because the driver's workers are separate subprocesses
that re-import ``self_evolve.meta_agent`` fresh — patching ``sys.modules`` in the parent
does not reach them. File replacement is exactly the "manual adjustment" that stops this
driver from being a drop-in plugin: it forks a file the driver's own authors keep editing,
so every upstream change has to be re-applied by hand. Injection through a documented
entry point removes that fork.

The injected object needs one method::

    edit(instruction, workspace, *, plan_mode=..., model=..., timeout_s=...,
         extra_args=..., log_path=...) -> result

and the result needs the fields the call sites read: ``text`` / ``exit_code`` / ``error``
/ ``usage`` / ``tool_calls``.

SUBPROCESS RE-ENTRY. Because workers re-import this module, an in-process ``set_editor``
does not survive into them. A worker rebuilds the editor from the campaign config instead
— see ``set_editor_from_spec`` and the loader in ``run_config``. That is why the spec form
exists at all, and why it must stay serialisable.
"""
from __future__ import annotations

import os
from typing import Any

# The native backends are imported lazily, inside the dispatch. Importing cursor_agent at
# module scope would drag its dependency chain (jinja2 and friends) into every process that
# merely touches this module -- including a host that injected its own editor and will never
# call a native backend at all. A seam that forces its alternatives to be installed is not
# much of a seam.

# Normalize the env value (incl. back-compat aliases) to a canonical backend id.
_ALIASES = {
    "cursor": "cursor",
    "monet": "monet_code",
    "monet_code": "monet_code",
    "claude": "claude_code",
    "claude_code": "claude_code",
}

# Process-local injected editor. None => fall back to the env-gated dispatch above.
_EDITOR: Any | None = None

# Optional host-supplied factory turning a serialisable spec into an editor. Registered
# by the embedder; absent when the driver runs standalone.
_EDITOR_BUILDER: Any | None = None


def register_editor_builder(builder: Any | None) -> None:
    """Register the host's ``spec -> editor`` factory (used by ``set_editor_from_spec``).

    Kept separate from ``set_editor`` because a worker subprocess cannot inherit an editor
    *object*, only the config that describes one.
    """
    global _EDITOR_BUILDER
    _EDITOR_BUILDER = builder


def set_editor(editor: Any | None) -> None:
    """Inject the proposer backend for this process (``None`` restores env dispatch)."""
    global _EDITOR
    _EDITOR = editor


def current_editor() -> Any | None:
    """The injected editor, or ``None`` when running standalone."""
    return _EDITOR


def set_editor_from_spec(spec: Any) -> Any:
    """Rebuild and inject the editor from a serialisable spec (``{name, config, model?}``).

    This is the subprocess-safe half of the seam: workers get the spec through the campaign
    config, not the object. Resolution order is a registered builder first, then a hosting
    package that exposes its own seam — so a standalone driver never imports a host, and a
    hosted one needs no extra wiring.
    """
    if spec is None:
        return None
    if _EDITOR_BUILDER is not None:
        editor = _EDITOR_BUILDER(spec)
        set_editor(editor)
        return editor
    shim = _host_shim()
    if shim is not None and hasattr(shim, "set_editor_from_spec"):
        # The host owns construction and its own module-level editor; mirror it locally so
        # active_backend() and run() agree with the host.
        editor = shim.set_editor_from_spec(spec)
        set_editor(editor)
        return editor
    raise RuntimeError(
        "meta_agent.set_editor_from_spec: no editor builder registered and no host seam "
        "available. Call register_editor_builder(...) during startup, or set_editor(...) "
        "directly, before the proposer runs."
    )


def _host_shim():
    """The hosting package's seam module, if this driver is running embedded.

    Imported lazily and defensively: standalone runs must not require the host to be
    installed, and a host that is present but broken must not take the proposer down at
    import time.
    """
    try:  # pragma: no cover - depends on the deployment
        from beagle.algorithms.darwinx import meta_agent as shim  # type: ignore
    except Exception:
        return None
    return shim


def _resolve_editor() -> Any | None:
    """The editor to use for this call: locally injected, else the host's, else None."""
    if _EDITOR is not None:
        return _EDITOR
    shim = _host_shim()
    if shim is not None:
        try:
            return shim.current_editor()
        except Exception:
            return None
    return None


def active_backend() -> str:
    """Canonical proposer backend id.

    Hosted: the injected editor's ``name`` (falling back to ``injected`` if it has none).
    Standalone: ``cursor`` | ``monet_code`` | ``claude_code`` from ``META_AGENT``. Unknown
    values fall back to ``cursor`` so a typo can never silently disable the proposer.
    """
    editor = _resolve_editor()
    if editor is not None:
        return getattr(editor, "name", None) or "injected"
    raw = os.environ.get("META_AGENT", "cursor").strip().lower()
    return _ALIASES.get(raw, "cursor")


def run(*args: Any, reasoning_effort: str | None = None, **kwargs: Any):
    """Dispatch one proposer call to the active backend.

    Hosted: forwarded to the injected editor's ``edit``. ``reasoning_effort`` is dropped —
    a host-owned editor carries its own effort in its config, and passing both would let
    two sources of truth disagree silently.

    Standalone: cursor is forwarded verbatim (``reasoning_effort`` intentionally dropped —
    cursor has no effort flag, it is encoded in the model slug); monet_code / claude_code
    receive it and translate to their ``--effort``.
    """
    editor = _resolve_editor()
    if editor is not None:
        return _run_editor(editor, *args, **kwargs)

    backend = active_backend()
    if backend == "claude_code":
        from . import claude_code_agent
        return claude_code_agent.run(*args, reasoning_effort=reasoning_effort, **kwargs)
    if backend == "monet_code":
        from . import monet_code_agent
        return monet_code_agent.run(*args, reasoning_effort=reasoning_effort, **kwargs)
    from . import cursor_agent
    return cursor_agent.run(*args, **kwargs)


def _run_editor(editor: Any, *args: Any, **kwargs: Any):
    """Adapt a proposer call site onto the editor's ``edit`` signature.

    Call sites have accumulated keywords over time and not every editor accepts all of
    them, so unknown keys are dropped rather than raising: a host editor that ignores, say,
    ``log_path`` should still run. Prefer the host's own adapter when present, since it
    tracks the editor interface more closely than this fallback can.
    """
    shim = _host_shim()
    if shim is not None and hasattr(shim, "run") and shim.current_editor() is editor:
        return shim.run(*args, **kwargs)
    passthrough = ("plan_mode", "model", "timeout_s", "extra_args", "log_path")
    return editor.edit(*args, **{k: v for k, v in kwargs.items() if k in passthrough})


__all__ = [
    "run",
    "active_backend",
    "set_editor",
    "set_editor_from_spec",
    "current_editor",
    "register_editor_builder",
]
