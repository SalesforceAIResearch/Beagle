"""Render a ``nodes.yaml`` roster from a Slurm script's node allocation.

HyperPod operators already maintain worker membership in the ``#SBATCH
--nodelist`` directive. This helper turns that directive into the static
roster the control plane uses for admin/CLI visibility, while preserving the
operator-owned ``policy:`` section from an existing destination file.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, TextIO

import yaml

from xrlenv.control.kwargs_policy import DEFAULT_POLICY

_IP_HOST_RE = re.compile(r"^ip-(\d+)-(\d+)-(\d+)-(\d+)$")
_NUMERIC_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


class _NodesYamlDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow=flow, indentless=False)


def _split_top_level_commas(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(value):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError(f"invalid Slurm nodelist {value!r}: unexpected ']'")
        elif char == "," and depth == 0:
            part = value[start:idx].strip()
            if part:
                parts.append(part)
            start = idx + 1
    if depth != 0:
        raise ValueError(f"invalid Slurm nodelist {value!r}: unmatched '['")
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _expand_bracket_expr(expr: str) -> list[str]:
    open_idx = expr.find("[")
    if open_idx == -1:
        if "]" in expr:
            raise ValueError(f"invalid Slurm nodelist segment {expr!r}: unexpected ']'")
        return [expr]

    close_idx = expr.find("]", open_idx)
    if close_idx == -1:
        raise ValueError(f"invalid Slurm nodelist segment {expr!r}: unmatched '['")

    prefix = expr[:open_idx]
    inner = expr[open_idx + 1:close_idx]
    suffix = expr[close_idx + 1:]
    expanded: list[str] = []
    for token in _split_top_level_commas(inner):
        match = _NUMERIC_RANGE_RE.fullmatch(token)
        if match is None:
            values = [token]
        else:
            start_s, end_s = match.groups()
            start, end = int(start_s), int(end_s)
            step = 1 if start <= end else -1
            width = max(len(start_s), len(end_s))
            values = [
                f"{number:0{width}d}"
                for number in range(start, end + step, step)
            ]
        for value in values:
            expanded.extend(_expand_bracket_expr(f"{prefix}{value}{suffix}"))
    return expanded


def expand_slurm_nodelist(value: str) -> list[str]:
    """Expand a Slurm nodelist expression into hostnames.

    Supports the common comma-separated and bracketed numeric range forms, for
    example ``node-host,node[01-03,09]``.
    """

    nodes: list[str] = []
    seen: set[str] = set()
    for segment in _split_top_level_commas(value):
        for node in _expand_bracket_expr(segment):
            if not node:
                continue
            if node not in seen:
                nodes.append(node)
                seen.add(node)
    if not nodes:
        raise ValueError("Slurm nodelist is empty")
    return nodes


def extract_slurm_nodelist(script_path: Path) -> str:
    """Return the ``--nodelist`` / ``-w`` value from ``#SBATCH`` directives."""

    nodelist: str | None = None
    for raw_line in script_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("#SBATCH"):
            continue
        directive = stripped.removeprefix("#SBATCH").strip()
        if not directive:
            continue
        try:
            tokens = shlex.split(directive, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"{script_path}: invalid #SBATCH directive {raw_line!r}") from exc

        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.startswith("--nodelist="):
                nodelist = token.split("=", 1)[1]
            elif token == "--nodelist":
                idx += 1
                if idx >= len(tokens):
                    raise ValueError(f"{script_path}: #SBATCH --nodelist missing value")
                nodelist = tokens[idx]
            elif token.startswith("-w="):
                nodelist = token.split("=", 1)[1]
            elif token == "-w":
                idx += 1
                if idx >= len(tokens):
                    raise ValueError(f"{script_path}: #SBATCH -w missing value")
                nodelist = tokens[idx]
            idx += 1

    if nodelist is None:
        raise ValueError(f"{script_path}: no #SBATCH --nodelist/-w directive found")
    return nodelist


def default_address_for_hostname(hostname: str) -> str:
    """Convert AWS ``ip-10-0-...`` hostnames to IPv4; otherwise use hostname."""

    match = _IP_HOST_RE.fullmatch(hostname)
    if match is None:
        return hostname
    return ".".join(match.groups())


def _format_template(template: str, *, hostname: str, address: str) -> str:
    try:
        return template.format(hostname=hostname, address=address)
    except KeyError as exc:
        raise ValueError(
            f"unknown template field {exc.args[0]!r}; valid fields are "
            "{hostname} and {address}",
        ) from exc


def _existing_policy(output_path: Path) -> dict[str, Any] | None:
    if not output_path.exists():
        return None
    raw = yaml.safe_load(output_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{output_path}: existing nodes.yaml top-level must be a mapping")
    policy = raw.get("policy")
    if policy is None:
        return None
    if not isinstance(policy, dict):
        raise ValueError(f"{output_path}: existing policy must be a mapping")
    return policy


def _existing_sysbox_ids(output_path: Path) -> set[str]:
    """Node ids marked ``sysbox: true`` in the existing destination file.

    The generator rebuilds the ``nodes:`` list fresh from the Slurm nodelist,
    so without this the operator-declared Sysbox pool would be wiped on every
    regeneration (like ``policy:``, the pool is operator-owned state, not
    derivable from the nodelist). We read it back and re-apply it, so the pool
    survives a regen — the same preservation contract as ``policy``."""
    if not output_path.exists():
        return set()
    raw = yaml.safe_load(output_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return set()
    ids: set[str] = set()
    for entry in raw.get("nodes") or []:
        if isinstance(entry, dict) and entry.get("sysbox") and entry.get("id"):
            ids.add(str(entry["id"]))
    return ids


def _existing_runtime_caps(output_path: Path) -> dict[str, dict[str, int]]:
    """Per-node ``max_concurrent_by_runtime`` in the existing destination file.

    Same preservation contract as ``policy:`` and the sysbox pool: the generator
    rebuilds ``nodes:`` fresh, so without re-applying it a per-node runtime
    concurrency cap (operator-owned state, not derivable from the nodelist) would
    be wiped on every regeneration. Returns ``{node_id: {runtime: cap}}`` for
    nodes that carry a non-empty map."""
    if not output_path.exists():
        return {}
    raw = yaml.safe_load(output_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    caps: dict[str, dict[str, int]] = {}
    for entry in raw.get("nodes") or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        m = entry.get("max_concurrent_by_runtime")
        if isinstance(m, dict) and m:
            caps[str(entry["id"])] = {str(k): int(v) for k, v in m.items()}
    return caps


def render_nodes_inventory(
    *,
    hostnames: list[str],
    source_script: Path,
    output_path: Path,
    id_template: str,
    address_template: str,
    cloud: str | None,
    backends: list[str],
    auth_token_env: str | None,
    sysbox_nodes: set[str] | None = None,
    allowed_runtimes: list[str] | None = None,
    sysbox_max_concurrent: int | None = None,
    allowed_host_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Build the serializable ``nodes.yaml`` document.

    ``sysbox_nodes`` (from ``--sysbox-node``) marks Sysbox-pool members; a
    node matches by either its raw hostname or its generated id (forgiving).
    Combined with the pool preserved from the existing destination file, so a
    marker set once survives regeneration and can also be (re)declared on the
    command line.

    ``allowed_runtimes`` (from ``--allowed-runtime``) is additively merged into
    ``policy.allowed_runtimes`` — so a deploy can ensure ``sysbox-runc`` is
    permitted at generation time without hand-editing the preserved policy. It
    never removes existing entries.

    ``sysbox_max_concurrent`` (from ``--sysbox-max-concurrent``) stamps
    ``max_concurrent_by_runtime: {sysbox-runc: N}`` on every Sysbox-pool node so
    the per-node runtime concurrency cap (sysbox-fs wedge prevention) is set at
    generation time rather than hand-edited. A per-node value already present in
    the destination file is PRESERVED and wins over this flag default — an
    operator override in the file survives a regen (same contract as ``policy``
    / the sysbox pool)."""

    flag_pool = set(sysbox_nodes or ())
    preserved_pool = _existing_sysbox_ids(output_path)
    preserved_caps = _existing_runtime_caps(output_path)

    nodes: list[dict[str, Any]] = []
    for hostname in hostnames:
        default_address = default_address_for_hostname(hostname)
        address = _format_template(
            address_template,
            hostname=hostname,
            address=default_address,
        )
        node_id = _format_template(
            id_template,
            hostname=hostname,
            address=default_address,
        )
        entry: dict[str, Any] = {"id": node_id}
        if cloud:
            entry["cloud"] = cloud
        if backends:
            entry["backends"] = list(backends)
        if auth_token_env:
            entry["auth_token_env"] = auth_token_env
        if address:
            entry["address"] = address
        # Sysbox pool membership — matched by hostname OR generated id (flag),
        # OR preserved from the existing file. Only emitted when true so the
        # ordinary node entry is unchanged.
        is_sysbox = (
            hostname in flag_pool
            or node_id in flag_pool
            or node_id in preserved_pool
        )
        if is_sysbox:
            entry["sysbox"] = True
        # Per-node runtime concurrency cap (sysbox-fs wedge prevention). A value
        # preserved from the destination file wins (operator override survives a
        # regen); otherwise the --sysbox-max-concurrent flag default is stamped
        # on Sysbox-pool nodes. Uncapped ⇒ field omitted ⇒ unlimited.
        if node_id in preserved_caps:
            entry["max_concurrent_by_runtime"] = preserved_caps[node_id]
        elif is_sysbox and sysbox_max_concurrent is not None:
            entry["max_concurrent_by_runtime"] = {
                "sysbox-runc": sysbox_max_concurrent,
            }
        nodes.append(entry)

    policy = _existing_policy(output_path)
    if policy is None:
        policy = DEFAULT_POLICY.model_dump(mode="json")

    # Additively merge --allowed-runtime into policy.allowed_runtimes (never
    # removes an existing entry), so a deploy can permit sysbox-runc at
    # generation time rather than hand-editing the preserved policy.
    if allowed_runtimes:
        merged = list(policy.get("allowed_runtimes") or [])
        for rt in allowed_runtimes:
            if rt not in merged:
                merged.append(rt)
        policy = {**policy, "allowed_runtimes": merged}

    # Additively merge --allowed-host-path into policy.allowed_host_paths (same
    # contract as allowed_runtimes). Lets a deploy authorize the EvoClaw golden
    # bind (a read-only host mount, real on sysbox nodes — spec 19) via an env
    # knob (XRLENV_ALLOWED_HOST_PATHS) instead of a personal absolute path baked
    # into the committed roster. Prefix-matched at gate time, so one shared
    # read-only data-root entry covers every mount under it.
    if allowed_host_paths:
        merged_paths = list(policy.get("allowed_host_paths") or [])
        for hp in allowed_host_paths:
            if hp not in merged_paths:
                merged_paths.append(hp)
        policy = {**policy, "allowed_host_paths": merged_paths}

    return {
        "version": 1,
        "nodes": nodes,
        "policy": policy,
    }


def write_nodes_inventory(inventory: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.dump(inventory, Dumper=_NodesYamlDumper, sort_keys=False)
    output_path.write_text(body, encoding="utf-8")


def cmd_nodes_from_slurm(
    *,
    slurm_script: Path,
    output: Path,
    id_template: str,
    address_template: str,
    cloud: str | None,
    backends: list[str],
    auth_token_env: str | None,
    out: TextIO,
    sysbox_nodes: list[str] | None = None,
    allowed_runtimes: list[str] | None = None,
    sysbox_max_concurrent: int | None = None,
    allowed_host_paths: list[str] | None = None,
) -> int:
    nodelist = extract_slurm_nodelist(slurm_script)
    hostnames = expand_slurm_nodelist(nodelist)
    inventory = render_nodes_inventory(
        hostnames=hostnames,
        source_script=slurm_script,
        output_path=output,
        id_template=id_template,
        address_template=address_template,
        cloud=cloud,
        backends=backends,
        auth_token_env=auth_token_env,
        sysbox_nodes=set(sysbox_nodes or ()),
        allowed_runtimes=allowed_runtimes,
        sysbox_max_concurrent=sysbox_max_concurrent,
        allowed_host_paths=allowed_host_paths,
    )
    write_nodes_inventory(inventory, output)
    _n_sysbox = sum(1 for n in inventory["nodes"] if n.get("sysbox"))
    _pool_note = f" ({_n_sysbox} in sysbox pool)" if _n_sysbox else ""
    out.write(f"wrote {len(hostnames)} nodes to {output}{_pool_note}\n")
    return 0


__all__ = [
    "cmd_nodes_from_slurm",
    "default_address_for_hostname",
    "expand_slurm_nodelist",
    "extract_slurm_nodelist",
    "render_nodes_inventory",
    "write_nodes_inventory",
]
