#!/usr/bin/env python3
"""Scratch-registry garbage collector (:5012).

Reclaims build-on-demand images by **TTL** (age out) and **per-namespace
quota** (oldest-first), NEVER touching a digest an active run still references.
The eviction *policy* is :func:`xrlenv.control.scratch_gc.select_reclaim_targets`
(unit-tested); this script is the registry-facing plumbing.

Run it on the control-plane box (FSx-visible) on a schedule::

    python deploy/registry/scratch_registry_gc.py \
        --registry 127.0.0.1:5012 \
        --storage-path /fsx/home/$USER/xrlenv-registry/scratch \
        --container xrlenv-registry-scratch \
        --ttl 72h --quota-gb 500 \
        --exempt-file /run/xrlenv/scratch-active-digests.txt

Active-run exemption: pass ``--exempt-file`` (one ``sha256:...`` per line) or
``--exempt-url`` (a CP endpoint returning the same). The control plane knows
which scratch digests active runs pin; point one of these at it. With neither,
the GC runs TTL/quota with an EMPTY exemption set — safe only when the TTL is
set well beyond the longest run (it warns loudly in that case).

``--dry-run`` prints what would be reclaimed and deletes nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Ensure the package is importable when run as a bare script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xrlenv.control.scratch_gc import (
    ScratchImage,
    select_reclaim_targets,
)

_MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ],
)


def _parse_duration(s: str) -> float:
    """Parse ``72h`` / ``30m`` / ``90s`` / ``3600`` (bare = seconds)."""
    s = s.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s and s[-1] in units:
        return float(s[:-1]) * units[s[-1]]
    return float(s)


def _http(method: str, url: str, *, accept: str | None = None) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, method=method)
    if accept:
        req.add_header("Accept", accept)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


def _list_scratch_repos(base: str) -> list[str]:
    # Paginate: registry:3 caps the page size (rejects a huge ``n``), so walk
    # ``n=1000`` pages following the RFC-5988 ``Link: ...; rel="next"`` header.
    repos: list[str] = []
    url: str = f"{base}/v2/_catalog?n=1000"
    while url:
        status, headers, body = _http("GET", url)
        if status != 200:
            raise SystemExit(f"registry _catalog returned {status}: {body[:200]!r}")
        repos.extend(json.loads(body).get("repositories", []) or [])
        link = headers.get("Link", "")
        if 'rel="next"' in link and "<" in link and ">" in link:
            nxt = link[link.find("<") + 1: link.find(">")]
            url = f"{base}{nxt}"
        else:
            url = ""
    return [r for r in repos if r.startswith("scratch/")]


def _resolve_image(base: str, repo: str, storage_path: Path | None) -> ScratchImage | None:
    """Resolve a repo's manifest digest + size + last-used. Returns None when
    the repo has no ``latest`` tag (already partially GC'd)."""
    status, headers, body = _http(
        "GET", f"{base}/v2/{repo}/manifests/latest", accept=_MANIFEST_ACCEPT,
    )
    if status != 200:
        return None
    digest = headers.get("Docker-Content-Digest", "")
    if not digest:
        return None
    size = _manifest_size(base, repo, json.loads(body))
    last_used = _last_used_at(storage_path, repo, digest)
    return ScratchImage(repo=repo, digest=digest, size_bytes=size, last_used_at=last_used)


def _manifest_size(base: str, repo: str, manifest: dict[str, Any]) -> int:
    """Total blob footprint of a manifest. Handles both a single image
    manifest (``config`` + ``layers``) and an OCI image index / manifest list
    (``manifests``) — docker 29 pushes the latter by default — by recursing
    into each child manifest."""
    children = manifest.get("manifests")
    if children:
        total = 0
        for child in children:
            child_digest = child.get("digest")
            if not child_digest:
                total += int(child.get("size", 0))
                continue
            status, _h, body = _http(
                "GET", f"{base}/v2/{repo}/manifests/{child_digest}",
                accept=_MANIFEST_ACCEPT,
            )
            if status == 200:
                total += _manifest_size(base, repo, json.loads(body))
            else:
                total += int(child.get("size", 0))
        return total
    size = int(manifest.get("config", {}).get("size", 0))
    for layer in manifest.get("layers", []):
        size += int(layer.get("size", 0))
    return size


def _last_used_at(storage_path: Path | None, repo: str, digest: str) -> float:
    """Best-effort last-used epoch. The distribution registry doesn't expose
    pull time, so we use the manifest revision link's mtime on the FSx store
    when available; otherwise fall back to ``now`` (never TTL-reclaim what we
    can't age)."""
    if storage_path is None or ":" not in digest:
        return time.time()
    algo, hexd = digest.split(":", 1)
    link = (
        storage_path / "docker" / "registry" / "v2" / "repositories" / repo
        / "_manifests" / "revisions" / algo / hexd / "link"
    )
    try:
        return link.stat().st_mtime
    except OSError:
        return time.time()


def _load_exempt(args: argparse.Namespace) -> frozenset[str]:
    digests: set[str] = set()
    if args.exempt_file:
        for line in Path(args.exempt_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                digests.add(line)
    if args.exempt_url:
        status, _h, body = _http("GET", args.exempt_url)
        if status != 200:
            raise SystemExit(f"--exempt-url returned {status}")
        payload = json.loads(body)
        if isinstance(payload, list):
            digests.update(payload)
        else:
            # The CP endpoint returns {"repos": [...]}; also accept
            # {"digests": [...]}. select_reclaim_targets matches either an
            # image's repo or its digest against this set.
            digests.update(payload.get("digests", []))
            digests.update(payload.get("repos", []))
    return frozenset(digests)


def _delete_manifest(base: str, repo: str, digest: str) -> bool:
    status, _h, _b = _http("DELETE", f"{base}/v2/{repo}/manifests/{digest}")
    return status in (202, 200, 404)  # 404 = already gone (idempotent)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scratch-registry GC (TTL + quota, active-run exempt).")
    ap.add_argument("--registry", required=True, help="host:port of the scratch registry")
    ap.add_argument("--scheme", default="http", choices=["http", "https"])
    ap.add_argument("--ttl", default=None, help="reclaim images older than this (e.g. 72h); omit to disable")
    ap.add_argument("--quota-gb", type=float, default=None, help="per-registry soft cap in GiB; omit to disable")
    ap.add_argument("--storage-path", default=None, help="FSx blob-store path (for last-used mtimes)")
    ap.add_argument("--container", default=None, help="registry container name for garbage-collect")
    ap.add_argument("--exempt-file", default=None, help="file of sha256:... digests active runs pin")
    ap.add_argument("--exempt-url", default=None, help="URL returning the active digest set (CP endpoint)")
    ap.add_argument("--dry-run", action="store_true", help="print reclaim targets; delete nothing")
    args = ap.parse_args(argv)

    base = f"{args.scheme}://{args.registry}"
    ttl = _parse_duration(args.ttl) if args.ttl else None
    quota = int(args.quota_gb * 1024**3) if args.quota_gb is not None else None
    storage = Path(args.storage_path) if args.storage_path else None
    exempt = _load_exempt(args)

    if not exempt and not args.exempt_file and not args.exempt_url:
        print(
            "WARNING: no --exempt-file/--exempt-url — running with an EMPTY "
            "active-run exemption set. Safe ONLY if --ttl is well beyond your "
            "longest run.",
            file=sys.stderr,
        )

    images = [
        img for repo in _list_scratch_repos(base)
        if (img := _resolve_image(base, repo, storage)) is not None
    ]
    targets = select_reclaim_targets(
        images, now=time.time(), ttl_seconds=ttl, quota_bytes=quota, exempt_digests=exempt,
    )
    total_gb = sum(i.size_bytes for i in images) / 1024**3
    reclaim_gb = sum(i.size_bytes for i in targets) / 1024**3
    print(
        f"scratch GC: {len(images)} images ({total_gb:.1f} GiB); "
        f"reclaim {len(targets)} ({reclaim_gb:.1f} GiB); exempt {len(exempt)}",
    )
    for img in targets:
        print(f"  reclaim {img.repo}@{img.digest} ({img.size_bytes / 1024**3:.2f} GiB)")

    if args.dry_run:
        print("dry-run: nothing deleted.")
        return 0

    deleted = sum(1 for img in targets if _delete_manifest(base, img.repo, img.digest))
    print(f"deleted {deleted}/{len(targets)} manifests")
    if deleted and args.container:
        # Reclaim blobs. registry:3 supports online GC; run inside the container.
        subprocess.run(
            ["docker", "exec", args.container, "registry", "garbage-collect",
             "/etc/distribution/config.yml"],
            check=False,
        )
        print("ran registry garbage-collect")
    elif deleted:
        print("NOTE: pass --container <name> to reclaim blobs via garbage-collect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
