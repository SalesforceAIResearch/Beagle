#!/usr/bin/env python3
"""Retag images in the xrlenv private registry from one repository namespace to
another, IN PLACE and WITHOUT rebuilding — e.g.
``xrlenv-seta-env/<id>:main`` → ``seta-env/<id>:main``.

A registry retag is just a manifest copy: layer + config blobs are
content-addressed and already stored, so this **cross-repo mounts** them into the
destination repo (no upload — the bytes never move) and ``PUT``s the same manifest
under the new name. The digest is identical under both names, so
``image_pin_mode: registry_digest`` pinning is unaffected.

What gets retagged is discovered from the registry's own ``/v2/_catalog`` (every
repo under ``--from-namespace/``), so this does not depend on a build plan.

  # 1. copy xrlenv-seta-env/* -> seta-env/* (idempotent; safe to re-run):
  python deploy/registry/retag_registry_images.py \
      --registry <registry-host>:5011 \
      --from-namespace xrlenv-seta-env --to-namespace seta-env

  # 2. once you've verified seta-env/* pulls, drop the old tags:
  python deploy/registry/retag_registry_images.py \
      --registry <registry-host>:5011 \
      --from-namespace xrlenv-seta-env --to-namespace seta-env --delete-source

``--delete-source`` removes the old manifests (the registry needs ``delete``
enabled — run-registry-private.sh's config-private.yml has it). The shared blobs
stay (they're referenced by the new name); reclaim any truly-orphaned blobs later
with the registry's offline ``garbage-collect``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_MANIFEST_ACCEPT = ", ".join((
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.index.v1+json",
))
_LIST_TYPES = {
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.index.v1+json",
}


# ── Pure helpers (unit-tested) ────────────────────────────────────────────────


def dst_repo(src_repo: str, from_ns: str, to_ns: str) -> str:
    """Map a source repository to its destination by swapping the leading
    namespace segment. ``xrlenv-seta-env/88`` → ``seta-env/88``."""
    if src_repo == from_ns:
        return to_ns
    prefix = from_ns + "/"
    if src_repo.startswith(prefix):
        return to_ns + "/" + src_repo[len(prefix):]
    return src_repo  # not under from_ns — leave unchanged


def repos_under(catalog: list[str], from_ns: str) -> list[str]:
    """The catalog repos that live under ``from_ns`` (exact or ``from_ns/...``)."""
    prefix = from_ns + "/"
    return sorted(r for r in catalog if r == from_ns or r.startswith(prefix))


# ── Registry v2 HTTP ──────────────────────────────────────────────────────────


def _req(
    method: str, url: str, *, headers: dict[str, str] | None = None,
    data: bytes | None = None, timeout: float = 120.0,
) -> tuple[int, dict[str, str], bytes]:
    """One HTTP request. Returns (status, headers, body); HTTP error responses
    (404, 202, …) are returned, not raised."""
    req = urllib.request.Request(url, method=method, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {k: v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k: v for k, v in (exc.headers or {}).items()}, exc.read()


def list_catalog(base: str, *, page: int = 1000) -> list[str]:
    """All repositories, following ``Link: rel=next`` pagination. (registry:3
    rejects very large ``n``; 1000 is the conventional page size.)"""
    repos: list[str] = []
    url: str | None = f"{base}/v2/_catalog?n={page}"
    while url:
        status, hdrs, body = _req("GET", url)
        if status != 200:
            raise RuntimeError(f"catalog {url} -> HTTP {status}: {body[:200]!r}")
        repos.extend(json.loads(body).get("repositories", []))
        link = hdrs.get("Link", "")
        url = None
        if 'rel="next"' in link:
            nxt = link.split(";")[0].strip().lstrip("<").rstrip(">")
            url = nxt if nxt.startswith("http") else base + nxt
    return repos


def list_tags(base: str, repo: str) -> list[str]:
    status, _, body = _req("GET", f"{base}/v2/{repo}/tags/list")
    if status != 200:
        return []
    return json.loads(body).get("tags") or []


def manifest_digest(base: str, repo: str, ref: str) -> str | None:
    """HEAD a manifest; return its digest if present, else None."""
    status, hdrs, _ = _req(
        "HEAD", f"{base}/v2/{repo}/manifests/{ref}",
        headers={"Accept": _MANIFEST_ACCEPT},
    )
    return hdrs.get("Docker-Content-Digest") if status == 200 else None


def _ensure_blob(base: str, dst: str, digest: str, src: str) -> None:
    """Make ``digest`` available in repo ``dst``. Cross-repo mounts from ``src``
    (201, no upload); if the registry declines the mount (202) it streams the blob
    from ``src`` and completes the upload."""
    status, hdrs, _ = _req(
        "POST", f"{base}/v2/{dst}/blobs/uploads/?mount={digest}&from={src}",
    )
    if status == 201:
        return
    if status != 202:
        raise RuntimeError(f"mount {digest} into {dst} -> HTTP {status}")
    # Fallback: pull the blob bytes from src and push them to the upload session.
    location = hdrs.get("Location", "")
    if location and not location.startswith("http"):
        location = base + location
    sb, _, blob = _req("GET", f"{base}/v2/{src}/blobs/{digest}")
    if sb != 200:
        raise RuntimeError(f"get blob {digest} from {src} -> HTTP {sb}")
    sep = "&" if "?" in location else "?"
    pb, _, _ = _req(
        "PUT", f"{location}{sep}digest={digest}", data=blob,
        headers={"Content-Type": "application/octet-stream"},
    )
    if pb not in (201, 202):
        raise RuntimeError(f"upload blob {digest} to {dst} -> HTTP {pb}")


def _get_manifest(base: str, repo: str, ref: str) -> tuple[int, str, str | None, bytes]:
    """GET a manifest by tag or digest → (status, content_type, digest, body)."""
    status, hdrs, body = _req(
        "GET", f"{base}/v2/{repo}/manifests/{ref}",
        headers={"Accept": _MANIFEST_ACCEPT},
    )
    ctype = hdrs.get("Content-Type", "").split(";")[0].strip()
    return status, ctype, hdrs.get("Docker-Content-Digest"), body


def _copy_manifest_tree(
    base: str, src_repo: str, dst: str, ref: str, body: bytes, ctype: str,
) -> None:
    """Copy a manifest from ``src_repo`` to ``dst`` addressed by ``ref`` (tag or
    digest). Handles both image manifests (mount config + layer blobs) and OCI
    indexes / manifest lists (recurse into each sub-manifest by digest FIRST, so
    the index's references resolve in dst), then PUT the manifest itself."""
    if ctype in _LIST_TYPES:
        for sub in json.loads(body).get("manifests", []):
            sd = sub["digest"]
            sstatus, sctype, _, sbody = _get_manifest(base, src_repo, sd)
            if sstatus != 200:
                raise RuntimeError(f"get sub-manifest {sd} from {src_repo} -> HTTP {sstatus}")
            _copy_manifest_tree(base, src_repo, dst, sd, sbody, sctype)
    else:
        manifest = json.loads(body)
        for d in [manifest["config"]["digest"],
                  *(layer["digest"] for layer in manifest.get("layers", []))]:
            _ensure_blob(base, dst, d, src_repo)
    pstatus, _, pbody = _req(
        "PUT", f"{base}/v2/{dst}/manifests/{ref}", data=body,
        headers={"Content-Type": ctype},
    )
    if pstatus != 201:
        raise RuntimeError(f"put manifest {dst}:{ref} -> HTTP {pstatus}: {pbody[:200]!r}")


def retag_one(
    base: str, src_repo: str, dst: str, tag: str, *,
    force: bool, delete_source: bool,
) -> dict[str, Any]:
    """Copy ``src_repo:tag`` → ``dst:tag`` (manifest tree + blob mounts),
    optionally deleting the source. Returns a result record; never raises."""
    rec: dict[str, Any] = {"src": f"{src_repo}:{tag}", "dst": f"{dst}:{tag}",
                           "status": "failed", "error": None, "deleted": False}
    try:
        status, ctype, src_digest, body = _get_manifest(base, src_repo, tag)
        if status != 200:
            rec["error"] = f"get src manifest -> HTTP {status}"
            return rec
        if not force and manifest_digest(base, dst, tag) == src_digest:
            rec["status"] = "skipped"
            if delete_source:
                _delete(base, src_repo, src_digest, rec)
            return rec
        _copy_manifest_tree(base, src_repo, dst, tag, body, ctype)
        if manifest_digest(base, dst, tag) != src_digest:
            rec["error"] = "dst digest != src digest after copy"
            return rec
        rec["status"] = "retagged"
        if delete_source:
            _delete(base, src_repo, src_digest, rec)
        return rec
    except Exception as exc:  # one image must not kill the run
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec


def _delete(base: str, repo: str, digest: str | None, rec: dict[str, Any]) -> None:
    if not digest:
        return
    status, _, _ = _req("DELETE", f"{base}/v2/{repo}/manifests/{digest}")
    # 202 Accepted on success; 404 means it's already gone (idempotent).
    rec["deleted"] = status in (202, 404)
    if status not in (202, 404):
        rec["error"] = (rec.get("error") or "") + f" [delete src -> HTTP {status}]"


# ── Orchestration ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="retag_registry_images", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--registry", required=True,
                   help="Registry host:port, e.g. <registry-host>:5011.")
    p.add_argument("--from-namespace", required=True,
                   help="Source repo namespace, e.g. xrlenv-seta-env.")
    p.add_argument("--to-namespace", required=True,
                   help="Destination repo namespace, e.g. seta-env.")
    p.add_argument("--scheme", default="http", choices=("http", "https"),
                   help="Registry scheme (default http for the trusted-VPC registry).")
    p.add_argument("--concurrency", type=int, default=16,
                   help="Concurrent retag operations (default 16).")
    p.add_argument("--force", action="store_true",
                   help="Re-copy even if the destination tag already matches.")
    p.add_argument("--delete-source", action="store_true",
                   help="Delete each source manifest after a successful copy "
                        "(needs delete-enabled registry).")
    p.add_argument("--dry-run", action="store_true",
                   help="List src→dst pairs and exit without copying.")
    args = p.parse_args(argv)

    base = f"{args.scheme}://{args.registry}"
    try:
        catalog = list_catalog(base)
    except Exception as exc:  # surface a clean operator message, not a traceback
        print(f"ERROR: cannot read registry catalog at {base}: {exc}", file=sys.stderr)
        return 2
    src_repos = repos_under(catalog, args.from_namespace)
    if not src_repos:
        print(f"no repositories under {args.from_namespace!r} in {base}; nothing to do.")
        return 0

    # (src_repo, dst_repo, tag) work items.
    work: list[tuple[str, str, str]] = []
    for repo in src_repos:
        dst = dst_repo(repo, args.from_namespace, args.to_namespace)
        for tag in list_tags(base, repo):
            work.append((repo, dst, tag))

    print(f"{len(src_repos)} source repo(s), {len(work)} tag(s): "
          f"{args.from_namespace}/* → {args.to_namespace}/* on {base}"
          + ("  [+delete-source]" if args.delete_source else ""), flush=True)
    if args.dry_run:
        for src, dst, tag in work[:20]:
            print(f"  would retag {src}:{tag} → {dst}:{tag}")
        if len(work) > 20:
            print(f"  … and {len(work) - 20} more")
        return 0

    counts: dict[str, int] = {}
    deleted = 0
    done = 0
    total = len(work)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [
            pool.submit(retag_one, base, src, dst, tag,
                        force=args.force, delete_source=args.delete_source)
            for src, dst, tag in work
        ]
        for fut in futs:
            rec = fut.result()
            done += 1
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            if rec["deleted"]:
                deleted += 1
            mark = {"retagged": "✓", "skipped": "·"}.get(rec["status"], "✗")
            extra = f"  {rec['error']}" if rec.get("error") else ""
            print(f"{done}/{total} {mark} {rec['status']:<8} {rec['dst']}{extra}",
                  flush=True)

    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    if args.delete_source:
        summary += f", deleted={deleted}"
    print(f"\ndone: {summary}", flush=True)
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
