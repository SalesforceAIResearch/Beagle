#!/usr/bin/env python3
"""Warm the FSx-backed pull-through registry mirror from a build plan.

Pulls each image's manifest + blobs THROUGH the mirror via the registry HTTP
API (``/v2/<repo>/...?ns=docker.io``), which makes the mirror fetch the content
from Docker Hub and cache it on FSx — with NO local docker extraction. So:

* it populates the *central* cache (not some box's local docker store), which is
  what a worker re-pull actually hits;
* it never fills the warming box's disk (no image is unpacked locally);
* it needs NO Docker Hub credentials — the warming client only talks to the
  mirror; the mirror handles upstream auth with its configured PAT.

Usage:
  python3 deploy/registry/warm_images.py <build_plan.yaml> \
      [--mirror http://127.0.0.1:5010] [--concurrency 8] [--limit N]

Live progress (on a TTY) shows count, %, GB cached, MB/s, img/s, elapsed, ETA.
When output is redirected it prints a checkpoint line instead of the live one.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request

_ACCEPT = ",".join(
    [
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
    ]
)


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url)
    req.add_header("Accept", _ACCEPT)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def _fetch_discard(url: str) -> None:
    """GET a blob through the mirror and discard the body (streams in chunks so
    a multi-GB layer never sits in memory). The GET is what makes the proxy
    fetch-and-cache the blob on FSx."""
    with urllib.request.urlopen(url, timeout=900) as r:
        while r.read(1024 * 1024):
            pass


def _manifest_blobs(base: str, mani: dict) -> list[tuple[str, int]]:
    """Return [(digest, size_bytes)] for config + layers, resolving a
    multi-arch index to amd64/linux first."""
    mt = mani.get("mediaType", "")
    if "list" in mt or "index" in mt:
        subs = mani.get("manifests", [])
        chosen = None
        for m in subs:
            p = m.get("platform", {})
            if p.get("architecture") == "amd64" and p.get("os") == "linux":
                chosen = m.get("digest")
                break
        if chosen is None and subs:
            chosen = subs[0].get("digest")
        if chosen is None:
            return []
        mani = _fetch_json(f"{base}/manifests/{chosen}?ns=docker.io")
    out: list[tuple[str, int]] = []
    cfg = mani.get("config", {})
    if cfg.get("digest"):
        out.append((cfg["digest"], int(cfg.get("size", 0))))
    for layer in mani.get("layers", []):
        if layer.get("digest"):
            out.append((layer["digest"], int(layer.get("size", 0))))
    return out


def _blob_path(store: str, digest: str) -> str:
    # distribution filesystem layout:
    #   <root>/docker/registry/v2/blobs/<algo>/<hex[:2]>/<hex>/data
    algo, _, hexd = digest.partition(":")
    return os.path.join(
        store, "docker", "registry", "v2", "blobs", algo, hexd[:2], hexd, "data",
    )


def _blob_cached(store: str | None, digest: str) -> bool:
    """True if the blob is already on the (shared FSx) registry store, so we can
    skip re-streaming it. ``store`` None disables the check (always re-fetch)."""
    return store is not None and os.path.isfile(_blob_path(store, digest))


def warm_one(
    mirror: str, image_ref: str, store: str | None,
) -> tuple[str, str, int, int, bool]:
    """Returns (image_ref, status, fetched_bytes, n_fetched, fully_cached)."""
    repo, _, tag = image_ref.rpartition(":")
    if not repo or not tag:
        return (image_ref, "bad-ref", 0, 0, False)
    base = f"{mirror.rstrip('/')}/v2/{repo}"
    try:
        mani = _fetch_json(f"{base}/manifests/{tag}?ns=docker.io")
        blobs = _manifest_blobs(base, mani)
        fetched_bytes = n_fetched = 0
        for digest, size in blobs:
            if _blob_cached(store, digest):
                continue  # already on FSx — skip the re-stream (idempotent)
            _fetch_discard(f"{base}/blobs/{digest}?ns=docker.io")
            fetched_bytes += size
            n_fetched += 1
        fully_cached = bool(blobs) and n_fetched == 0
        return (image_ref, "ok", fetched_bytes, n_fetched, fully_cached)
    except urllib.error.HTTPError as e:
        return (image_ref, f"http-{e.code}", 0, 0, False)
    except Exception as e:  # report per-image failure and continue the batch
        return (image_ref, f"err-{type(e).__name__}", 0, 0, False)


def _extract_refs(plan_path: str, limit: int) -> list[str]:
    import yaml

    with open(plan_path) as f:
        doc = yaml.safe_load(f)
    refs: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "image_ref" and isinstance(v, str):
                    refs.append(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(doc)
    seen, out = set(), []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out[:limit] if limit > 0 else out


def _fmt_dur(secs: float) -> str:
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _progress(done: int, total: int, ok: int, cached: int, fail: int,
              nbytes: int, start: float) -> str:
    el = max(1e-6, time.time() - start)
    rate = done / el
    eta = (total - done) / rate if rate > 0 else 0.0
    pct = (100 * done // total) if total else 100
    return (
        f"[{done}/{total}] {pct:3d}%  ok={ok} cached={cached} fail={fail}  "
        f"{nbytes / 1e9:.1f} GB fetched  {nbytes / 1e6 / el:.0f} MB/s  "
        f"{rate:.1f} img/s  elapsed {_fmt_dur(el)}  ETA {_fmt_dur(eta)}"
    )


def _default_store() -> str:
    # Prefer the namespaced XRLENV_MIRROR_REGISTRY_STORAGE (three registries now:
    # mirror / private / scratch); fall back to the deprecated
    # XRLENV_REGISTRY_STORAGE with a warning.
    p = os.environ.get("XRLENV_MIRROR_REGISTRY_STORAGE")
    if not p:
        legacy = os.environ.get("XRLENV_REGISTRY_STORAGE")
        if legacy:
            print(
                "==> WARN: XRLENV_REGISTRY_STORAGE is deprecated — rename it to "
                "XRLENV_MIRROR_REGISTRY_STORAGE.", flush=True,
            )
            p = legacy
    if p:
        return p
    return f"/fsx/home/{os.environ.get('USER', '')}/xrlenv-registry/proxy"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Warm the pull-through mirror from a build plan.")
    ap.add_argument("plan", help="build plan YAML")
    ap.add_argument("--mirror", default="http://127.0.0.1:5010",
                    help="mirror base URL (default http://127.0.0.1:5010)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="warm only the first N images")
    ap.add_argument("--store-path", default=None,
                    help="FSx registry blob-store root used to SKIP already-cached "
                         "blobs (default: $XRLENV_MIRROR_REGISTRY_STORAGE or "
                         "/fsx/home/$USER/xrlenv-registry/proxy)")
    ap.add_argument("--no-skip", action="store_true",
                    help="re-stream every blob even if already cached (repair mode)")
    args = ap.parse_args(argv)

    store: str | None = None
    if not args.no_skip:
        store = args.store_path or _default_store()
        if not os.path.isdir(store):
            print(f"==> note: store path {store!r} not found — skip-existing "
                  f"disabled (every blob will be re-streamed). Pass --store-path "
                  f"or set XRLENV_MIRROR_REGISTRY_STORAGE to enable resume.", flush=True)
            store = None

    refs = _extract_refs(args.plan, args.limit)
    total = len(refs)
    skip_note = f"skip-existing via {store}" if store else "no skip (re-stream all)"
    print(f"==> warming {total} image(s) through {args.mirror} "
          f"(concurrency={args.concurrency}, no local extraction, {skip_note})", flush=True)
    if total == 0:
        return 0

    is_tty = sys.stdout.isatty()
    checkpoint = max(1, total // 40)  # ~40 lines when redirected
    start = time.time()
    done = ok = cached = 0
    nbytes = 0
    fails: list[tuple[str, str]] = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(warm_one, args.mirror, r, store): r for r in refs}
        for fut in cf.as_completed(futs):
            ref, status, by, _n_fetched, fully_cached = fut.result()
            done += 1
            if status == "ok":
                ok += 1
                nbytes += by
                if fully_cached:
                    cached += 1
            else:
                fails.append((ref, status))
            line = _progress(done, total, ok, cached, len(fails), nbytes, start)
            if is_tty:
                print("\r\033[K" + line, end="", flush=True)
            elif done % checkpoint == 0 or done == total:
                print(line, flush=True)
    if is_tty:
        print()  # close the live line

    print(f"==> done in {_fmt_dur(time.time() - start)}: "
          f"{ok}/{total} warmed ({cached} already cached, "
          f"{nbytes / 1e9:.1f} GB newly fetched), {len(fails)} failed")
    if fails:
        print("==> failures (reason  image_ref):")
        for ref, status in fails[:50]:
            print(f"    {status:10s} {ref}")
        if len(fails) > 50:
            print(f"    ... and {len(fails) - 50} more")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
