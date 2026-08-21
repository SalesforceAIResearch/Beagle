#!/usr/bin/env bash
# overlay2_ab_test.sh — measure (and optionally switch to) the classic overlay2
# image store vs the containerd image store, to validate the per-node disk
# reclaim and runtime compatibility BEFORE committing the fleet.
#
# Why: the containerd image store keeps TWO copies of every layer under the
# data-root — the compressed blob (content store) AND the unpacked snapshot
# (overlayfs snapshotter) — roughly 2x for poorly-compressible images. Classic
# overlay2 discards the compressed copy after extraction, so it keeps ~1x.
#
# Portable: resolves the data-root from `docker info`; no cloud-specific paths.
# Nothing here is AWS-specific.
#
# Modes:
#   (default) measure  — NON-destructive. Splits the current data-root into the
#                        containerd content store (compressed blobs overlay2
#                        would NOT keep) vs the overlayfs snapshotter (unpacked
#                        layers overlay2 WOULD keep), and prints the estimated
#                        overlay2 footprint + reclaim. No daemon changes.
#   --switch           — DESTRUCTIVE. Backs up daemon.json, disables the
#                        containerd-snapshotter, WIPES the data-root, restarts
#                        Docker, re-pulls a sample, and A/Bs the real footprint
#                        so you can confirm images run. Requires CONFIRM_WIPE=1.
#                        Rollback instructions are printed at the end.
#
# Usage (note: env vars go AFTER sudo, or use the --yes flag — sudo strips
# leading env vars):
#   sudo bash scripts/overlay2_ab_test.sh                    # measure (safe)
#   sudo bash scripts/overlay2_ab_test.sh --switch --yes     # destructive A/B
#   sudo SAMPLE_N=40 bash scripts/overlay2_ab_test.sh --switch --yes
#   sudo SAMPLE_IMAGES="repo/a:1 repo/b:2" bash scripts/overlay2_ab_test.sh --switch --yes
set -euo pipefail

DAEMON_JSON="${DAEMON_JSON:-/etc/docker/daemon.json}"
SAMPLE_N="${SAMPLE_N:-20}"           # re-pull this many images after a switch
SAMPLE_IMAGES="${SAMPLE_IMAGES:-}"   # or pass an explicit space-separated list
LIST_FILE="/tmp/overlay2_ab_images.txt"

_gib() { awk -v b="${1:-0}" 'BEGIN{ printf "%.1f GiB", b/1073741824 }'; }

_dir_bytes() {
    # Total bytes of a directory, 0 if missing. Apparent-ish via du -sb.
    local d="$1"
    if [[ -d "$d" ]]; then du -sb "$d" 2>/dev/null | awk '{print $1}'; else echo 0; fi
}

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker not found on PATH." >&2; exit 1
    fi
}

current_driver() {
    docker info -f '{{.Driver}}' 2>/dev/null || echo "unknown"
}

data_root() {
    docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo ""
}

containerd_root() {
    # When Docker uses the containerd image store, the layer content +
    # snapshots live under the SYSTEM containerd's root, NOT under Docker's
    # data-root. Resolve it: `containerd config dump` → /etc/containerd/
    # config.toml → default. Top-level `root = "..."` (non-indented).
    local r=""
    if command -v containerd >/dev/null 2>&1; then
        r="$(containerd config dump 2>/dev/null \
            | awk -F'=' '/^root[[:space:]]*=/{gsub(/[\047" ]/,"",$2); print $2; exit}')"
    fi
    if [[ -z "$r" && -f /etc/containerd/config.toml ]]; then
        r="$(awk -F'=' '/^root[[:space:]]*=/{gsub(/[\047" ]/,"",$2); print $2; exit}' \
            /etc/containerd/config.toml)"
    fi
    [[ -z "$r" ]] && r="/var/lib/containerd"
    echo "$r"
}

# Echo "content_dir snapshotter_dir base" for whichever root actually holds the
# containerd image store (Docker data-root or the system containerd root).
locate_image_store() {
    local base content snap
    for base in "$@"; do
        [[ -z "$base" ]] && continue
        content="$base/io.containerd.content.v1.content"
        snap="$base/io.containerd.snapshotter.v1.overlayfs"
        if [[ -d "$content" || -d "$snap" ]]; then
            echo "$content $snap $base"
            return 0
        fi
    done
    echo "  "  # nothing found
}

measure() {
    local droot croot content snap base
    droot="$(data_root)"
    croot="$(containerd_root)"
    if [[ -z "$droot" ]]; then
        echo "ERROR: could not resolve Docker data-root (is the daemon up?)." >&2
        exit 1
    fi
    echo "==> Docker storage driver: $(current_driver)"
    echo "==> Docker data-root:   $droot"
    echo "==> containerd root:    $croot"
    echo
    echo "==> docker system df:"
    docker system df || true
    echo

    # The containerd image store may live under EITHER root — find it.
    read -r content snap base < <(locate_image_store "$droot" "$croot")

    echo "==> Data-root breakdown (du -sh of the big subdirs):"
    du -sh "$droot"/* 2>/dev/null | sort -h | tail -n 6 || true
    if [[ -n "$base" && "$base" != "$droot" ]]; then
        echo "==> containerd-root breakdown ($base):"
        du -sh "$base"/* 2>/dev/null | sort -h | tail -n 6 || true
    fi
    echo

    if [[ -z "$base" ]]; then
        echo "NOTE: no containerd content/snapshotter dirs found under either"
        echo "      root — this host is most likely ALREADY on classic overlay2."
        echo "      Current image footprint ~ docker system df 'SIZE' above."
        return 0
    fi

    local cbytes sbytes img_total est_overlay2 reclaim pct
    cbytes="$(_dir_bytes "$content")"
    sbytes="$(_dir_bytes "$snap")"
    img_total=$(( cbytes + sbytes ))
    # overlay2 keeps the unpacked layers (≈ snapshotter today) and drops the
    # compressed content store. Estimate = snapshotter; reclaim = content store.
    est_overlay2="$sbytes"
    reclaim="$cbytes"
    pct=$(awk -v r="$reclaim" -v t="$img_total" \
        'BEGIN{ if (t>0) printf "%.0f", 100*r/t; else print 0 }')

    echo "==> Estimated overlay2 reclaim (NON-destructive), store found under $base:"
    echo "    containerd content store (compressed, dropped):  $(_gib "$cbytes")"
    echo "    overlayfs snapshotter   (unpacked,  kept):       $(_gib "$sbytes")"
    echo "    -------------------------------------------------------------"
    echo "    image store total (content + snapshots):         $(_gib "$img_total")"
    echo "    estimated overlay2 footprint (~kept):            $(_gib "$est_overlay2")"
    echo "    estimated reclaim by switching:                  $(_gib "$reclaim") (~${pct}%)"
    echo
    echo "    (Estimate only. Run --switch (see --help) to validate the real"
    echo "     footprint AND that your images still run on overlay2.)"
}

backup_daemon_json() {
    if [[ -f "$DAEMON_JSON" ]]; then
        local bak="${DAEMON_JSON}.bak.$(date +%Y%m%d-%H%M%S)"
        cp -a "$DAEMON_JSON" "$bak"
        echo "==> Backed up $DAEMON_JSON -> $bak"
        echo "$bak"
    else
        echo ""
    fi
}

disable_containerd_snapshotter() {
    # Merge-edit daemon.json: features.containerd-snapshotter = false. Mirrors
    # the merge approach in set_docker_data_root.sh so other keys are kept.
    python3 - "$DAEMON_JSON" <<'PY'
import json, os, sys
path = sys.argv[1]
cfg = {}
if os.path.exists(path):
    with open(path) as fh:
        txt = fh.read().strip()
        cfg = json.loads(txt) if txt else {}
cfg.setdefault("features", {})["containerd-snapshotter"] = False
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")
print(f"set features.containerd-snapshotter=false in {path}")
PY
}

switch_to_overlay2() {
    if [[ "${CONFIRM_WIPE:-0}" != "1" ]]; then
        echo "REFUSING: --switch is DESTRUCTIVE. It wipes the Docker data-root" >&2
        echo "  AND the system containerd image store, and restarts containerd —" >&2
        echo "  which kills ALL containers on this node (incl. node sidecars like" >&2
        echo "  the EFA exporter, and anything kubelet runs). Cordon/drain the" >&2
        echo "  node first. Then re-run with --yes (or 'sudo CONFIRM_WIPE=1 ...')." >&2
        exit 2
    fi
    local root cstore_content cstore_snap cstore_base
    root="$(data_root)"
    [[ -n "$root" && -d "$root" ]] || { echo "ERROR: no data-root." >&2; exit 1; }
    read -r cstore_content cstore_snap cstore_base \
        < <(locate_image_store "$root" "$(containerd_root)")

    echo "### PHASE A (current = $(current_driver)) ###"
    measure
    echo

    # Snapshot the present image list so we can re-pull the same set after.
    if [[ -n "$SAMPLE_IMAGES" ]]; then
        printf '%s\n' $SAMPLE_IMAGES > "$LIST_FILE"
    else
        docker images --format '{{.Repository}}:{{.Tag}}' \
            | grep -v '<none>' | head -n "$SAMPLE_N" > "$LIST_FILE" || true
    fi
    echo "==> Will re-pull $(wc -l < "$LIST_FILE") images after the switch (from $LIST_FILE)."
    echo

    local bak
    bak="$(backup_daemon_json)"
    disable_containerd_snapshotter

    echo "==> Stopping Docker + containerd, then WIPING image stores..."
    systemctl stop docker.socket docker 2>/dev/null || systemctl stop docker || true
    # Wipe the Docker data-root (overlay2 will recreate it).
    rm -rf "${root:?}" && mkdir -p "$root"
    # Wipe the orphaned containerd image store (the compressed blobs +
    # snapshots that overlay2 won't use). Stop containerd so nothing is mounted.
    if [[ -n "$cstore_base" ]]; then
        echo "    wiping containerd image store under: $cstore_base"
        systemctl stop containerd 2>/dev/null || true
        [[ -n "$cstore_content" ]] && rm -rf "${cstore_content:?}"
        [[ -n "$cstore_snap" ]] && rm -rf "${cstore_snap:?}"
        systemctl start containerd 2>/dev/null || true
        sleep 2
    fi
    systemctl start docker
    sleep 3

    local drv
    drv="$(current_driver)"
    echo "==> New storage driver: $drv"
    if [[ "$drv" != "overlay2" ]]; then
        echo "WARNING: driver is '$drv', expected 'overlay2'. Check $DAEMON_JSON." >&2
    fi

    echo "==> Re-pulling sample (validates registry auth + that images run)..."
    local ok=0 fail=0
    while IFS= read -r ref; do
        [[ -z "$ref" ]] && continue
        if docker pull "$ref" >/dev/null 2>&1; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1)); echo "    pull FAILED: $ref" >&2
        fi
    done < "$LIST_FILE"
    echo "==> Re-pull: $ok ok, $fail failed."

    # Smoke: actually RUN one image to confirm overlay2 mounts it.
    local first
    first="$(head -n1 "$LIST_FILE" 2>/dev/null || true)"
    if [[ -n "$first" ]]; then
        echo "==> Runtime smoke: docker run --rm $first true"
        if docker run --rm "$first" true 2>/dev/null \
            || docker run --rm "$first" /bin/true 2>/dev/null; then
            echo "    OK — image runs on overlay2."
        else
            echo "    NOTE: smoke run returned non-zero (image may have no shell);"
            echo "          not necessarily a driver problem." >&2
        fi
    fi

    echo
    echo "### PHASE B (now = $(current_driver)) ###"
    measure
    echo
    echo "============================================================"
    echo "ROLLBACK to the containerd image store, if needed:"
    if [[ -n "$bak" ]]; then
        echo "  sudo cp -a '$bak' '$DAEMON_JSON'"
    else
        echo "  # remove features.containerd-snapshotter from $DAEMON_JSON"
    fi
    echo "  sudo systemctl stop docker.socket docker"
    echo "  sudo rm -rf '${root}'/* && sudo systemctl start docker"
    echo "  # then re-pull/rebuild your images"
    echo "============================================================"
}

main() {
    require_docker
    case "${1:-measure}" in
        measure|"") measure ;;
        --switch|switch)
            # `--yes` survives sudo (env vars don't); CONFIRM_WIPE=1 also works.
            [[ "${2:-}" == "--yes" ]] && CONFIRM_WIPE=1
            switch_to_overlay2 ;;
        -h|--help)
            sed -n '2,45p' "$0" ;;
        *) echo "unknown mode: $1 (use 'measure' or '--switch')" >&2; exit 1 ;;
    esac
}

main "$@"
