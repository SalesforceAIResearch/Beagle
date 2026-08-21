#!/usr/bin/env bash
# A1 / D20 (P1.2) — build-once-ship-many image distribution recipe.
#
# Operator-facing recipe for the deployment topology where:
#
#   - One "builder" host runs `build-task-images.sh` and produces
#     the per-task images locally (`<bench>/<task>:0.1`).
#   - Multiple "receiver" nodes need the same bytes but cannot
#     reach a registry (corporate firewall, air-gapped cluster,
#     no shared registry yet).
#
# The script `docker save`s each tag on the builder, scp's the
# tarball to each receiver, and `docker load`s it. Every node ends
# up with byte-identical bytes (content-addressed save+load is
# faithful), but no registry digest exists — the corresponding
# manifest must declare ``image_pin_mode: per_node_local`` so the
# catalog skips central pinning.
#
# This is the minimum-viable shell-script form per Fork 4 of the
# P1.2.a design discussion. A polished CLI wrapper
# (`xrlenv images ship ...`) lands in P1.5.
#
# Usage:
#   ./deploy/ship-images.sh --tags '<bench>/<task1>:0.1 <bench>/<task2>:0.1' \
#                           --receivers user@host1.example.com,user@host2.example.com
#
# Optional flags:
#   --tag-file <path>        Read tags from a file (one per line) instead
#                            of --tags.
#   --concurrency <n>        Parallel ship jobs (default 4; positive int).
#   --tarball-dir <dir>      Where to stage the .tar files on the
#                            builder (default $TMPDIR or /tmp).
#   --dry-run                Print the ssh/scp commands without
#                            running them.
#
# Audit response (L1 hardening, 2026-05-02): all command
# execution goes through ``run_cmd`` which takes argv as an array
# and never invokes ``eval``. Operator-supplied tags / paths /
# receivers can therefore contain shell metacharacters without
# breaking quoting or executing unintended shell. A ``%q``-quoted
# remote command is the one place where shell expansion is
# necessary (the ssh remote runs in a shell on the receiver), and
# the values that flow into it are first sanitised through
# ``printf %q`` before composition.

set -euo pipefail

err()  { printf 'ship-images: %s\n' "$*" >&2; }
die()  { err "$@"; exit 1; }
info() { printf 'ship-images: %s\n' "$*"; }

TAGS=""
TAG_FILE=""
RECEIVERS=""
CONCURRENCY=4
TARBALL_DIR="${TMPDIR:-/tmp}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tags)        TAGS="$2"; shift 2 ;;
        --tag-file)    TAG_FILE="$2"; shift 2 ;;
        --receivers)   RECEIVERS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --tarball-dir) TARBALL_DIR="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,40p' "$0" | sed -e 's/^# \{0,1\}//'
            exit 0 ;;
        *)             die "unknown flag: $1" ;;
    esac
done

# ── Input validation ──────────────────────────────────────────────────────────

[[ -n "$RECEIVERS" ]] || die "--receivers is required (comma-separated user@host list)"
if [[ -z "$TAGS" && -z "$TAG_FILE" ]]; then
    die "must specify --tags or --tag-file"
fi
if [[ -n "$TAG_FILE" ]]; then
    [[ -r "$TAG_FILE" ]] || die "--tag-file $TAG_FILE not readable"
    TAGS="$(tr '\n' ' ' < "$TAG_FILE")"
fi

# Concurrency: positive integer only. Reject empty, non-numeric,
# negative, zero. Catches typos like ``--concurrency abc`` /
# ``--concurrency -1`` at boot rather than producing weird parallel
# behavior.
if ! [[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    die "--concurrency must be a positive integer; got $(printf '%q' "$CONCURRENCY")"
fi

mkdir -p "$TARBALL_DIR"

# ── Run helper ────────────────────────────────────────────────────────────────
#
# Takes argv as positional parameters and either prints (dry-run)
# or invokes them directly via ``"$@"`` — never via ``eval``. The
# dry-run output uses ``%q`` so the printed command is
# copy-pasteable + accurately shows what would have run.

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        local quoted=""
        local arg
        for arg in "$@"; do
            quoted+=" $(printf '%q' "$arg")"
        done
        printf '+%s\n' "$quoted"
    else
        "$@"
    fi
}

# ── Save phase: one tarball per tag on the builder ────────────────────────────

declare -a TAG_LIST
read -r -a TAG_LIST <<< "$TAGS"
[[ ${#TAG_LIST[@]} -gt 0 ]] || die "no tags to ship"

declare -a TARBALLS
for tag in "${TAG_LIST[@]}"; do
    [[ -n "$tag" ]] || continue  # skip stray empty entries from extra spaces
    safe="$(printf '%s' "$tag" | tr '/:' '__')"
    tarball="$TARBALL_DIR/$safe.tar"
    info "saving $tag -> $tarball"
    run_cmd docker save -o "$tarball" "$tag"
    TARBALLS+=("$tarball")
done

# ── Ship phase: scp + docker load on each receiver ────────────────────────────
#
# Parallelised across receivers up to --concurrency. Per-receiver
# work is a closed sequence of save/scp/ssh; no shared state across
# receivers other than the tarball read-only inputs.

declare -a RECEIVER_LIST
IFS=',' read -r -a RECEIVER_LIST <<< "$RECEIVERS"

# Drop empty / whitespace-only entries (e.g., trailing comma in
# operator's --receivers list). Validating after the split is the
# right place — comma-stripped tokens are what we'll actually use.
declare -a RECEIVER_CLEAN
for receiver in "${RECEIVER_LIST[@]}"; do
    # Trim whitespace.
    receiver="${receiver#"${receiver%%[![:space:]]*}"}"
    receiver="${receiver%"${receiver##*[![:space:]]}"}"
    [[ -n "$receiver" ]] || continue
    RECEIVER_CLEAN+=("$receiver")
done
[[ ${#RECEIVER_CLEAN[@]} -gt 0 ]] || die "--receivers list is empty after parsing"

ship_one_receiver() {
    local receiver="$1"
    info "shipping ${#TARBALLS[@]} tarball(s) to $receiver"
    for tarball in "${TARBALLS[@]}"; do
        local remote="/tmp/$(basename "$tarball")"
        run_cmd scp -q "$tarball" "$receiver:$remote"
        # The ssh remote runs in a shell on the receiver, so
        # composition into a single command-string IS necessary.
        # Quote ``$remote`` with %q so receiver-side shell sees the
        # literal path even if it contains shell metacharacters.
        local remote_q
        remote_q="$(printf '%q' "$remote")"
        run_cmd ssh "$receiver" "docker load -i $remote_q && rm -f $remote_q"
    done
    info "done shipping to $receiver"
}

# Bounded parallelism via batched background jobs.
#
# We launch up to $CONCURRENCY workers, ``wait`` for the whole
# batch, then start the next batch. Not as tight as ``wait -n``
# (which fires the next worker as soon as any one finishes), but
# ``wait -n`` requires bash 4.3+ — and macOS ships 3.2 by default,
# which would have made this script unusable as a builder-side
# tool on dev laptops. For our use case (small N receivers, roughly
# uniform work per receiver), batch-wait is close to optimal.

batch_pids=()
for receiver in "${RECEIVER_CLEAN[@]}"; do
    ship_one_receiver "$receiver" &
    batch_pids+=("$!")
    if [[ "${#batch_pids[@]}" -ge "$CONCURRENCY" ]]; then
        wait "${batch_pids[@]}"
        batch_pids=()
    fi
done
if [[ "${#batch_pids[@]}" -gt 0 ]]; then
    wait "${batch_pids[@]}"
fi

# ── Cleanup phase: drop the local tarballs ────────────────────────────────────

if [[ "$DRY_RUN" -eq 0 ]]; then
    for tarball in "${TARBALLS[@]}"; do
        rm -f "$tarball"
    done
fi
info "ship-images: done — ${#TAG_LIST[@]} tag(s) shipped to ${#RECEIVER_CLEAN[@]} receiver(s)"
