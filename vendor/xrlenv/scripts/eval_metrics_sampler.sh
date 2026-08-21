#!/usr/bin/env bash
# eval_metrics_sampler.sh — sample the cluster during a benchmark run so we can
# decide Phase C (size-aware / repo-affinity scheduling) from REAL data instead
# of guessing. Captures the dimensions the admin panel doesn't log over time:
#
#   per worker:  free disk on /opt/sagemaker, running containers, disk %util
#   mirror:      registry CPU%, blob requests/s served, FSx store size
#
# Phase C is mainly about node *disk pressure* during image extraction. If, with
# the mirror + the I/O-aware pull throttle live, no node's free disk approaches
# the eviction floor and %util isn't pegged, Phase C's hard disk-guard is low
# priority. If a node still trends toward full or two 15-20 GB images collide,
# that's the signal to implement the spread.
#
# Usage (run on a box that can ssh the cluster, e.g. this login node):
#   bash scripts/eval_metrics_sampler.sh --interval 15 --csv eval_metrics.csv
# Ctrl-C to stop. Override targets via env:
#   NODES="ip-a ip-b ip-c"  MIRROR_HOST=ip-cp  REGISTRY_STORE=/path/to/data
set -uo pipefail

INTERVAL=15
CSV=""
NODES="${NODES:-node-host node-host node-host}"
MIRROR_HOST="${MIRROR_HOST:-node-host}"
REGISTRY_STORE="${REGISTRY_STORE:-/path/to/xrlenv-registry/proxy}"
while [ $# -gt 0 ]; do
    case "$1" in
        --interval) INTERVAL="$2"; shift 2 ;;
        --csv) CSV="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Per-node snapshot computed ON the node: "free_gb running util_pct".
# %util = delta(io_ticks ms)/1000ms over a 1s window on the data-root device.
node_snapshot() {
    ssh -o ConnectTimeout=8 -o BatchMode=yes "$1" 'bash -s' <<'RS' 2>/dev/null
dev=$(findmnt -no SOURCE --target /opt/sagemaker 2>/dev/null | xargs -r basename)
free=$(df -BG --output=avail /opt/sagemaker 2>/dev/null | tail -1 | tr -dc 0-9)
run=$(timeout 10 docker ps -q 2>/dev/null | wc -l)
a=$(awk '{print $10}' "/sys/block/$dev/stat" 2>/dev/null)
sleep 1
b=$(awk '{print $10}' "/sys/block/$dev/stat" 2>/dev/null)
util=$(( ( ${b:-0} - ${a:-0} ) / 10 ))
echo "${free:-NA} ${run:-NA} ${util:-NA}"
RS
}

# Mirror snapshot: "cpu_pct reqs_in_interval fsx_gb".
mirror_snapshot() {
    ssh -o ConnectTimeout=8 -o BatchMode=yes "$MIRROR_HOST" "
cpu=\$(docker stats --no-stream --format '{{.CPUPerc}}' xrlenv-registry-proxy 2>/dev/null | tr -d '%')
reqs=\$(docker logs --since ${INTERVAL}s xrlenv-registry-proxy 2>&1 | grep -c 'blobs/sha256')
fsx=\$(du -sBG '$REGISTRY_STORE' 2>/dev/null | cut -f1 | tr -dc 0-9)
echo \"\${cpu:-NA} \${reqs:-0} \${fsx:-NA}\"
" 2>/dev/null
}

hdr="ts"
for n in $NODES; do hdr="$hdr,${n}_freeGB,${n}_run,${n}_util"; done
hdr="$hdr,mirror_cpu,mirror_req_per_s,fsx_GB"
echo "# $hdr"
[ -n "$CSV" ] && echo "$hdr" > "$CSV"

while true; do
    ts=$(date +%H:%M:%S)
    row="$ts"; human="$ts |"
    for n in $NODES; do
        read -r free run util <<<"$(node_snapshot "$n")"
        row="$row,${free:-NA},${run:-NA},${util:-NA}"
        human="$human ${n##*-}: ${free:-?}G ${run:-?}r ${util:-?}%io |"
    done
    read -r mcpu mreqs fsx <<<"$(mirror_snapshot)"
    rps=0
    [ "${mreqs:-NA}" != "NA" ] && rps=$(( mreqs / INTERVAL ))
    row="$row,${mcpu:-NA},${rps},${fsx:-NA}"
    human="$human mirror ${mcpu:-?}%cpu ${rps}req/s ${fsx:-?}G"
    echo "$human"
    [ -n "$CSV" ] && echo "$row" >> "$CSV"
    sleep "$INTERVAL"
done
