#!/usr/bin/env sh
# echo_bench in-sandbox grader.
#
# Uploaded by EchoBenchInstanceResolver to /opt/xrlenv/run-echo-tests.sh
# at reward time (D12 stage 1 — the agent never sees this script during
# step()). Runs after the agent's step() loop returns done=True.
#
# Reads the target string + the agent's submitted output from the
# adapter-written files at /tmp/echo_bench/{target,output}.txt and
# emits a single float to stdout (1.0 for byte-for-byte match,
# 0.0 otherwise). The platform's stdout_float reward parser turns
# that into the rollout's final_reward.
#
# Uses /bin/sh + cmp instead of bash + a Python script: minimum
# moving parts, runs on alpine and any python:slim base unchanged.

set -eu

STATE_DIR=/tmp/echo_bench
TARGET="${STATE_DIR}/target.txt"
OUTPUT="${STATE_DIR}/output.txt"

if [ ! -f "$TARGET" ]; then
    echo "echo_bench grader: target file missing at $TARGET" >&2
    echo "0.0"
    exit 0
fi
if [ ! -f "$OUTPUT" ]; then
    echo "echo_bench grader: output file missing at $OUTPUT" >&2
    echo "0.0"
    exit 0
fi

if cmp -s "$TARGET" "$OUTPUT"; then
    echo "1.0"
else
    echo "0.0"
fi
