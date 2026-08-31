#!/usr/bin/env bash
# _preflight.sh — colored log helpers + required-env validation for the
# bootstrap scripts. Sourced *before* package installs by
# bootstrap-{aws,gcp}.sh so a missing XRLENV_CONTROL_PLANE / XRLENV_NODE_ID
# fails loud in red before we download 77MB of dnf/apt packages we'd then
# have to roll back. Also re-sourced by bootstrap-common.sh — the sentinel
# below makes that a no-op.
#
# Inputs (must be set in the parent shell before sourcing):
#   XRLENV_CONTROL_PLANE  e.g. "control.example.com:50051"
#   XRLENV_NODE_ID        stable identifier (cloud bootstrap may auto-fill
#                         this from the metadata service before sourcing)

if [[ "${_XRLENV_PREFLIGHT_SOURCED:-}" == "1" ]]; then
    return 0
fi
_XRLENV_PREFLIGHT_SOURCED=1

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    _C_RED=$'\033[31m'; _C_YELLOW=$'\033[33m'; _C_GREEN=$'\033[32m'
    _C_BOLD=$'\033[1m'; _C_RESET=$'\033[0m'
else
    _C_RED=""; _C_YELLOW=""; _C_GREEN=""; _C_BOLD=""; _C_RESET=""
fi

log()  { printf '%s[xrlenv-bootstrap]%s %s\n' "$_C_BOLD" "$_C_RESET" "$*"; }
ok()   { printf '%s[xrlenv-bootstrap] OK%s %s\n' "$_C_GREEN" "$_C_RESET" "$*"; }
warn() { printf '%s[xrlenv-bootstrap] WARN%s %s\n' "$_C_YELLOW" "$_C_RESET" "$*" >&2; }
die()  { printf '%s[xrlenv-bootstrap] ERROR%s %s\n' "$_C_RED" "$_C_RESET" "$*" >&2; exit 2; }

require_env() {
    # require_env VAR_NAME "hint shown when missing"
    local name="$1" hint="$2"
    if [[ -z "${!name:-}" ]]; then
        die "${name} must be set. ${hint}"
    fi
}
