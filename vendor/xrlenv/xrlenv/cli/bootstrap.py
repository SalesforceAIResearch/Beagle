"""``xrlenv bootstrap`` subcommand (B8.1, P1.x slice 5).

Replaces ``deploy/bootstrap-{gcp,aws}.sh`` with a Python entry point.
The shell scripts remain as thin wrappers (3-line scripts that
``exec xrlenv bootstrap --target {gcp,aws} "$@"``) so existing
operator muscle memory + cloud-init snippets keep working.

What ``xrlenv bootstrap`` does, in order:

1. **Validate** the operator-set knobs (``--control-plane``,
   ``--node-id``). On GCP / AWS the node-id may be auto-detected
   from the cloud's metadata service if not passed.
2. **Detect the OS** by parsing ``/etc/os-release``. Supports
   Debian-family (apt) and RHEL-family (dnf) installs. An operator
   override is available via ``--target-os {amzn,ubuntu,debian,...}``
   for edge cases (e.g. running on a custom AMI whose os-release is
   missing or wrong).
3. **Install Docker** via the right package manager for the OS. On
   GCP this is always apt (we use the upstream Docker repo because
   GCP Debian/Ubuntu ships ancient docker.io). On AWS this branches
   between dnf (AL2023/RHEL/Fedora) and apt (Ubuntu).
4. **Ensure the system user + directory layout** the systemd unit
   expects: ``/opt/xrlenv``, ``/etc/xrlenv``, ``/var/lib/xrlenv``,
   ``/var/cache/xrlenv`` (plus the harbor task cache + build-context
   cache subdirs).
5. **Install Python 3.12+** if not already present. Tries operator-
   pinned ``XRLENV_PYTHON`` first, then native distro packages
   (``dnf install python3.12`` on AL2023, ``apt install python3.12``
   on Ubuntu 24.04+), then falls back to ``uv python install`` for
   distros that don't ship 3.12 natively (Ubuntu 22.04, Debian 12).
6. **Build the venv + install xrlenv** from one of three sources
   (priority order):
   - ``--xrlenv-wheel /path`` — production install from a local wheel.
   - ``--xrlenv-repo /path`` — checkout dir, installed non-editable
     (so the systemd unit's ``ProtectHome=read-only`` doesn't break
     module resolution).
   - PyPI fallback (``pip install xrlenv==<version>``).
7. **Drop the systemd unit** at ``/etc/systemd/system/xrlenv-node.
   service`` plus a ``node.env`` EnvironmentFile and a node-token
   drop-in (mode 0600, root-owned) when ``XRLENV_NODE_TOKEN`` is set.
   ``systemctl daemon-reload && systemctl enable --now`` brings the
   daemon up.
8. **Add the operator's interactive user to the docker group** so
   they can run per-VM helpers (``populate-harbor-cache.sh`` etc.)
   without sudo. Opt-out via ``--skip-operator-docker-group``.

Two modes:

- ``--dry-run`` prints the planned sequence + the commands each
  step would run, then exits. Run this first on a new VM to see
  what the bootstrap is about to do without actually touching the
  system. (Read the output before re-running without --dry-run.)
- Default (live) runs each step. Steps are idempotent — re-running
  on an already-bootstrapped node is safe and fast (most steps
  short-circuit via ``skip_if`` predicates).

Operator UX: this module is the slice's *implementation*; the
operator-facing knobs are the CLI subcommand registered in
``xrlenv/cli/__main__.py``. See
``docs/deploy/multi_node_deployment/runbook.md`` for the full
walkthrough.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal, TextIO

LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants — kept in lock-step with deploy/bootstrap-common.sh so the
# two entry points install into the same paths.
# ──────────────────────────────────────────────────────────────────────────────

INSTALL_ROOT = Path("/opt/xrlenv")
ETC_DIR = Path("/etc/xrlenv")
SCRATCH_DIR = Path("/var/lib/xrlenv")
CACHE_DIR = Path("/var/cache/xrlenv")
SYSTEMD_UNIT = Path("/etc/systemd/system/xrlenv-node.service")
RUNTIME_USER = "xrlenv"

Target = Literal["gcp", "aws", "linux-generic"]
DistroFamily = Literal["debian", "rhel"]

# Cloud metadata endpoints. Both are best-effort — failure falls
# through to "operator must pass --node-id explicitly".
GCP_METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/name"
)
AWS_IMDSV2_TOKEN_URL = "http://169.254.169.254/latest/api/token"
AWS_IMDSV2_INSTANCE_ID_URL = (
    "http://169.254.169.254/latest/meta-data/instance-id"
)


# ──────────────────────────────────────────────────────────────────────────────
# OS detection
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OSInfo:
    """Resolved OS identity. ``id`` matches ``/etc/os-release``'s
    ``ID`` field (``ubuntu``, ``debian``, ``amzn``, ``rhel``,
    ``fedora``). ``family`` is the higher-level package-manager
    bucket: ``debian`` → apt, ``rhel`` → dnf."""

    id: str
    family: DistroFamily
    version_id: str | None
    version_codename: str | None


def _parse_os_release(text: str) -> dict[str, str]:
    """Parse a KEY=VALUE-formatted os-release file into a dict.

    Strips quotes around values. Skips blank lines and comments.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Strip surrounding quotes, single or double.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        out[key.strip()] = value.strip()
    return out


def detect_os(override: str | None = None) -> OSInfo:
    """Probe the host's OS by parsing ``/etc/os-release``.

    Returns an :class:`OSInfo` whose ``family`` chooses the package
    manager (``apt`` vs ``dnf``). Raises :class:`RuntimeError` if
    the OS is one we don't know how to bootstrap.

    ``override`` is the operator's ``--target-os`` value — it lets
    them name an ``ID`` explicitly when ``/etc/os-release`` is
    missing or wrong (custom AMIs, hardened images).
    """
    if override is not None:
        os_id = override.strip().lower()
        version_id = None
        codename = None
    else:
        release_path = Path("/etc/os-release")
        if not release_path.exists():
            raise RuntimeError(
                "cannot detect OS: /etc/os-release missing. "
                "Pass --target-os {amzn,rhel,fedora,ubuntu,debian} explicitly.",
            )
        fields = _parse_os_release(release_path.read_text(encoding="utf-8"))
        os_id = fields.get("ID", "").strip().lower()
        version_id = fields.get("VERSION_ID") or None
        codename = fields.get("VERSION_CODENAME") or None
        if not os_id:
            raise RuntimeError(
                "cannot detect OS: /etc/os-release has no ID= field. "
                "Pass --target-os explicitly.",
            )
    if os_id in ("amzn", "rhel", "fedora", "centos", "rocky", "almalinux"):
        family: DistroFamily = "rhel"
    elif os_id in ("ubuntu", "debian"):
        family = "debian"
    else:
        raise RuntimeError(
            f"unsupported OS ID={os_id!r}. Supported families: "
            "Debian/Ubuntu (apt) and Amazon Linux 2023 / RHEL / Fedora (dnf). "
            "Pass --target-os to override the probe if /etc/os-release is wrong.",
        )
    return OSInfo(
        id=os_id, family=family,
        version_id=version_id, version_codename=codename,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cloud metadata auto-detect
# ──────────────────────────────────────────────────────────────────────────────


def autodetect_node_id_gcp(timeout_s: float = 1.0) -> str | None:
    """Hit the GCP metadata service and return ``gcp-<instance-name>``.

    Returns ``None`` on any failure (DNS, connection refused, 404,
    timeout, bad response). The metadata host requires the
    ``Metadata-Flavor: Google`` header to honor the request.
    """
    req = urllib.request.Request(
        GCP_METADATA_URL,
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            name = resp.read().decode("utf-8").strip()
    except (TimeoutError, urllib.error.URLError, OSError):
        return None
    if not name:
        return None
    return f"gcp-{name}"


def autodetect_node_id_aws(timeout_s: float = 1.0) -> str | None:
    """IMDSv2 two-step: PUT for a token, GET for the instance id.

    IMDSv2 explicitly refuses unauthenticated GETs (the v1 escape
    hatch is widely disabled for security), so we always do the
    token dance even though the bash script's older form sometimes
    skipped it.
    """
    try:
        token_req = urllib.request.Request(
            AWS_IMDSV2_TOKEN_URL,
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(token_req, timeout=timeout_s) as resp:
            token = resp.read().decode("utf-8").strip()
        if not token:
            return None
        inst_req = urllib.request.Request(
            AWS_IMDSV2_INSTANCE_ID_URL,
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(inst_req, timeout=timeout_s) as resp:
            instance_id = resp.read().decode("utf-8").strip()
    except (TimeoutError, urllib.error.URLError, OSError):
        return None
    if not instance_id:
        return None
    return f"aws-{instance_id}"


# ──────────────────────────────────────────────────────────────────────────────
# BootstrapConfig + Step
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BootstrapConfig:
    """Resolved, validated input to the bootstrap plan.

    Operator-set values arrive via CLI flags or env vars. After
    :func:`build_config`, every field below is either a concrete
    value or ``None`` with documented meaning.
    """

    target: Target
    control_plane: str
    node_id: str
    node_token: str | None
    dockerhub_user: str | None
    dockerhub_token: str | None
    xrlenv_wheel: Path | None
    xrlenv_repo: Path | None
    xrlenv_version: str
    runtime_user: str
    install_root: Path
    etc_dir: Path
    cache_dir: Path
    scratch_dir: Path
    systemd_unit: Path
    skip_operator_docker_group: bool
    operator_user: str | None
    os_info: OSInfo


@dataclass
class Step:
    """One install step in the plan.

    A step is a named operation with:

    - ``description`` — single-sentence headline shown in dry-run
      and in the live-mode log line.
    - ``commands`` — list of subprocess argv lists. Each list is run
      via :func:`subprocess.run` with ``check=True``. In dry-run we
      print the joined argv instead of executing.
    - ``skip_if`` — optional predicate. When it returns ``True`` the
      step is logged as skipped and not run. Used for idempotency
      (e.g. "skip the user-add if the xrlenv user already exists").
    - ``python_fn`` — optional in-process callable. When set, runs
      instead of ``commands`` (for steps that are awkward to express
      as a single subprocess call, e.g. writing an EnvironmentFile).
    - ``env`` — process env overrides for ``commands``.
    - ``allow_failure`` — when ``True``, a non-zero exit from
      ``commands`` (or an exception from ``python_fn``) is logged as
      "soft-failed; expected" and the plan continues with the next
      step. Used for fallback chains where a later step picks up the
      slack (e.g. ``apt-install-python312`` is expected to fail on
      Ubuntu 22.04 / Debian; the ``uv-python-fallback`` step handles
      that case).
    """

    name: str
    description: str
    commands: list[list[str]] = field(default_factory=list)
    skip_if: Callable[[], bool] | None = None
    python_fn: Callable[[], None] | None = None
    env: dict[str, str] | None = None
    allow_failure: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Plan construction
# ──────────────────────────────────────────────────────────────────────────────


def build_config(
    *,
    target: str,
    control_plane: str | None,
    node_id: str | None,
    target_os: str | None,
    xrlenv_wheel: Path | None,
    xrlenv_repo: Path | None,
    xrlenv_version: str | None,
    runtime_user: str | None,
    install_root: Path | None,
    skip_operator_docker_group: bool,
    env: dict[str, str] | None = None,
) -> BootstrapConfig:
    """Resolve operator inputs + env vars into a fully-typed
    :class:`BootstrapConfig`. Performs the same pre-flight checks
    the shell script's ``validate_required_env_for_bootstrap`` runs.

    ``env`` defaults to ``os.environ``; tests pass a synthetic env.
    """
    if target not in ("gcp", "aws", "linux-generic"):
        raise ValueError(
            f"--target must be one of gcp / aws / linux-generic; got {target!r}",
        )
    env_map = env if env is not None else dict(os.environ)
    resolved_target: Target = target  # type: ignore[assignment]

    cp = control_plane or env_map.get("XRLENV_CONTROL_PLANE")
    if not cp:
        raise ValueError(
            "missing control-plane address. Pass --control-plane "
            "host:port or set XRLENV_CONTROL_PLANE.",
        )

    nid = node_id or env_map.get("XRLENV_NODE_ID")
    if not nid:
        # Best-effort auto-detect on cloud targets. linux-generic has
        # no metadata service — the operator must supply --node-id.
        if resolved_target == "gcp":
            nid = autodetect_node_id_gcp()
        elif resolved_target == "aws":
            nid = autodetect_node_id_aws()
    if not nid:
        raise ValueError(
            f"missing node id (target={resolved_target!r}). Pass --node-id "
            "or set XRLENV_NODE_ID; cloud metadata auto-detect failed.",
        )

    os_info = detect_os(target_os)

    wheel_path = xrlenv_wheel
    if wheel_path is None and env_map.get("XRLENV_WHEEL"):
        wheel_path = Path(env_map["XRLENV_WHEEL"]).expanduser()
    if wheel_path is not None and not wheel_path.exists():
        raise ValueError(
            f"--xrlenv-wheel points to a non-existent file: {wheel_path}",
        )

    repo_path = xrlenv_repo
    if repo_path is None and env_map.get("XRLENV_REPO"):
        repo_path = Path(env_map["XRLENV_REPO"]).expanduser()
    if repo_path is not None and not (repo_path / "pyproject.toml").exists():
        raise ValueError(
            f"--xrlenv-repo {repo_path} does not contain pyproject.toml",
        )

    version = (
        xrlenv_version
        or env_map.get("XRLENV_VERSION")
        or "main"
    )

    return BootstrapConfig(
        target=resolved_target,
        control_plane=cp,
        node_id=nid,
        node_token=env_map.get("XRLENV_NODE_TOKEN") or None,
        dockerhub_user=env_map.get("DOCKERHUB_USER") or None,
        dockerhub_token=env_map.get("DOCKERHUB_TOKEN") or None,
        xrlenv_wheel=wheel_path,
        xrlenv_repo=repo_path,
        xrlenv_version=version,
        runtime_user=runtime_user or env_map.get("XRLENV_USER") or RUNTIME_USER,
        install_root=install_root or INSTALL_ROOT,
        etc_dir=ETC_DIR,
        cache_dir=CACHE_DIR,
        scratch_dir=SCRATCH_DIR,
        systemd_unit=SYSTEMD_UNIT,
        skip_operator_docker_group=skip_operator_docker_group,
        operator_user=env_map.get("SUDO_USER") or None,
        os_info=os_info,
    )


def build_plan(config: BootstrapConfig) -> list[Step]:
    """Construct the ordered install plan for ``config``.

    Steps mirror ``deploy/bootstrap-common.sh``'s
    ``bootstrap_xrlenv`` function plus the target-specific Docker
    install prelude that lives in ``bootstrap-{aws,gcp}.sh``.
    """
    steps: list[Step] = []
    steps.extend(_docker_install_steps(config))
    steps.append(Step(
        name="enable-docker",
        description="Enable and start the docker systemd service",
        commands=[["systemctl", "enable", "--now", "docker"]],
        skip_if=_service_already_active("docker"),
    ))
    steps.append(_ensure_user_step(config))
    steps.append(_ensure_directories_step(config))
    steps.extend(_install_python_venv_steps(config))
    steps.append(_write_node_env_step(config))
    steps.append(_install_systemd_unit_step(config))
    if config.node_token:
        steps.append(_install_node_token_dropin_step(config))
    if config.dockerhub_user and config.dockerhub_token:
        steps.append(_install_dockerhub_auth_step(config))
    steps.append(_systemctl_reload_step())
    steps.append(_systemctl_enable_node_step())
    steps.append(_systemctl_restart_node_step())
    if not config.skip_operator_docker_group and config.operator_user:
        steps.append(_ensure_operator_docker_group_step(config))
    # Advisory warning is always added (last) — it short-circuits at
    # runtime when a config.json is in place. Always-on means new
    # operators see the gap immediately on first bootstrap rather
    # than discovering it via a later failed apply.
    steps.append(_warn_no_dockerhub_auth_step(config))
    return steps


def _docker_install_steps(config: BootstrapConfig) -> list[Step]:
    """Docker install branches by OS family + target.

    The bash split was:
    - GCP: always upstream Docker apt repo (handles both Debian + Ubuntu).
    - AWS: dnf install docker on AL2023/RHEL/Fedora; apt install docker.io
      on Ubuntu/Debian.
    - linux-generic: same OS-family branch as AWS.

    The Python port collapses that to: on RHEL family → dnf, on
    Debian family → apt. GCP always uses the upstream apt repo
    (which is the only path tested for GCP Debian/Ubuntu); AWS +
    linux-generic on Debian use the simpler ``apt install
    docker.io`` (matches the shell). Operators wanting the upstream
    apt repo on AWS Ubuntu can drop in their own apt-source ahead
    of running this command.
    """
    out: list[Step] = []
    if config.os_info.family == "rhel":
        out.append(Step(
            name="dnf-install-docker",
            description="Install docker + tar via dnf",
            commands=[["dnf", "install", "-y", "docker", "tar"]],
            skip_if=_binary_present("docker"),
        ))
    elif config.target == "gcp":
        # GCP always uses upstream Docker apt repo. The bash script
        # builds the apt source via a multi-step pipeline; the Python
        # port flattens it but the underlying commands match.
        out.append(Step(
            name="apt-update",
            description="Refresh apt package metadata",
            commands=[["apt-get", "update", "-y"]],
            env={"DEBIAN_FRONTEND": "noninteractive"},
        ))
        out.append(Step(
            name="apt-install-prereqs",
            description="Install Docker apt-source prerequisites (curl + gnupg)",
            commands=[[
                "apt-get", "install", "-y", "--no-install-recommends",
                "ca-certificates", "curl", "gnupg", "tar",
            ]],
            env={"DEBIAN_FRONTEND": "noninteractive"},
        ))
        out.append(Step(
            name="install-docker-apt-key-and-repo",
            description=(
                "Add the upstream Docker apt repo + GPG key, then "
                "apt-get install docker-ce. Branches on /etc/os-release "
                "for the distro path component."
            ),
            python_fn=lambda: _install_docker_upstream_apt(config),
            skip_if=_binary_present("docker"),
        ))
    else:
        # AWS Ubuntu / linux-generic Debian — simpler path.
        out.append(Step(
            name="apt-install-docker",
            description="Install docker.io + tar via apt",
            commands=[
                ["apt-get", "update", "-y"],
                ["apt-get", "install", "-y", "--no-install-recommends",
                 "ca-certificates", "curl", "gnupg", "tar", "docker.io"],
            ],
            env={"DEBIAN_FRONTEND": "noninteractive"},
            skip_if=_binary_present("docker"),
        ))
    return out


def _install_docker_upstream_apt(config: BootstrapConfig) -> None:
    """Replicate the bash one-liner that registers the upstream
    Docker apt repo on Debian / Ubuntu. We invoke each piece with
    subprocess.run so any failure surfaces clearly."""
    distro = config.os_info.id  # "debian" or "ubuntu"
    codename = config.os_info.version_codename or "stable"
    arch = subprocess.check_output(
        ["dpkg", "--print-architecture"],
    ).decode("utf-8").strip()
    subprocess.run(
        ["install", "-m", "0755", "-d", "/etc/apt/keyrings"], check=True,
    )
    # Download the GPG key, dearmor it.
    gpg_url = f"https://download.docker.com/linux/{distro}/gpg"
    keyring = "/etc/apt/keyrings/docker.gpg"
    curl = subprocess.run(
        ["curl", "-fsSL", gpg_url],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["gpg", "--dearmor", "-o", keyring],
        check=True, input=curl.stdout,
    )
    subprocess.run(["chmod", "a+r", keyring], check=True)
    repo = (
        f"deb [arch={arch} signed-by={keyring}] "
        f"https://download.docker.com/linux/{distro} {codename} stable\n"
    )
    Path("/etc/apt/sources.list.d/docker.list").write_text(repo, encoding="utf-8")
    env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    subprocess.run(["apt-get", "update", "-y"], check=True, env=env)
    subprocess.run(
        ["apt-get", "install", "-y", "docker-ce", "docker-ce-cli",
         "containerd.io"],
        check=True, env=env,
    )


def _ensure_user_step(config: BootstrapConfig) -> Step:
    return Step(
        name="ensure-user",
        description=f"Create system user {config.runtime_user!r}",
        commands=[
            ["useradd", "--system", "--shell", "/usr/sbin/nologin",
             "--home-dir", str(config.install_root), config.runtime_user],
            # Best-effort docker-group membership; bash uses `|| log`
            # to swallow failure when docker group doesn't exist yet,
            # but at this point docker is installed so the group should
            # exist.
            ["usermod", "-aG", "docker", config.runtime_user],
        ],
        skip_if=_user_already_exists(config.runtime_user),
    )


def _ensure_directories_step(config: BootstrapConfig) -> Step:
    """Create the install + state + cache directory tree.

    The shell script uses a single ``mkdir -p`` call for all paths;
    we mirror that — ``mkdir -p`` is idempotent so re-runs are
    harmless. Chown / chmod are also idempotent.
    """
    paths = [
        str(config.install_root),
        str(config.etc_dir),
        str(config.scratch_dir),
        str(config.cache_dir),
        str(config.cache_dir / "harbor" / "tasks"),
        str(config.cache_dir / "build-context-cache"),
    ]
    return Step(
        name="ensure-directories",
        description=(
            f"mkdir -p the platform's install/state/cache tree "
            f"({', '.join(paths)})"
        ),
        commands=[
            ["mkdir", "-p", *paths],
            ["chown", "-R",
             f"{config.runtime_user}:{config.runtime_user}",
             str(config.install_root),
             str(config.scratch_dir),
             str(config.cache_dir)],
            ["chmod", "0775", str(config.cache_dir / "harbor"),
             str(config.cache_dir / "harbor" / "tasks")],
        ],
    )


def _install_python_venv_steps(config: BootstrapConfig) -> list[Step]:
    """Build the venv + install xrlenv (wheel / repo / PyPI).

    The bash version probes for python3.12 across distros + falls
    back to uv when the distro doesn't ship 3.12. The Python port
    keeps that exact ordering and resolves the interpreter at venv-
    creation time via :func:`_resolve_python_312` so all four
    candidate sources contribute: operator-pinned ``XRLENV_PYTHON``,
    ``python3.{14,13,12}`` on ``PATH``, native distro install, and
    the uv-managed ``python-build-standalone`` under
    ``$install_root/python``.

    The apt-install-python312 step is marked ``allow_failure=True``
    so a failure on Ubuntu 22.04 / Debian (where python3.12 isn't
    in the apt repos) lets the plan continue to the uv fallback
    rather than aborting the whole bootstrap.
    """
    steps: list[Step] = []
    skip_when_312_resolvable = _python_312_resolvable(config)
    # Step 1: try to install python3.12 via the native package manager.
    if config.os_info.family == "rhel":
        steps.append(Step(
            name="dnf-install-python312",
            description="dnf install -y python3.12 python3.12-pip",
            commands=[
                ["dnf", "install", "-y", "python3.12", "python3.12-pip"],
            ],
            skip_if=skip_when_312_resolvable,
            allow_failure=True,
        ))
    else:
        # Ubuntu 24.04+ ships python3.12; 22.04 / Debian don't. The
        # apt-install attempt is allowed to fail — the uv fallback
        # below picks up the slack.
        steps.append(Step(
            name="apt-install-python312",
            description=(
                "apt-get install -y python3.12 python3.12-venv "
                "(works on Ubuntu 24.04+; failure expected on 22.04 / "
                "Debian — the uv fallback below picks up the slack)"
            ),
            commands=[
                ["apt-get", "install", "-y", "--no-install-recommends",
                 "python3.12", "python3.12-venv"],
            ],
            env={"DEBIAN_FRONTEND": "noninteractive"},
            skip_if=skip_when_312_resolvable,
            allow_failure=True,
        ))
    # Step 2: uv fallback for distros without 3.12 in their repos.
    steps.append(Step(
        name="uv-python-fallback",
        description=(
            "If python3.12 still isn't resolvable (operator XRLENV_PYTHON, "
            "PATH, or distro pkg), install uv + use `uv python install 3.12` "
            "to fetch a portable build under $install_root/python"
        ),
        python_fn=lambda: _install_python_via_uv(config),
        skip_if=skip_when_312_resolvable,
    ))
    # Step 3: build the venv. Resolve the interpreter at runtime via
    # the same probe order the skip predicate uses, so the uv-managed
    # interpreter (which lands under $install_root/python and is NOT
    # on PATH) is picked up correctly.
    steps.append(Step(
        name="create-venv",
        description=(
            f"<resolved-python3.12> -m venv {config.install_root}/.venv"
            " (resolves: $XRLENV_PYTHON → python3.{14,13,12} on PATH → "
            f"{config.install_root}/python/*/bin/python3.12)"
        ),
        python_fn=lambda: _create_venv(config),
        skip_if=_path_exists(config.install_root / ".venv" / "bin" / "python"),
    ))
    # Step 4: install xrlenv into the venv.
    pip = str(config.install_root / ".venv" / "bin" / "pip")
    if config.xrlenv_wheel is not None:
        target = str(config.xrlenv_wheel)
        install_desc = f"pip install {config.xrlenv_wheel}"
    elif config.xrlenv_repo is not None:
        target = str(config.xrlenv_repo)
        install_desc = f"pip install {config.xrlenv_repo} (non-editable)"
    else:
        target = f"xrlenv=={config.xrlenv_version}"
        install_desc = (
            f"pip install xrlenv=={config.xrlenv_version} (PyPI fallback)"
        )
    install_cmd = [pip, "install", target]
    steps.append(Step(
        name="pip-install-xrlenv",
        description=install_desc,
        commands=[
            [pip, "install", "--upgrade", "pip"],
            # Plain install first so deps land on a fresh node…
            install_cmd,
            # …then force-reinstall just the package so a RE-bootstrap
            # actually refreshes the code. The version string is a fixed
            # ``0.0.1`` for the non-editable repo/wheel install, so a plain
            # ``pip install`` of an already-satisfied version no-ops —
            # which silently left re-deployed nodes running stale code.
            # ``--no-deps`` keeps it fast (deps were handled just above).
            [pip, "install", "--force-reinstall", "--no-deps", target],
        ],
    ))
    # Step 5: sanity-check the install actually produced xrlenv-node.
    steps.append(Step(
        name="verify-xrlenv-node",
        description=(
            "Verify xrlenv.node imports cleanly + xrlenv-node CLI "
            "is on the venv PATH"
        ),
        python_fn=lambda: _verify_xrlenv_install(config),
    ))
    return steps


def _write_node_env_step(config: BootstrapConfig) -> Step:
    """Write ``/etc/xrlenv/node.env`` with the platform paths the
    daemon needs to pick up at startup. Matches the heredoc in
    ``bootstrap-common.sh``'s ``install_systemd_unit``."""
    return Step(
        name="write-node-env",
        description=f"Write {config.etc_dir}/node.env (mode 0640)",
        python_fn=lambda: _write_node_env(config),
    )


def _capture_build_sha() -> str:
    """Snapshot ``git rev-parse --short=12 HEAD`` from the repo being installed.

    Written into ``node.env`` so ``buildinfo.build_sha()`` picks it up
    at daemon startup — the installed venv at ``/opt/xrlenv`` is not a
    git checkout, so the ``git rev-parse`` fallback in ``buildinfo.py``
    won't work there.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(_repo_root()), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _write_node_env(config: BootstrapConfig) -> None:
    build_sha = _capture_build_sha()
    # Adaptive (AIMD) image-pull concurrency knobs — kept in lock-step with
    # deploy/bootstrap-common.sh. A single node-local limiter moves between
    # the floor (busy minimum) and ceiling (idle maximum); operators may
    # override any of these by exporting them before bootstrap, else the
    # library defaults (2 / 64 / 16) are stamped.
    pull_floor = os.environ.get("XRLENV_PULL_CONCURRENCY", "2")
    pull_ceiling = os.environ.get("XRLENV_PULL_CONCURRENCY_CEILING", "64")
    pull_initial = os.environ.get("XRLENV_PULL_CONCURRENCY_INITIAL", "16")
    # Eviction headroom caps (GiB). Upper bound on the adaptive reserve
    # (slots x largest_cached_image x safety) so one pathologically large
    # base image can't reserve an unreasonable share of the disk.
    evict_threshold_cap_gb = os.environ.get("XRLENV_EVICT_THRESHOLD_CAP_GB", "50")
    evict_target_cap_gb = os.environ.get("XRLENV_EVICT_TARGET_CAP_GB", "75")
    body = (
        f"XRLENV_CONTROL_PLANE={config.control_plane}\n"
        f"XRLENV_NODE_ID={config.node_id}\n"
        f"XRLENV_BENCHMARK_CACHE={config.cache_dir}/harbor/tasks\n"
        f"XRLENV_BUILD_CONTEXT_CACHE={config.cache_dir}/build-context-cache\n"
        f"XRLENV_BUILD_SHA={build_sha}\n"
        f"XRLENV_PULL_CONCURRENCY={pull_floor}\n"
        f"XRLENV_PULL_CONCURRENCY_CEILING={pull_ceiling}\n"
        f"XRLENV_PULL_CONCURRENCY_INITIAL={pull_initial}\n"
        f"XRLENV_EVICT_THRESHOLD_CAP_GB={evict_threshold_cap_gb}\n"
        f"XRLENV_EVICT_TARGET_CAP_GB={evict_target_cap_gb}\n"
    )
    target = config.etc_dir / "node.env"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    subprocess.run(
        ["chown", f"root:{config.runtime_user}", str(target)], check=True,
    )
    subprocess.run(["chmod", "0640", str(target)], check=True)


def _install_systemd_unit_step(config: BootstrapConfig) -> Step:
    """Install the systemd unit file from ``deploy/systemd/`` to
    ``/etc/systemd/system/xrlenv-node.service``. The source path is
    resolved relative to the repo root via :func:`_repo_root`."""
    return Step(
        name="install-systemd-unit",
        description=f"Install systemd unit to {config.systemd_unit}",
        python_fn=lambda: _install_systemd_unit(config),
    )


def _install_systemd_unit(config: BootstrapConfig) -> None:
    src = _repo_root() / "deploy" / "systemd" / "xrlenv-node.service"
    if not src.exists():
        raise RuntimeError(
            f"systemd unit source not found at {src}. The Python "
            "bootstrap expects the deploy/ directory to be on disk; "
            "this is provided by --xrlenv-repo or by running the "
            "wrapper script from a checkout.",
        )
    subprocess.run(
        ["install", "-m", "0644", str(src), str(config.systemd_unit)],
        check=True,
    )


def _install_node_token_dropin_step(config: BootstrapConfig) -> Step:
    """Write a mode-0600 systemd drop-in carrying
    ``XRLENV_NODE_TOKEN`` so the daemon authenticates to the
    control plane on first connect."""
    return Step(
        name="install-node-token-dropin",
        description=(
            "Write systemd drop-in 10-token.conf "
            "(mode 0600, root-owned) carrying XRLENV_NODE_TOKEN"
        ),
        python_fn=lambda: _install_node_token_dropin(config),
    )


def _install_node_token_dropin(config: BootstrapConfig) -> None:
    assert config.node_token is not None  # gated by build_plan caller.
    dropin_dir = Path(str(config.systemd_unit) + ".d")
    dropin = dropin_dir / "10-token.conf"
    dropin_dir.mkdir(parents=True, exist_ok=True)
    dropin.write_text(
        f"[Service]\nEnvironment=\"XRLENV_NODE_TOKEN={config.node_token}\"\n",
        encoding="utf-8",
    )
    subprocess.run(["chown", "root:root", str(dropin)], check=True)
    subprocess.run(["chmod", "0600", str(dropin)], check=True)


def _install_dockerhub_auth_step(config: BootstrapConfig) -> Step:
    """Write the runtime user's docker auth config so ``docker pull``
    calls initiated by ``xrlenv-node`` are authenticated.

    This is the only step that decides whether a node's pulls (for
    ``xrlenv build apply``, pull-on-demand at acquire time, or base-
    image pulls during ``docker build`` for git/tarball entries) hit
    Docker Hub's unauth rate cap (~100/6h per source IP) or the
    operator's account-tier cap. End users submitting jobs to the
    control plane never touch Docker Hub directly; their cold
    acquires are insulated from rate limits via this step alone.
    """
    return Step(
        name="install-dockerhub-auth",
        description=(
            f"Write {config.install_root}/.docker/config.json "
            f"(mode 0600, owner {config.runtime_user}) with "
            f"DOCKERHUB_USER:DOCKERHUB_TOKEN so xrlenv-node pulls "
            f"are authenticated"
        ),
        python_fn=lambda: _install_dockerhub_auth(config),
    )


def _install_dockerhub_auth(config: BootstrapConfig) -> None:
    assert config.dockerhub_user is not None  # gated by build_plan caller.
    assert config.dockerhub_token is not None
    import base64

    docker_dir = config.install_root / ".docker"
    config_path = docker_dir / "config.json"
    docker_dir.mkdir(parents=True, exist_ok=True)

    creds = f"{config.dockerhub_user}:{config.dockerhub_token}".encode()
    auth_b64 = base64.b64encode(creds).decode("ascii")
    config_path.write_text(
        json.dumps(
            {
                "auths": {
                    "https://index.docker.io/v1/": {"auth": auth_b64},
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["chown", "-R",
         f"{config.runtime_user}:{config.runtime_user}",
         str(docker_dir)],
        check=True,
    )
    subprocess.run(["chmod", "0700", str(docker_dir)], check=True)
    subprocess.run(["chmod", "0600", str(config_path)], check=True)


def _warn_no_dockerhub_auth_step(config: BootstrapConfig) -> Step:
    """Print a loud one-time WARN at end-of-plan when no Docker Hub
    creds were wired AND no surviving config.json exists. Catches the
    "operator forgot at bootstrap, would otherwise discover 30 min
    later when the first large pull rate-limits" failure mode."""
    return Step(
        name="warn-no-dockerhub-auth",
        description=(
            "(advisory) print a loud warning if no Docker Hub auth "
            "was configured on this node"
        ),
        python_fn=lambda: _warn_no_dockerhub_auth(config),
    )


def _warn_no_dockerhub_auth(config: BootstrapConfig) -> None:
    config_path = config.install_root / ".docker" / "config.json"
    if config_path.is_file():
        return
    msg = (
        "\n"
        "============================================================\n"
        "WARNING: no Docker Hub auth configured on this node.\n"
        "The docker daemon will rate-limit at ~100 image pulls / 6h\n"
        "per source IP. For large sweeps (e.g. 500-instance SWE-bench\n"
        "Verified) end users submitting jobs will hit\n"
        "InsufficientCapacity-shaped failures partway through.\n"
        "\n"
        "Fix at any time, one of:\n"
        "  # (preferred) re-run this bootstrap with creds set:\n"
        "  export DOCKERHUB_USER=<your-handle>\n"
        "  export DOCKERHUB_TOKEN=<your-PAT>\n"
        "  sudo -E bash deploy/bootstrap-{gcp,aws}.sh\n"
        "\n"
        "  # or, log in as the runtime user on this node directly:\n"
        f"  sudo -u {config.runtime_user} docker login\n"
        "\n"
        "A Docker Hub Personal Access Token (PAT) lives at\n"
        "https://hub.docker.com/settings/security — Business / Pro /\n"
        "Team tiers carry a much higher per-account pull cap.\n"
        "============================================================\n"
    )
    sys.stderr.write(msg)


def _systemctl_reload_step() -> Step:
    return Step(
        name="systemctl-daemon-reload",
        description="systemctl daemon-reload (pick up the new unit)",
        commands=[["systemctl", "daemon-reload"]],
    )


def _systemctl_enable_node_step() -> Step:
    return Step(
        name="systemctl-enable-xrlenv-node",
        description="systemctl enable xrlenv-node.service",
        commands=[["systemctl", "enable", "xrlenv-node.service"]],
    )


def _systemctl_restart_node_step() -> Step:
    return Step(
        name="systemctl-restart-xrlenv-node",
        description="systemctl restart xrlenv-node.service",
        commands=[["systemctl", "restart", "xrlenv-node.service"]],
    )


def _ensure_operator_docker_group_step(config: BootstrapConfig) -> Step:
    op = config.operator_user
    assert op is not None
    return Step(
        name="ensure-operator-docker-group",
        description=(
            f"Add operator user {op!r} to the docker group "
            "(so they can run helper scripts without sudo)"
        ),
        commands=[["usermod", "-aG", "docker", op]],
        skip_if=_user_in_group(op, "docker"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Predicates (idempotency)
# ──────────────────────────────────────────────────────────────────────────────


def _binary_present(name: str) -> Callable[[], bool]:
    def predicate() -> bool:
        return shutil.which(name) is not None
    return predicate


def _path_exists(path: Path) -> Callable[[], bool]:
    def predicate() -> bool:
        return path.exists()
    return predicate


def _safe_run(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run ``argv`` capturing output. Returns ``None`` if the binary
    is missing (e.g. predicate probing for ``systemctl`` on macOS
    during a dry-run preview). Predicates fall back to ``False``
    (= "needs to run") on probe failure, which is the conservative
    default — better to show the step in the plan than silently skip
    it because we couldn't even probe."""
    try:
        return subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError:
        return None


def _user_already_exists(name: str) -> Callable[[], bool]:
    def predicate() -> bool:
        result = _safe_run(["id", "-u", name])
        return result is not None and result.returncode == 0
    return predicate


def _user_in_group(user: str, group: str) -> Callable[[], bool]:
    def predicate() -> bool:
        result = _safe_run(["id", "-nG", user])
        if result is None or result.returncode != 0:
            return False
        return group in result.stdout.split()
    return predicate


def _service_already_active(service: str) -> Callable[[], bool]:
    def predicate() -> bool:
        result = _safe_run(["systemctl", "is-active", service])
        if result is None:
            return False
        return result.returncode == 0 and result.stdout.strip() == "active"
    return predicate


# ──────────────────────────────────────────────────────────────────────────────
# uv fallback + verify
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_python_312(config: BootstrapConfig) -> Path | None:
    """Resolve a Python 3.12+ interpreter on this host.

    Probe order (mirrors the bash ``ensure_python_312`` function):

    1. Operator-pinned ``XRLENV_PYTHON`` env var (set explicitly,
       or auto-set by :func:`_install_python_via_uv` after the uv
       fallback ran).
    2. ``python3.14``, ``python3.13``, ``python3.12`` on ``PATH``.
    3. uv-managed interpreters under
       ``$install_root/python/cpython-*/bin/python3.12`` (where
       :func:`_install_python_via_uv` lands its install).

    Returns ``None`` when no candidate satisfies the 3.12 floor —
    callers use that as the "we need the uv fallback to run" signal.
    """
    pinned = os.environ.get("XRLENV_PYTHON")
    if pinned and _python_binary_meets_312(pinned):
        return Path(pinned)
    for candidate in ("python3.14", "python3.13", "python3.12"):
        resolved = shutil.which(candidate)
        if resolved and _python_binary_meets_312(resolved):
            return Path(resolved)
    uv_root = config.install_root / "python"
    if uv_root.exists():
        # Walk for a python3.12 binary; uv places it under a
        # versioned dir like cpython-3.12.x-linux-gnu/bin/python3.12.
        for candidate_path in uv_root.rglob("bin/python3.12"):
            if _python_binary_meets_312(str(candidate_path)):
                return candidate_path
    return None


def _python_binary_meets_312(binary: str) -> bool:
    """Run ``<binary> -c '...' `` and return ``True`` if the resulting
    interpreter reports ``sys.version_info >= (3, 12)``. Safe against
    a non-executable / non-Python path (returns ``False``)."""
    if not Path(binary).exists() and shutil.which(binary) is None:
        return False
    result = _safe_run([binary, "-c",
        "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"])
    return result is not None and result.returncode == 0


def _python_312_resolvable(config: BootstrapConfig) -> Callable[[], bool]:
    """Skip predicate factory: ``True`` iff
    :func:`_resolve_python_312` finds an interpreter. The predicate
    captures ``config`` by closure so the planner builds it once and
    every step that wants this skip semantic shares the same probe."""
    def predicate() -> bool:
        return _resolve_python_312(config) is not None
    return predicate


def _create_venv(config: BootstrapConfig) -> None:
    """Resolve the Python 3.12 interpreter and create
    ``$install_root/.venv`` with it. Raises :class:`RuntimeError`
    with the resolver's probe chain in the error message when no
    candidate is found — the uv fallback should have run first."""
    py = _resolve_python_312(config)
    if py is None:
        raise RuntimeError(
            "no Python >= 3.12 found after the install steps. Probe "
            "order checked: $XRLENV_PYTHON, "
            "python3.{14,13,12} on PATH, "
            f"{config.install_root}/python/*/bin/python3.12. "
            "Re-run with XRLENV_PYTHON=/path/to/python3.12 if you have "
            "one installed elsewhere, or inspect why the uv fallback "
            "didn't complete (network egress / disk space).",
        )
    subprocess.run(
        [str(py), "-m", "venv", str(config.install_root / ".venv")],
        check=True,
    )


def _install_python_via_uv(config: BootstrapConfig) -> None:
    """Last-resort Python 3.12 installer: fetch ``uv`` and use it
    to install a portable python-build-standalone interpreter.

    Mirrors ``bootstrap-common.sh``'s ``_install_python_via_uv``.
    """
    install_dir = config.install_root / "python"
    install_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("uv") is None:
        arch = platform.machine()
        if arch == "x86_64":
            uv_target = "x86_64-unknown-linux-gnu"
        elif arch == "aarch64":
            uv_target = "aarch64-unknown-linux-gnu"
        else:
            raise RuntimeError(
                f"uv prebuilt binaries do not cover arch {arch!r}. "
                "Install Python 3.12 manually and re-run with "
                "XRLENV_PYTHON=/path/to/python3.12 in the environment.",
            )
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tarball_url = (
                f"https://github.com/astral-sh/uv/releases/latest/"
                f"download/uv-{uv_target}.tar.gz"
            )
            tarball = Path(tmp) / "uv.tgz"
            subprocess.run(
                ["curl", "-fsSL", "-o", str(tarball), tarball_url],
                check=True,
            )
            subprocess.run(
                ["tar", "-xzC", tmp, "-f", str(tarball)], check=True,
            )
            uv_bin = Path(tmp) / f"uv-{uv_target}" / "uv"
            subprocess.run(
                ["install", "-m", "0755", str(uv_bin), "/usr/local/bin/uv"],
                check=True,
            )
    subprocess.run(
        ["uv", "python", "install", "3.12"],
        check=True,
        env={**os.environ, "UV_PYTHON_INSTALL_DIR": str(install_dir)},
    )
    # Pin the resolved binary into XRLENV_PYTHON so any downstream
    # diagnostics that show env vars (and a re-run that re-checks
    # the skip predicate without re-probing the install dir) find
    # the installed interpreter even if uv didn't symlink it onto
    # PATH. The resolver still falls back to the rglob probe under
    # ``install_dir`` so a stale-env scenario doesn't break us.
    resolved = next(
        (p for p in install_dir.rglob("bin/python3.12")
         if _python_binary_meets_312(str(p))),
        None,
    )
    if resolved is not None:
        os.environ["XRLENV_PYTHON"] = str(resolved)


def _verify_xrlenv_install(config: BootstrapConfig) -> None:
    venv_bin = config.install_root / ".venv" / "bin"
    if not (venv_bin / "xrlenv-node").is_file():
        raise RuntimeError(
            f"xrlenv-node missing from {venv_bin}. The pip install "
            "succeeded but the console-script didn't land — common "
            "causes: stale editable .pth pointing at a checkout "
            "outside the venv, or an old venv built with python<3.12. "
            f"Wipe with `rm -rf {config.install_root}/.venv` and re-run.",
        )
    # Importable from a neutral cwd so a sibling checkout doesn't
    # shadow the venv-installed copy.
    result = subprocess.run(
        [str(venv_bin / "python"),
         "-c", "import xrlenv.node, xrlenv.node.cli"],
        capture_output=True, text=True,
        cwd="/",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"xrlenv.node not importable from {venv_bin}. "
            f"Error: {result.stderr.strip()}",
        )


def _repo_root() -> Path:
    """Locate the xrlenv repo root by walking up from this file
    until we find ``deploy/``. The systemd unit lives in
    ``deploy/systemd/`` relative to that root."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "deploy" / "systemd").is_dir():
            return parent
    raise RuntimeError(
        f"could not locate xrlenv repo root above {here}; "
        "deploy/systemd/ must be reachable.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plan runner
# ──────────────────────────────────────────────────────────────────────────────


def run_plan(
    plan: list[Step],
    *,
    dry_run: bool,
    out: IO[str],
) -> int:
    """Execute (or preview) every step in ``plan``.

    Dry-run mode prints a numbered list of step names + descriptions
    + the commands each step would run. Live mode runs the commands
    via :func:`subprocess.run`. Either mode short-circuits on the
    first ``skip_if`` predicate that returns ``True`` for a step.
    """
    label = "DRY-RUN" if dry_run else "LIVE"
    out.write(f"[xrlenv bootstrap — {label}] {len(plan)} step(s)\n")
    for i, step in enumerate(plan, start=1):
        skipped = step.skip_if is not None and step.skip_if()
        header = (
            f"\n[{i:02d}/{len(plan):02d}] {step.name}"
            f"{'  (skipped — already done)' if skipped else ''}\n"
            f"        {step.description}\n"
        )
        out.write(header)
        if skipped:
            continue
        if dry_run:
            if step.python_fn is not None:
                out.write(
                    "        DRY: in-process Python step (see "
                    f"{step.name} in xrlenv/cli/bootstrap.py)\n",
                )
            for cmd in step.commands:
                out.write(f"        DRY: {' '.join(cmd)}\n")
            continue
        try:
            if step.python_fn is not None:
                step.python_fn()
            for cmd in step.commands:
                env = (
                    {**os.environ, **step.env}
                    if step.env is not None else None
                )
                subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            if step.allow_failure:
                out.write(
                    f"        SOFT-FAIL (allowed): command exited "
                    f"{exc.returncode} — continuing\n",
                )
                continue
            out.write(
                f"        FAIL: command exited {exc.returncode}\n",
            )
            return 1
        except Exception as exc:
            if step.allow_failure:
                out.write(
                    f"        SOFT-FAIL (allowed): {exc} — continuing\n",
                )
                continue
            out.write(f"        FAIL: {exc}\n")
            return 1
        out.write("        OK\n")
    out.write(f"\n[xrlenv bootstrap] {label} complete.\n")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────


def cmd_bootstrap(
    *,
    target: str,
    control_plane: str | None,
    node_id: str | None,
    target_os: str | None,
    xrlenv_wheel: Path | None,
    xrlenv_repo: Path | None,
    xrlenv_version: str | None,
    runtime_user: str | None,
    install_root: Path | None,
    skip_operator_docker_group: bool,
    dry_run: bool,
    out: TextIO,
) -> int:
    """``xrlenv bootstrap`` entry point — see module docstring."""
    try:
        config = build_config(
            target=target,
            control_plane=control_plane,
            node_id=node_id,
            target_os=target_os,
            xrlenv_wheel=xrlenv_wheel,
            xrlenv_repo=xrlenv_repo,
            xrlenv_version=xrlenv_version,
            runtime_user=runtime_user,
            install_root=install_root,
            skip_operator_docker_group=skip_operator_docker_group,
        )
    except (ValueError, RuntimeError) as exc:
        out.write(f"error: {exc}\n")
        return 2
    plan = build_plan(config)
    return run_plan(plan, dry_run=dry_run, out=out)


__all__ = [
    "BootstrapConfig",
    "OSInfo",
    "Step",
    "autodetect_node_id_aws",
    "autodetect_node_id_gcp",
    "build_config",
    "build_plan",
    "cmd_bootstrap",
    "detect_os",
    "run_plan",
]


# ──────────────────────────────────────────────────────────────────────────────
# Standalone script entry point
# ──────────────────────────────────────────────────────────────────────────────
#
# Two ways to reach ``cmd_bootstrap``:
#
# 1. Via the unified CLI: ``xrlenv bootstrap --target gcp ...``. Wired in
#    ``xrlenv/cli/__main__.py``. Requires xrlenv to be installed first.
# 2. Direct: ``python3 xrlenv/cli/bootstrap.py --target gcp ...``.
#    Stdlib-only — runs under any Python 3.10+ system interpreter, even
#    before xrlenv itself is installed. This is how the
#    ``deploy/bootstrap-{gcp,aws}.sh`` wrappers invoke it on a fresh VM
#    that hasn't bootstrapped its venv yet.
#
# The second path lets the slice keep its chicken-and-egg promise:
# replace the bash logic with Python *and* still run on a freshly
# provisioned host whose ``xrlenv`` import would fail because pydantic
# isn't installed.


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point. Used by deploy/bootstrap-{gcp,aws}.sh
    on fresh VMs where xrlenv itself isn't installed yet."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="xrlenv-bootstrap",
        description=(
            "Install + configure the xrlenv-node daemon on a freshly "
            "provisioned VM. Replaces deploy/bootstrap-{gcp,aws}.sh."
        ),
    )
    parser.add_argument(
        "--target", choices=("gcp", "aws", "linux-generic"), required=True,
    )
    parser.add_argument("--control-plane", default=None)
    parser.add_argument("--node-id", default=None)
    parser.add_argument("--target-os", default=None)
    parser.add_argument("--xrlenv-wheel", default=None)
    parser.add_argument("--xrlenv-repo", default=None)
    parser.add_argument("--xrlenv-version", default=None)
    parser.add_argument("--runtime-user", default=None)
    parser.add_argument("--install-root", default=None)
    parser.add_argument(
        "--skip-operator-docker-group", action="store_true", default=False,
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args(argv)

    return cmd_bootstrap(
        target=args.target,
        control_plane=args.control_plane,
        node_id=args.node_id,
        target_os=args.target_os,
        xrlenv_wheel=Path(args.xrlenv_wheel).expanduser() if args.xrlenv_wheel else None,
        xrlenv_repo=Path(args.xrlenv_repo).expanduser() if args.xrlenv_repo else None,
        xrlenv_version=args.xrlenv_version,
        runtime_user=args.runtime_user,
        install_root=Path(args.install_root).expanduser() if args.install_root else None,
        skip_operator_docker_group=args.skip_operator_docker_group,
        dry_run=args.dry_run,
        out=sys.stdout,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
