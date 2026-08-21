"""Unit tests for slurm_scripts/generate_deployment_script.py.

The generator rewrites topology-bearing lines in six Slurm shell scripts from a
``clusters.yaml`` source of truth.  All file I/O in ``render_all`` is
exercised against ``tmp_path`` stubs (real filesystem, no mocks) so the regex
matching / replacement logic is tested with realistic text, not just string
literals.

Coverage targets
----------------
* Successful rendering: render_deploy / render_node / render_control each update
  every topology field and preserve unrelated content.
* Idempotence: writing rendered output to disk then re-rendering produces
  identical content.
* Check-mode detection: render_all reports files as changed vs. unchanged.
* Validation failures: pool outside workers, pool overlap, control-plane among
  workers, malformed YAML, missing / extra / invalid config fields.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "slurm_scripts" / "generate_deployment_script.py"


# ── module loader (script is not a package) ───────────────────────────────────

def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_deployment_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load_module()
ConfigError = gen.ConfigError
Cluster = gen.Cluster
RegistryTopology = gen.RegistryTopology


# ── shared helpers ────────────────────────────────────────────────────────────

def _cluster(
    name: str = "dev",
    cp: str = "cp-node",
    workers: tuple[str, ...] = ("worker-1", "worker-2"),
    sysbox: tuple[str, ...] = ("worker-1",),
    cpu: tuple[str, ...] = ("worker-2",),
    registry: Any = None,
) -> Cluster:
    if registry is None:
        registry = RegistryTopology(mirror_host="reg-host", private_host="reg-host")
    return Cluster(name, cp, registry, workers, sysbox, cpu)


def _valid_raw() -> dict[str, Any]:
    return {
        "control_plane": "cp-host",
        "registry": {"mirror_host": "reg-host", "private_host": "reg-host"},
        "workers": ["w1", "w2"],
        "sysbox_pool": ["w1"],
        "cpu_isolation_pool": ["w2"],
    }


# ── _host_list ────────────────────────────────────────────────────────────────


def test_host_list_rejects_non_list() -> None:
    with pytest.raises(ConfigError, match="non-empty YAML list"):
        gen._host_list("host", cluster="dev", field="workers", allow_empty=False)


def test_host_list_rejects_empty_when_not_allowed() -> None:
    with pytest.raises(ConfigError, match="non-empty YAML list"):
        gen._host_list([], cluster="dev", field="workers", allow_empty=False)


def test_host_list_accepts_empty_when_allowed() -> None:
    assert gen._host_list([], cluster="dev", field="sysbox_pool", allow_empty=True) == ()


def test_host_list_rejects_invalid_hostname_chars() -> None:
    with pytest.raises(ConfigError, match="hostnames containing only"):
        gen._host_list(["bad host!"], cluster="dev", field="workers", allow_empty=False)


def test_host_list_rejects_hostname_starting_with_dot() -> None:
    with pytest.raises(ConfigError, match="hostnames containing only"):
        gen._host_list([".bad"], cluster="dev", field="workers", allow_empty=False)


def test_host_list_rejects_duplicates() -> None:
    with pytest.raises(ConfigError, match="duplicate hosts"):
        gen._host_list(["w1", "w1"], cluster="dev", field="workers", allow_empty=False)


def test_host_list_accepts_valid_hostnames() -> None:
    result = gen._host_list(
        ["node-host", "host.example.com", "node_2"],
        cluster="dev",
        field="workers",
        allow_empty=False,
    )
    assert result == ("node-host", "host.example.com", "node_2")


# ── _parse_cluster ────────────────────────────────────────────────────────────


def test_parse_cluster_happy_path() -> None:
    c = gen._parse_cluster("dev", _valid_raw())
    assert c.name == "dev"
    assert c.control_plane == "cp-host"
    assert c.registry == RegistryTopology(mirror_host="reg-host", private_host="reg-host")
    assert c.workers == ("w1", "w2")
    assert c.sysbox_pool == ("w1",)
    assert c.cpu_isolation_pool == ("w2",)


# ── _parse_registry ─────────────────────────────────────────────────────────


def test_parse_registry_happy_path() -> None:
    r = gen._parse_registry(
        {"mirror_host": "m-host", "private_host": "p-host"}, cluster="dev"
    )
    assert r.mirror_host == "m-host"
    assert r.private_host == "p-host"


def test_parse_registry_ports_default_when_omitted() -> None:
    r = gen._parse_registry(
        {"mirror_host": "m", "private_host": "p"}, cluster="dev"
    )
    assert r.mirror_port == gen.DEFAULT_MIRROR_PORT == 5010
    assert r.private_port == gen.DEFAULT_PRIVATE_PORT == 5011


def test_parse_registry_accepts_custom_ports() -> None:
    r = gen._parse_registry(
        {"mirror_host": "m", "private_host": "p",
         "mirror_port": 6010, "private_port": 6011},
        cluster="dev",
    )
    assert r.mirror_port == 6010
    assert r.private_port == 6011


def test_parse_registry_rejects_out_of_range_port() -> None:
    with pytest.raises(ConfigError, match="mirror_port must be an integer port"):
        gen._parse_registry(
            {"mirror_host": "m", "private_host": "p", "mirror_port": 70000},
            cluster="dev",
        )


def test_parse_registry_rejects_non_int_port() -> None:
    with pytest.raises(ConfigError, match="private_port must be an integer port"):
        gen._parse_registry(
            {"mirror_host": "m", "private_host": "p", "private_port": "5011"},
            cluster="dev",
        )


def test_parse_registry_rejects_bool_port() -> None:
    # bool is an int subclass — must not be silently accepted as port 1.
    with pytest.raises(ConfigError, match="mirror_port must be an integer port"):
        gen._parse_registry(
            {"mirror_host": "m", "private_host": "p", "mirror_port": True},
            cluster="dev",
        )


def test_parse_registry_rejects_non_dict() -> None:
    with pytest.raises(ConfigError, match="registry must be a YAML mapping"):
        gen._parse_registry(["not", "a", "dict"], cluster="dev")


def test_parse_registry_rejects_missing_field() -> None:
    with pytest.raises(ConfigError, match="missing: private_host"):
        gen._parse_registry({"mirror_host": "m-host"}, cluster="dev")


def test_parse_registry_rejects_extra_field() -> None:
    with pytest.raises(ConfigError, match="unknown: extra"):
        gen._parse_registry(
            {"mirror_host": "m", "private_host": "p", "extra": "x"}, cluster="dev"
        )


def test_parse_registry_rejects_invalid_hostname() -> None:
    with pytest.raises(ConfigError, match="registry.mirror_host must be a valid hostname"):
        gen._parse_registry(
            {"mirror_host": "bad host!", "private_host": "p-host"}, cluster="dev"
        )


def test_parse_cluster_rejects_missing_registry() -> None:
    raw = _valid_raw()
    del raw["registry"]
    with pytest.raises(ConfigError, match="missing: registry"):
        gen._parse_cluster("dev", raw)


def test_parse_cluster_rejects_non_dict() -> None:
    with pytest.raises(ConfigError, match="YAML mapping"):
        gen._parse_cluster("dev", ["not", "a", "dict"])


def test_parse_cluster_rejects_missing_field() -> None:
    raw = _valid_raw()
    del raw["workers"]
    with pytest.raises(ConfigError, match="missing: workers"):
        gen._parse_cluster("dev", raw)


def test_parse_cluster_rejects_extra_field() -> None:
    raw = _valid_raw()
    raw["extra_key"] = "unexpected"
    with pytest.raises(ConfigError, match="unknown: extra_key"):
        gen._parse_cluster("dev", raw)


def test_parse_cluster_rejects_invalid_control_plane_hostname() -> None:
    raw = _valid_raw()
    raw["control_plane"] = "bad host!"
    with pytest.raises(ConfigError, match="control_plane must be a valid hostname"):
        gen._parse_cluster("dev", raw)


def test_parse_cluster_rejects_sysbox_pool_outside_workers() -> None:
    raw = _valid_raw()
    raw["sysbox_pool"] = ["w1", "outsider"]
    with pytest.raises(ConfigError, match="sysbox_pool contains non-workers"):
        gen._parse_cluster("dev", raw)


def test_parse_cluster_rejects_cpu_pool_outside_workers() -> None:
    raw = _valid_raw()
    raw["cpu_isolation_pool"] = ["w2", "outsider"]
    with pytest.raises(ConfigError, match="cpu_isolation_pool contains non-workers"):
        gen._parse_cluster("dev", raw)


def test_parse_cluster_rejects_pool_overlap() -> None:
    raw = _valid_raw()
    raw["sysbox_pool"] = ["w1", "w2"]
    raw["cpu_isolation_pool"] = ["w2"]
    with pytest.raises(ConfigError, match="overlap"):
        gen._parse_cluster("dev", raw)


def test_parse_cluster_rejects_cp_among_workers() -> None:
    raw = _valid_raw()
    raw["workers"] = ["w1", "w2", "cp-host"]
    raw["sysbox_pool"] = ["w1"]
    raw["cpu_isolation_pool"] = ["w2"]
    with pytest.raises(ConfigError, match="must not also be a worker"):
        gen._parse_cluster("dev", raw)


def test_parse_cluster_empty_pools_are_valid() -> None:
    raw = _valid_raw()
    raw["sysbox_pool"] = []
    raw["cpu_isolation_pool"] = []
    c = gen._parse_cluster("dev", raw)
    assert c.sysbox_pool == ()
    assert c.cpu_isolation_pool == ()


# ── load_config ───────────────────────────────────────────────────────────────


def _write_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "clusters.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_config_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        gen.load_config(tmp_path / "nonexistent.yaml", ("dev",))


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    p = _write_config(tmp_path, "key: [\nunclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        gen.load_config(p, ("dev",))


def test_load_config_non_dict_root(tmp_path: Path) -> None:
    p = _write_config(tmp_path, "- item1\n- item2\n")
    with pytest.raises(ConfigError, match="root must be a YAML mapping"):
        gen.load_config(p, ("dev",))


def test_load_config_missing_cluster(tmp_path: Path) -> None:
    # File has prod but not dev.
    content = textwrap.dedent("""\
        prod:
          control_plane: prod-cp
          workers: [p1]
          sysbox_pool: []
          cpu_isolation_pool: []
    """)
    p = _write_config(tmp_path, content)
    with pytest.raises(ConfigError, match="missing clusters: dev"):
        gen.load_config(p, ("dev",))


def test_load_config_valid_single_cluster(tmp_path: Path) -> None:
    content = textwrap.dedent("""\
        dev:
          control_plane: dev-cp
          registry:
            mirror_host: reg-host
            private_host: reg-host
          workers:
            - w1
            - w2
          sysbox_pool:
            - w1
          cpu_isolation_pool:
            - w2
    """)
    p = _write_config(tmp_path, content)
    clusters = gen.load_config(p, ("dev",))
    assert len(clusters) == 1
    c = clusters[0]
    assert c.name == "dev"
    assert c.control_plane == "dev-cp"
    assert c.registry.mirror_host == "reg-host"
    assert c.workers == ("w1", "w2")


def test_load_config_multiple_clusters_preserves_order(tmp_path: Path) -> None:
    content = textwrap.dedent("""\
        dev:
          control_plane: dev-cp
          registry: {mirror_host: dreg, private_host: dreg}
          workers: [dw1, dw2]
          sysbox_pool: [dw1]
          cpu_isolation_pool: [dw2]
        prod:
          control_plane: prod-cp
          registry: {mirror_host: preg, private_host: preg}
          workers: [pw1, pw2]
          sysbox_pool: [pw1]
          cpu_isolation_pool: [pw2]
    """)
    p = _write_config(tmp_path, content)
    clusters = gen.load_config(p, ("dev", "prod"))
    assert len(clusters) == 2
    assert clusters[0].name == "dev"
    assert clusters[1].name == "prod"


# ── render_deploy ─────────────────────────────────────────────────────────────

# Minimal deploy-script stub containing all three patterns render_deploy touches.
_DEPLOY_TEMPLATE = textwrap.dedent("""\
    #!/usr/bin/env bash
    CP_NODE="old-cp"                    # control-plane host
    TUNNEL_FWDS=(-L 9080:localhost:8080)
    SYSBOX_POOL=(old-sysbox1 old-sysbox2)
    CPU_ISOLATION_POOL=(old-cpu1 old-cpu2)
    OTHER_VAR=unchanged
""")


def test_render_deploy_updates_cp_node() -> None:
    c = _cluster(cp="new-cp")
    result = gen.render_deploy(_DEPLOY_TEMPLATE, c, Path("deploy.sh"))
    assert 'CP_NODE="new-cp"' in result


def test_render_deploy_preserves_inline_comment() -> None:
    """The trailing content on the CP_NODE line must survive replacement."""
    c = _cluster(cp="new-cp")
    result = gen.render_deploy(_DEPLOY_TEMPLATE, c, Path("deploy.sh"))
    assert 'CP_NODE="new-cp"                    # control-plane host' in result


def test_render_deploy_updates_sysbox_pool() -> None:
    c = _cluster(workers=("s1", "s2", "c1"), sysbox=("s1", "s2"), cpu=("c1",))
    result = gen.render_deploy(_DEPLOY_TEMPLATE, c, Path("deploy.sh"))
    assert "SYSBOX_POOL=(s1 s2)" in result


def test_render_deploy_updates_cpu_isolation_pool() -> None:
    c = _cluster(workers=("s1", "c1"), sysbox=("s1",), cpu=("c1",))
    result = gen.render_deploy(_DEPLOY_TEMPLATE, c, Path("deploy.sh"))
    assert "CPU_ISOLATION_POOL=(c1)" in result


def test_render_deploy_empty_pools() -> None:
    """Both pools may be empty — the bash array form ``()`` is still valid."""
    c = _cluster(workers=("w1",), sysbox=(), cpu=())
    result = gen.render_deploy(_DEPLOY_TEMPLATE, c, Path("deploy.sh"))
    assert "SYSBOX_POOL=()" in result
    assert "CPU_ISOLATION_POOL=()" in result


def test_render_deploy_preserves_unrelated_lines() -> None:
    c = _cluster()
    result = gen.render_deploy(_DEPLOY_TEMPLATE, c, Path("deploy.sh"))
    assert "TUNNEL_FWDS=(-L 9080:localhost:8080)" in result
    assert "OTHER_VAR=unchanged" in result


def test_render_deploy_missing_pattern_raises() -> None:
    bad = "#!/usr/bin/env bash\n# none of the three required topology lines\n"
    with pytest.raises(ConfigError, match="expected exactly one match"):
        gen.render_deploy(bad, _cluster(), Path("deploy.sh"))


# ── render_node ───────────────────────────────────────────────────────────────

# Minimal node-script stub containing all three patterns render_node touches.
_NODE_TEMPLATE = textwrap.dedent("""\
    #!/bin/bash
    #SBATCH --nodes=2                # Number of nodes
    #SBATCH --nodelist=old-w1,old-w2
    echo "some setup"
    : "${XRLENV_GRPC_HOST:=old-cp}"
    : "${XRLENV_GRPC_PORT:=50051}"
""")


def test_render_node_updates_node_count() -> None:
    c = _cluster(workers=("w1", "w2", "w3"), sysbox=("w1",), cpu=("w2",))
    result = gen.render_node(_NODE_TEMPLATE, c, Path("node.sh"))
    assert "#SBATCH --nodes=3" in result


def test_render_node_preserves_trailing_comment() -> None:
    """Trailing text after the --nodes value must survive the replacement."""
    c = _cluster(workers=("w1",), sysbox=(), cpu=())
    result = gen.render_node(_NODE_TEMPLATE, c, Path("node.sh"))
    assert "#SBATCH --nodes=1                # Number of nodes" in result


def test_render_node_updates_nodelist() -> None:
    c = _cluster(workers=("wa", "wb", "wc"), sysbox=("wa",), cpu=("wb",))
    result = gen.render_node(_NODE_TEMPLATE, c, Path("node.sh"))
    assert "#SBATCH --nodelist=wa,wb,wc" in result


def test_render_node_updates_grpc_host() -> None:
    c = _cluster(cp="new-cp")
    result = gen.render_node(_NODE_TEMPLATE, c, Path("node.sh"))
    assert ': "${XRLENV_GRPC_HOST:=new-cp}"' in result


def test_render_node_preserves_unrelated_lines() -> None:
    c = _cluster()
    result = gen.render_node(_NODE_TEMPLATE, c, Path("node.sh"))
    assert ': "${XRLENV_GRPC_PORT:=50051}"' in result
    assert 'echo "some setup"' in result


def test_render_node_missing_pattern_raises() -> None:
    bad = "#!/bin/bash\n# no SBATCH headers here\n"
    with pytest.raises(ConfigError, match="expected exactly one match"):
        gen.render_node(bad, _cluster(), Path("node.sh"))


# ── render_control ────────────────────────────────────────────────────────────

# Minimal control-script stub containing all three patterns render_control touches.
_CONTROL_TEMPLATE = textwrap.dedent("""\
    #!/bin/bash
    #SBATCH --nodelist=old-cp
    exec xrlenv up \\
        --grpc-host old-cp  --grpc-port 50051 \\
        --admin-port 8080
    # scancel job && sbatch foo.sh && ssh -fN  old-cp
""")


def test_render_control_updates_nodelist() -> None:
    c = _cluster(cp="new-cp")
    result = gen.render_control(_CONTROL_TEMPLATE, c, Path("control.sh"))
    assert "#SBATCH --nodelist=new-cp" in result


def test_render_control_updates_grpc_host() -> None:
    c = _cluster(cp="new-cp")
    result = gen.render_control(_CONTROL_TEMPLATE, c, Path("control.sh"))
    assert "    --grpc-host new-cp  --grpc-port 50051" in result


def test_render_control_updates_scancel_comment() -> None:
    c = _cluster(cp="new-cp")
    result = gen.render_control(_CONTROL_TEMPLATE, c, Path("control.sh"))
    # old hostname must be gone; new hostname must appear at line end
    assert "old-cp" not in result
    assert result.endswith("  new-cp\n")


def test_render_control_preserves_unrelated_lines() -> None:
    c = _cluster(cp="new-cp")
    result = gen.render_control(_CONTROL_TEMPLATE, c, Path("control.sh"))
    assert "--admin-port 8080" in result


def test_render_control_missing_pattern_raises() -> None:
    bad = "#!/bin/bash\n# no SBATCH / grpc-host lines here\n"
    with pytest.raises(ConfigError, match="expected exactly one match"):
        gen.render_control(bad, _cluster(), Path("control.sh"))


# ── render_all + idempotence ──────────────────────────────────────────────────


def _write_fake_scripts(scripts_dir: Path, cluster_name: str) -> None:
    """Write minimal but pattern-complete script stubs for render_all testing."""
    (scripts_dir / f"deploy_{cluster_name}.sh").write_text(
        textwrap.dedent("""\
            #!/usr/bin/env bash
            CP_NODE="placeholder"
            SYSBOX_POOL=(placeholder)
            CPU_ISOLATION_POOL=(placeholder)
        """),
        encoding="utf-8",
    )
    (scripts_dir / f"{cluster_name}_xrlenv_node.sh").write_text(
        textwrap.dedent("""\
            #!/bin/bash
            #SBATCH --nodes=1
            #SBATCH --nodelist=placeholder
            : "${XRLENV_GRPC_HOST:=placeholder}"
        """),
        encoding="utf-8",
    )
    (scripts_dir / f"{cluster_name}_xrlenv_control.sh").write_text(
        textwrap.dedent("""\
            #!/bin/bash
            #SBATCH --nodelist=placeholder
                --grpc-host placeholder  --grpc-port 50051 \\
            # scancel job && sbatch foo.sh && ssh -fN  placeholder
        """),
        encoding="utf-8",
    )


def test_render_all_produces_expected_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gen, "SCRIPT_DIR", tmp_path)
    _write_fake_scripts(tmp_path, "dev")
    c = _cluster(name="dev", cp="new-cp", workers=("w1", "w2"), sysbox=("w1",), cpu=("w2",))

    rendered = gen.render_all((c,))

    deploy_out = rendered[tmp_path / "deploy_dev.sh"]
    assert 'CP_NODE="new-cp"' in deploy_out
    assert "SYSBOX_POOL=(w1)" in deploy_out
    assert "CPU_ISOLATION_POOL=(w2)" in deploy_out

    node_out = rendered[tmp_path / "dev_xrlenv_node.sh"]
    assert "#SBATCH --nodes=2" in node_out
    assert "#SBATCH --nodelist=w1,w2" in node_out
    assert ': "${XRLENV_GRPC_HOST:=new-cp}"' in node_out

    ctrl_out = rendered[tmp_path / "dev_xrlenv_control.sh"]
    assert "#SBATCH --nodelist=new-cp" in ctrl_out
    assert "    --grpc-host new-cp" in ctrl_out


def test_render_all_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writing first-pass output to disk then re-rendering must yield identical content."""
    monkeypatch.setattr(gen, "SCRIPT_DIR", tmp_path)
    _write_fake_scripts(tmp_path, "dev")
    c = _cluster(name="dev", cp="cp-host", workers=("w1", "w2"), sysbox=("w1",), cpu=("w2",))

    first_pass = gen.render_all((c,))
    for path, content in first_pass.items():
        path.write_text(content, encoding="utf-8")

    second_pass = gen.render_all((c,))
    assert first_pass == second_pass, "render_all is not idempotent"


def test_render_all_detects_stale_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scripts containing old topology values are flagged as needing update."""
    monkeypatch.setattr(gen, "SCRIPT_DIR", tmp_path)
    _write_fake_scripts(tmp_path, "dev")
    c = _cluster(name="dev", cp="new-cp", workers=("w1", "w2"), sysbox=("w1",), cpu=("w2",))

    rendered = gen.render_all((c,))
    changed = [
        path
        for path, content in rendered.items()
        if path.read_text(encoding="utf-8") != content
    ]
    assert changed, "expected stale scripts to be detected"


def test_render_all_no_changes_when_uptodate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After writing rendered content, a second render_all reports no diff."""
    monkeypatch.setattr(gen, "SCRIPT_DIR", tmp_path)
    _write_fake_scripts(tmp_path, "dev")
    c = _cluster(name="dev", cp="cp-host", workers=("w1", "w2"), sysbox=("w1",), cpu=("w2",))

    first = gen.render_all((c,))
    for path, content in first.items():
        path.write_text(content, encoding="utf-8")

    second = gen.render_all((c,))
    unchanged = [
        path
        for path, content in second.items()
        if path.read_text(encoding="utf-8") == content
    ]
    assert len(unchanged) == len(second), "expected all scripts to be up to date"


def test_render_all_missing_script_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gen, "SCRIPT_DIR", tmp_path)
    # Do NOT write any stubs — all three files are absent.
    c = _cluster(name="dev")
    with pytest.raises(ConfigError, match="deployment script does not exist"):
        gen.render_all((c,))


def test_render_all_covers_all_six_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """render_all with both clusters touches exactly 6 output paths (3 per cluster)."""
    monkeypatch.setattr(gen, "SCRIPT_DIR", tmp_path)
    for name in ("dev", "prod"):
        _write_fake_scripts(tmp_path, name)
    clusters = (
        _cluster(name="dev", cp="dev-cp", workers=("dw1",), sysbox=(), cpu=("dw1",)),
        _cluster(name="prod", cp="prod-cp", workers=("pw1",), sysbox=("pw1",), cpu=()),
    )
    rendered = gen.render_all(clusters)
    assert len(rendered) == 6


# ── render_env ────────────────────────────────────────────────────────────────

# A realistic .env: secrets interleaved with the three topology host keys plus
# the fixed ports. Only the three HOST lines must change; everything else is
# byte-preserved.
_ENV_TEMPLATE = textwrap.dedent("""\
    XRLENV_NODE_TOKEN=super-secret-node-token
    XRLENV_CONSUMER_TOKEN=super-secret-consumer-token
    XRLENV_GRPC_HOST=old-cp
    XRLENV_GRPC_PORT=50051
    XRLENV_MIRROR_REGISTRY_HOST=old-reg
    XRLENV_MIRROR_REGISTRY_PORT=5010
    XRLENV_PRIVATE_REGISTRY_HOST=old-reg
    XRLENV_PRIVATE_REGISTRY_PORT=5011
    DOCKERHUB_USER=someuser
    DOCKERHUB_TOKEN=super-secret-dockerhub-token
""")


def _reg(
    mirror: str = "reg-new", private: str = "reg-new",
    mirror_port: int = 5010, private_port: int = 5011,
) -> Any:
    return RegistryTopology(
        mirror_host=mirror, private_host=private,
        mirror_port=mirror_port, private_port=private_port,
    )


def test_render_env_updates_all_topology_keys() -> None:
    c = _cluster(cp="new-cp", registry=_reg("mir-new", "prv-new", 6010, 6011))
    result = gen.render_env(_ENV_TEMPLATE, c, Path(".env"))
    assert "XRLENV_GRPC_HOST=new-cp\n" in result
    assert "XRLENV_MIRROR_REGISTRY_HOST=mir-new\n" in result
    assert "XRLENV_MIRROR_REGISTRY_PORT=6010\n" in result
    assert "XRLENV_PRIVATE_REGISTRY_HOST=prv-new\n" in result
    assert "XRLENV_PRIVATE_REGISTRY_PORT=6011\n" in result


def test_render_env_preserves_secrets_and_grpc_port() -> None:
    c = _cluster(cp="new-cp", registry=_reg())
    result = gen.render_env(_ENV_TEMPLATE, c, Path(".env"))
    # Secrets untouched.
    assert "XRLENV_NODE_TOKEN=super-secret-node-token\n" in result
    assert "DOCKERHUB_TOKEN=super-secret-dockerhub-token\n" in result
    assert "XRLENV_CONSUMER_TOKEN=super-secret-consumer-token\n" in result
    # XRLENV_GRPC_PORT is NOT a synced key — left untouched.
    assert "XRLENV_GRPC_PORT=50051\n" in result
    # Old topology host values gone.
    assert "old-cp" not in result
    assert "old-reg" not in result


def test_render_env_is_idempotent() -> None:
    c = _cluster(cp="new-cp", registry=_reg("mir", "prv"))
    once = gen.render_env(_ENV_TEMPLATE, c, Path(".env"))
    twice = gen.render_env(once, c, Path(".env"))
    assert once == twice


def test_render_env_appends_missing_key() -> None:
    # A .env lacking the registry host keys gets them appended (fresh checkout).
    minimal = "XRLENV_NODE_TOKEN=t\nXRLENV_GRPC_HOST=old\n"
    c = _cluster(cp="cp1", registry=_reg("m1", "p1"))
    result = gen.render_env(minimal, c, Path(".env"))
    assert "XRLENV_GRPC_HOST=cp1\n" in result
    assert "XRLENV_MIRROR_REGISTRY_HOST=m1\n" in result
    assert "XRLENV_PRIVATE_REGISTRY_HOST=p1\n" in result
    assert "XRLENV_NODE_TOKEN=t\n" in result


def test_render_env_rejects_duplicate_key() -> None:
    dup = "XRLENV_GRPC_HOST=a\nXRLENV_GRPC_HOST=b\n"
    c = _cluster(cp="cp1", registry=_reg())
    with pytest.raises(ConfigError, match="XRLENV_GRPC_HOST appears 2 times"):
        gen.render_env(dup, c, Path(".env"))


def test_render_all_syncs_env_when_env_cluster_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gen, "SCRIPT_DIR", tmp_path / "slurm_scripts")
    (tmp_path / "slurm_scripts").mkdir()
    _write_fake_scripts(tmp_path / "slurm_scripts", "dev")
    env_path = tmp_path / ".env"
    env_path.write_text(_ENV_TEMPLATE, encoding="utf-8")

    c = _cluster(name="dev", cp="dev-cp", workers=("w1",), sysbox=(), cpu=("w1",),
                 registry=_reg("dev-reg", "dev-reg"))
    rendered = gen.render_all((c,), env_cluster="dev", env_path=env_path)

    assert env_path in rendered
    assert "XRLENV_GRPC_HOST=dev-cp\n" in rendered[env_path]
    assert "XRLENV_MIRROR_REGISTRY_HOST=dev-reg\n" in rendered[env_path]
    # secrets preserved
    assert "XRLENV_NODE_TOKEN=super-secret-node-token\n" in rendered[env_path]


def test_render_all_env_cluster_not_selected_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gen, "SCRIPT_DIR", tmp_path / "slurm_scripts")
    (tmp_path / "slurm_scripts").mkdir()
    _write_fake_scripts(tmp_path / "slurm_scripts", "dev")
    env_path = tmp_path / ".env"
    env_path.write_text(_ENV_TEMPLATE, encoding="utf-8")
    c = _cluster(name="dev")
    with pytest.raises(ConfigError, match="not among the selected clusters"):
        gen.render_all((c,), env_cluster="prod", env_path=env_path)


def test_render_all_env_file_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gen, "SCRIPT_DIR", tmp_path / "slurm_scripts")
    (tmp_path / "slurm_scripts").mkdir()
    _write_fake_scripts(tmp_path / "slurm_scripts", "dev")
    c = _cluster(name="dev")
    with pytest.raises(ConfigError, match="env file does not exist"):
        gen.render_all((c,), env_cluster="dev", env_path=tmp_path / ".env")
