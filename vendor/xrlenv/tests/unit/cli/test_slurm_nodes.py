from __future__ import annotations

import io
from pathlib import Path

import pytest
import yaml
from xrlenv.cli.slurm_nodes import (
    cmd_nodes_from_slurm,
    default_address_for_hostname,
    expand_slurm_nodelist,
    extract_slurm_nodelist,
    render_nodes_inventory,
)
from xrlenv.control.kwargs_policy import DEFAULT_POLICY


def test_extract_slurm_nodelist_from_sbatch_equals_form(tmp_path: Path) -> None:
    script = tmp_path / "workers.sh"
    script.write_text(
        "\n".join([
            "#!/bin/bash",
            "#SBATCH --nodes=2",
            "#SBATCH --nodelist=ip-10-0-7-8,ip-10-0-11-12 # workers",
            "srun hostname",
        ]),
        encoding="utf-8",
    )

    assert extract_slurm_nodelist(script) == "ip-10-0-7-8,ip-10-0-11-12"


def test_extract_slurm_nodelist_from_short_flag(tmp_path: Path) -> None:
    script = tmp_path / "workers.sh"
    script.write_text("#SBATCH -w node[01-02]\n", encoding="utf-8")

    assert extract_slurm_nodelist(script) == "node[01-02]"


def test_expand_slurm_nodelist_preserves_order_and_removes_duplicates() -> None:
    assert expand_slurm_nodelist("node[01-03,02],other") == [
        "node01",
        "node02",
        "node03",
        "other",
    ]


def test_default_address_for_hostname_converts_hyperpod_ip_names() -> None:
    assert default_address_for_hostname("ip-10-0-7-8") == "10.0.7.8"
    assert default_address_for_hostname("worker-01") == "worker-01"


def test_render_nodes_inventory_preserves_existing_policy(tmp_path: Path) -> None:
    output = tmp_path / "nodes.yaml"
    output.write_text(
        yaml.safe_dump({
            "version": 1,
            "nodes": [],
            "policy": {
                "allowed_devices": ["/dev/kvm"],
                "denied_caps": ["SYS_MODULE"],
                "allow_host_network": True,
                "allow_privileged": False,
                "allowed_host_paths": [],
            },
        }),
        encoding="utf-8",
    )

    inventory = render_nodes_inventory(
        hostnames=["ip-10-0-7-8"],
        source_script=tmp_path / "workers.sh",
        output_path=output,
        id_template="aws-{hostname}",
        address_template="{address}",
        cloud="aws",
        backends=["docker"],
        auth_token_env="XRLENV_NODE_TOKEN",
    )

    assert inventory["nodes"] == [{
        "id": "aws-ip-10-0-7-8",
        "cloud": "aws",
        "backends": ["docker"],
        "auth_token_env": "XRLENV_NODE_TOKEN",
        "address": "10.0.7.8",
    }]
    assert inventory["policy"]["denied_caps"] == ["SYS_MODULE"]
    assert inventory["policy"]["allow_host_network"] is True


def test_extract_slurm_nodelist_space_separated_long_flag(tmp_path: Path) -> None:
    script = tmp_path / "workers.sh"
    script.write_text("#SBATCH --nodelist node[01-02]\n", encoding="utf-8")

    assert extract_slurm_nodelist(script) == "node[01-02]"


def test_extract_slurm_nodelist_raises_when_no_directive(tmp_path: Path) -> None:
    script = tmp_path / "workers.sh"
    script.write_text("#!/bin/bash\n#SBATCH --nodes=4\nsrun hostname\n", encoding="utf-8")

    with pytest.raises(ValueError, match="--nodelist"):
        extract_slurm_nodelist(script)


def test_expand_slurm_nodelist_raises_on_empty_string() -> None:
    with pytest.raises(ValueError, match="empty"):
        expand_slurm_nodelist("")


def test_expand_slurm_nodelist_zero_padded_range() -> None:
    assert expand_slurm_nodelist("node[001-003]") == ["node001", "node002", "node003"]


def test_expand_slurm_nodelist_raises_on_unmatched_bracket() -> None:
    with pytest.raises(ValueError, match=r"unmatched"):
        expand_slurm_nodelist("node[01-03")


def test_expand_slurm_nodelist_raises_on_unexpected_close_bracket() -> None:
    with pytest.raises(ValueError, match=r"unexpected '\]'"):
        expand_slurm_nodelist("node]")


def test_render_nodes_inventory_uses_default_policy_when_no_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "nodes.yaml"  # does not exist

    inventory = render_nodes_inventory(
        hostnames=["worker-01"],
        source_script=tmp_path / "job.sh",
        output_path=output,
        id_template="{hostname}",
        address_template="{hostname}",
        cloud="aws",
        backends=["docker"],
        auth_token_env=None,
    )

    assert inventory["policy"] == DEFAULT_POLICY.model_dump(mode="json")


def test_render_nodes_inventory_omits_optional_fields_when_none(tmp_path: Path) -> None:
    output = tmp_path / "nodes.yaml"  # does not exist

    inventory = render_nodes_inventory(
        hostnames=["worker-01"],
        source_script=tmp_path / "job.sh",
        output_path=output,
        id_template="{hostname}",
        address_template="{hostname}",
        cloud=None,
        backends=[],
        auth_token_env=None,
    )

    node = inventory["nodes"][0]
    assert "cloud" not in node
    assert "backends" not in node
    assert "auth_token_env" not in node
    assert "sysbox" not in node  # ordinary node carries no sysbox marker


def test_render_nodes_inventory_marks_sysbox_by_hostname_or_id(tmp_path: Path) -> None:
    """--sysbox-node matches the raw hostname OR the generated id, and emits
    ``sysbox: true`` only on those entries."""
    output = tmp_path / "nodes.yaml"  # does not exist

    inventory = render_nodes_inventory(
        hostnames=["ip-10-0-1-2", "ip-10-0-3-4"],
        source_script=tmp_path / "job.sh",
        output_path=output,
        id_template="aws-{hostname}",
        address_template="{address}",
        cloud="aws",
        backends=["docker"],
        auth_token_env=None,
        sysbox_nodes={"ip-10-0-1-2", "aws-ip-10-0-3-4"},  # one by host, one by id
    )
    marked = {n["id"] for n in inventory["nodes"] if n.get("sysbox")}
    assert marked == {"aws-ip-10-0-1-2", "aws-ip-10-0-3-4"}


def test_render_nodes_inventory_merges_allowed_runtimes(tmp_path: Path) -> None:
    """--allowed-runtime additively merges into policy.allowed_runtimes without
    dropping an existing entry."""
    output = tmp_path / "nodes.yaml"
    output.write_text(
        "version: 1\n"
        "nodes: []\n"
        "policy:\n"
        "  allowed_runtimes: [kata-runtime]\n",
        encoding="utf-8",
    )
    inventory = render_nodes_inventory(
        hostnames=["worker-01"],
        source_script=tmp_path / "job.sh",
        output_path=output,
        id_template="{hostname}",
        address_template="{hostname}",
        cloud=None,
        backends=["docker"],
        auth_token_env=None,
        allowed_runtimes=["sysbox-runc"],
    )
    assert inventory["policy"]["allowed_runtimes"] == ["kata-runtime", "sysbox-runc"]


def test_render_nodes_inventory_preserves_sysbox_across_regen(tmp_path: Path) -> None:
    """A ``sysbox: true`` marker in the existing destination file survives a
    regeneration even without re-passing --sysbox-node (same contract as
    ``policy:``)."""
    output = tmp_path / "nodes.yaml"
    output.write_text(
        "version: 1\n"
        "nodes:\n"
        "  - id: aws-ip-10-0-1-2\n"
        "    sysbox: true\n"
        "  - id: aws-ip-10-0-3-4\n"
        "policy: {}\n",
        encoding="utf-8",
    )
    inventory = render_nodes_inventory(
        hostnames=["ip-10-0-1-2", "ip-10-0-3-4"],
        source_script=tmp_path / "job.sh",
        output_path=output,
        id_template="aws-{hostname}",
        address_template="{address}",
        cloud="aws",
        backends=["docker"],
        auth_token_env=None,
        sysbox_nodes=None,  # NOT re-passed
    )
    marked = {n["id"] for n in inventory["nodes"] if n.get("sysbox")}
    assert marked == {"aws-ip-10-0-1-2"}


def test_cmd_nodes_from_slurm_writes_nodes_yaml(tmp_path: Path) -> None:
    script = tmp_path / "workers.sh"
    output = tmp_path / "generated" / "nodes.yaml"
    script.write_text("#SBATCH --nodelist=ip-10-0-7-8,node02\n", encoding="utf-8")
    out = io.StringIO()

    exit_code = cmd_nodes_from_slurm(
        slurm_script=script,
        output=output,
        id_template="cluster-{hostname}",
        address_template="{address}",
        cloud="aws",
        backends=["docker"],
        auth_token_env="XRLENV_NODE_TOKEN",
        out=out,
    )

    assert exit_code == 0
    assert "wrote 2 nodes" in out.getvalue()
    body = output.read_text(encoding="utf-8")
    assert "&id" not in body
    assert "*id" not in body
    raw = yaml.safe_load(body)
    assert set(raw) == {"version", "nodes", "policy"}
    assert raw["nodes"][0]["id"] == "cluster-ip-10-0-7-8"
    assert raw["nodes"][0]["address"] == "10.0.7.8"
    assert raw["nodes"][1]["id"] == "cluster-node02"
    assert raw["nodes"][1]["address"] == "node02"


# ── per-node runtime concurrency cap (sysbox-fs wedge prevention) ──────────────


def test_sysbox_max_concurrent_flag_stamps_cap_on_pool_nodes(
    tmp_path: Path,
) -> None:
    """--sysbox-max-concurrent stamps max_concurrent_by_runtime on Sysbox-pool
    nodes only; non-sysbox nodes are untouched (unlimited)."""
    inventory = render_nodes_inventory(
        hostnames=["ip-10-0-1-2", "ip-10-0-3-4"],
        source_script=tmp_path / "job.sh",
        output_path=tmp_path / "nodes.yaml",  # does not exist
        id_template="aws-{hostname}",
        address_template="{address}",
        cloud="aws",
        backends=["docker"],
        auth_token_env=None,
        sysbox_nodes={"ip-10-0-1-2"},
        sysbox_max_concurrent=8,
    )
    by_id = {n["id"]: n for n in inventory["nodes"]}
    assert by_id["aws-ip-10-0-1-2"]["max_concurrent_by_runtime"] == {
        "sysbox-runc": 8,
    }
    assert "max_concurrent_by_runtime" not in by_id["aws-ip-10-0-3-4"]


def test_sysbox_max_concurrent_omitted_leaves_no_cap(tmp_path: Path) -> None:
    inventory = render_nodes_inventory(
        hostnames=["ip-10-0-1-2"],
        source_script=tmp_path / "job.sh",
        output_path=tmp_path / "nodes.yaml",
        id_template="aws-{hostname}",
        address_template="{address}",
        cloud="aws",
        backends=["docker"],
        auth_token_env=None,
        sysbox_nodes={"ip-10-0-1-2"},
        # no sysbox_max_concurrent
    )
    assert "max_concurrent_by_runtime" not in inventory["nodes"][0]


def test_runtime_cap_preserved_across_regen_and_wins_over_flag(
    tmp_path: Path,
) -> None:
    """An operator's per-node max_concurrent_by_runtime in the destination file
    survives a regen (like sysbox / policy) AND wins over the flag default."""
    output = tmp_path / "nodes.yaml"
    output.write_text(
        "version: 1\n"
        "nodes:\n"
        "  - id: aws-ip-10-0-1-2\n"
        "    sysbox: true\n"
        "    max_concurrent_by_runtime:\n"
        "      sysbox-runc: 12\n"      # operator override
        "policy: {}\n",
        encoding="utf-8",
    )
    inventory = render_nodes_inventory(
        hostnames=["ip-10-0-1-2"],
        source_script=tmp_path / "job.sh",
        output_path=output,
        id_template="aws-{hostname}",
        address_template="{address}",
        cloud="aws",
        backends=["docker"],
        auth_token_env=None,
        sysbox_nodes=None,          # NOT re-passed — preserved from file
        sysbox_max_concurrent=8,    # flag default is 8, but 12 must win
    )
    node = inventory["nodes"][0]
    assert node["sysbox"] is True   # marker also preserved
    assert node["max_concurrent_by_runtime"] == {"sysbox-runc": 12}


def test_allowed_host_paths_additively_merged(tmp_path: Path) -> None:
    """--allowed-host-path merges into policy.allowed_host_paths without dropping
    an existing entry (same contract as --allowed-runtime) — the env-driven knob
    that replaces a personal absolute path baked into the roster."""
    output = tmp_path / "nodes.yaml"
    output.write_text(
        "version: 1\n"
        "nodes: []\n"
        "policy:\n"
        "  allowed_host_paths: [/fsx/shared/preexisting]\n",
        encoding="utf-8",
    )
    inventory = render_nodes_inventory(
        hostnames=["worker-01"],
        source_script=tmp_path / "job.sh",
        output_path=output,
        id_template="{hostname}",
        address_template="{hostname}",
        cloud=None,
        backends=["docker"],
        auth_token_env=None,
        allowed_host_paths=["/fsx/shared/evoclaw-golden"],
    )
    assert inventory["policy"]["allowed_host_paths"] == [
        "/fsx/shared/preexisting",
        "/fsx/shared/evoclaw-golden",
    ]


def test_allowed_host_paths_omitted_leaves_policy_untouched(tmp_path: Path) -> None:
    inventory = render_nodes_inventory(
        hostnames=["worker-01"],
        source_script=tmp_path / "job.sh",
        output_path=tmp_path / "nodes.yaml",  # no existing file
        id_template="{hostname}",
        address_template="{hostname}",
        cloud=None,
        backends=["docker"],
        auth_token_env=None,
        # no allowed_host_paths → default policy's value unchanged
    )
    assert inventory["policy"]["allowed_host_paths"] == list(
        DEFAULT_POLICY.allowed_host_paths
    )
