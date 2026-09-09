from __future__ import annotations

from types import SimpleNamespace

from beagle.rollout.runtime.runtime import ContainerHandle, LocalDockerRuntime


def test_exec_passes_environment_values_out_of_argv(monkeypatch) -> None:
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["env"] = kwargs.get("env")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("beagle.rollout.runtime.runtime.subprocess.run", fake_run)
    secret = "sk-secret-that-must-not-reach-ps"
    result = LocalDockerRuntime().exec(
        ContainerHandle(container_id="cid", name="n"),
        ["sh", "-lc", "true"],
        env={"BEAGLE_GATEWAY_CONFIG": secret},
    )

    assert result.ok
    assert secret not in " ".join(seen["argv"])
    assert seen["argv"][:4] == ["docker", "exec", "-e", "BEAGLE_GATEWAY_CONFIG"]
    assert seen["env"]["BEAGLE_GATEWAY_CONFIG"] == secret
