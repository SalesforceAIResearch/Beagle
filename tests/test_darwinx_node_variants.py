

def test_consolidate_requires_a_complexity_measurement_for_this_evolvee():
    """CONSOLIDATE is accepted for lowering complexity, and the driver measures complexity over
    mini-swe-agent's tree by default -- which matches zero files in any other evolvee, so the
    accept rule would rest on an empty measurement while the run looks healthy."""
    import pytest
    from beagle.algorithms.darwinx.config import DarwinXConfig

    with pytest.raises(Exception, match="complexity_code_globs"):
        DarwinXConfig(consolidate_enabled=True)

    cfg = DarwinXConfig(consolidate_enabled=True, complexity_code_globs="packages/opencode/src/**/*.ts")
    env = cfg.to_driver_env()
    assert env["DARWINX_GATE_CONSOLIDATE_ENABLED"] == "1"
    assert env["DARWINX_GATE_COMPLEXITY_CODE_GLOBS"] == "packages/opencode/src/**/*.ts"


def test_node_variant_rates_must_leave_some_additive_nodes():
    import pytest
    from beagle.algorithms.darwinx.config import DarwinXConfig

    with pytest.raises(Exception, match="additive"):
        DarwinXConfig(
            prune_enabled=True, prune_rate=0.9,
            consolidate_enabled=True, consolidate_rate_late=0.9,
            complexity_code_globs="x/**/*.ts",
        )
    with pytest.raises(Exception, match=r"within \[0, 1\]"):
        DarwinXConfig(prune_enabled=True, prune_rate=1.5)


def test_node_variants_are_off_by_default_so_the_control_arm_is_unchanged():
    from beagle.algorithms.darwinx.config import DarwinXConfig

    env = DarwinXConfig().to_driver_env()
    assert "DARWINX_GATE_PRUNE_ENABLED" not in env
    assert "DARWINX_GATE_CONSOLIDATE_ENABLED" not in env
