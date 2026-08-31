"""forward_env normalization — bare string (V→V) vs [container, host] pair."""

from __future__ import annotations

from beagle.agents.core.forward_env import normalize_forward_env


def test_string_means_same_name_both_sides() -> None:
    assert normalize_forward_env(["API_KEY", "PROXY_URL"]) == [
        ("API_KEY", "API_KEY"), ("PROXY_URL", "PROXY_URL")]


def test_pair_when_names_differ() -> None:
    assert normalize_forward_env([["CONTAINER_KEY", "HOST_KEY"]]) == [("CONTAINER_KEY", "HOST_KEY")]


def test_dict_form_one_pair_per_item() -> None:
    # The vendored driver's config emits [{container: host}, …] — accept it so a config
    # copied from there forwards its creds instead of silently dropping them.
    assert normalize_forward_env([{"K": "K"}, {"C": "H"}]) == [("K", "K"), ("C", "H")]
    assert normalize_forward_env([{"A": "A", "B": "B"}]) == [("A", "A"), ("B", "B")]


def test_mixed_and_edge_cases(caplog) -> None:
    assert normalize_forward_env(["A", ["B_IN", "B_OUT"], ["C"]]) == [
        ("A", "A"), ("B_IN", "B_OUT"), ("C", "C")]
    assert normalize_forward_env(None) == []
    assert normalize_forward_env([]) == []
    # Unrecognized entries are still skipped — but now LOUDLY (a silent drop is a cred that
    # never reaches the container, which cost a live run to diagnose).
    with caplog.at_level("WARNING"):
        assert normalize_forward_env([["too", "many", "parts"]]) == []
    assert "skipping unrecognized entry" in caplog.text
