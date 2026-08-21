"""Egress schema + compiler + netns guard (spec 07).

Pure, no root / no daemon. Pins the exact iptables program the enforcer will
install, so a regression (e.g. metadata DROP moved after an ACCEPT, or the v6
lockdown dropped) fails here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from xrlenv.backends.egress import (
    DEFAULT_METADATA_IPS,
    EgressAllowlist,
    EgressRule,
    compile_egress_rules,
    container_can_escape_egress,
    is_shared_netns,
)

_REJECT4 = ["-A", "OUTPUT", "-j", "REJECT", "--reject-with", "icmp-admin-prohibited"]
_REJECT6 = ["-A", "OUTPUT", "-j", "REJECT", "--reject-with", "icmp6-adm-prohibited"]


# ── schema ────────────────────────────────────────────────────────────────────


def test_rule_cidr_only_and_with_ports() -> None:
    assert EgressRule(cidr="internal-ip/8").ports is None
    assert EgressRule(cidr="internal-ip/8", ports=(443, 8443)).ports == (443, 8443)


def test_rule_tolerates_host_bits() -> None:
    assert EgressRule(cidr="internal-ip/8").cidr == "internal-ip/8"


def test_rule_rejects_invalid_cidr() -> None:
    with pytest.raises(ValidationError, match="invalid cidr"):
        EgressRule(cidr="not-a-network")


def test_rule_rejects_ipv6() -> None:
    with pytest.raises(ValidationError, match="IPv6"):
        EgressRule(cidr="2001:db8::/32")


def test_rule_rejects_bad_ports() -> None:
    with pytest.raises(ValidationError, match="out of range"):
        EgressRule(cidr="internal-ip/8", ports=(0,))
    with pytest.raises(ValidationError, match="multiport"):
        EgressRule(cidr="internal-ip/8", ports=tuple(range(1, 17)))


def test_empty_allowlist_is_valid() -> None:
    # Empty = block-all; NOT an error (unlike a non-empty-required schema).
    al = EgressAllowlist()
    assert al.rules == ()


# ── compiler ──────────────────────────────────────────────────────────────────


def test_empty_allowlist_blocks_all() -> None:
    prog = compile_egress_rules(EgressAllowlist())
    assert prog.v4[0] == ["-F", "OUTPUT"]
    assert ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"] in prog.v4
    for mip in DEFAULT_METADATA_IPS:
        assert ["-A", "OUTPUT", "-d", mip, "-j", "DROP"] in prog.v4
    # No ACCEPT to any address — only loopback + reject.
    assert not any(r[-1] == "ACCEPT" and "-d" in r for r in prog.v4)
    assert prog.v4[-1] == _REJECT4


def test_no_broad_established_accept() -> None:
    # Strict: a flow opened before the restriction is NOT blanket-allowed.
    prog = compile_egress_rules(EgressAllowlist(rules=(EgressRule(cidr="internal-ip/8"),)))
    assert not any("conntrack" in r or "ESTABLISHED,RELATED" in r for r in prog.v4)


def test_metadata_dropped_before_accepts() -> None:
    prog = compile_egress_rules(EgressAllowlist(rules=(EgressRule(cidr="0.0.0.0/0"),)))
    accept_idx = prog.v4.index(["-A", "OUTPUT", "-d", "0.0.0.0/0", "-j", "ACCEPT"])
    for mip in DEFAULT_METADATA_IPS:
        assert prog.v4.index(["-A", "OUTPUT", "-d", mip, "-j", "DROP"]) < accept_idx


def test_ports_emit_tcp_udp_multiport() -> None:
    prog = compile_egress_rules(
        EgressAllowlist(rules=(EgressRule(cidr="internal-ip/8", ports=(443,)),)),
    )
    for proto in ("tcp", "udp"):
        assert [
            "-A", "OUTPUT", "-p", proto, "-d", "internal-ip/8",
            "-m", "multiport", "--dports", "443", "-j", "ACCEPT",
        ] in prog.v4


def test_dns_resolver_validated_in_pure_path() -> None:
    # M1 (audit): a malformed / IPv6 resolver fails at compile, before any
    # container is touched — not mid-nsenter after a partial apply.
    al = EgressAllowlist(rules=(EgressRule(cidr="internal-ip/8"),))
    with pytest.raises(ValueError, match="invalid dns_resolver"):
        compile_egress_rules(al, dns_resolver="not-an-ip")
    with pytest.raises(ValueError, match="IPv6"):
        compile_egress_rules(al, dns_resolver="2001:db8::1")
    # A valid v4 resolver compiles fine.
    compile_egress_rules(al, dns_resolver="internal-ip/32")


def test_dns_resolver_opens_53_only_when_given() -> None:
    without = compile_egress_rules(EgressAllowlist(rules=(EgressRule(cidr="internal-ip/8"),)))
    assert not any("53" in r for r in without.v4)
    withdns = compile_egress_rules(
        EgressAllowlist(rules=(EgressRule(cidr="internal-ip/8"),)),
        dns_resolver="internal-ip/32",
    )
    for proto in ("udp", "tcp"):
        assert [
            "-A", "OUTPUT", "-p", proto, "-d", "internal-ip/32",
            "--dport", "53", "-j", "ACCEPT",
        ] in withdns.v4


def test_v6_locked_down() -> None:
    prog = compile_egress_rules(EgressAllowlist(rules=(EgressRule(cidr="internal-ip/8"),)))
    assert prog.v6[0] == ["-F", "OUTPUT"]
    assert ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"] in prog.v6
    assert not any(r[-1] == "ACCEPT" and "-d" in r for r in prog.v6)
    assert prog.v6[-1] == _REJECT6


# ── netns guard ───────────────────────────────────────────────────────────────


def test_is_shared_netns() -> None:
    assert is_shared_netns("host") is True
    assert is_shared_netns("HOST") is True
    assert is_shared_netns("container:abc123") is True
    assert is_shared_netns(" container:x ") is True
    for nm in ("bridge", "default", "none", "", None, "my-net"):
        assert is_shared_netns(nm) is False


def test_container_can_escape_egress() -> None:
    # H1 (audit): privileged or NET_ADMIN means the workload can flush the rules.
    assert container_can_escape_egress({"Privileged": True}) is True
    assert container_can_escape_egress({"CapAdd": ["NET_ADMIN"]}) is True
    assert container_can_escape_egress({"CapAdd": ["CAP_NET_ADMIN"]}) is True
    assert container_can_escape_egress({"CapAdd": ["SYS_PTRACE", "net_admin"]}) is True
    # Safe: default caps, no privileged.
    assert container_can_escape_egress({}) is False
    assert container_can_escape_egress({"Privileged": False, "CapAdd": None}) is False
    assert container_can_escape_egress({"CapAdd": ["SYS_PTRACE", "CHOWN"]}) is False


def test_sysbox_runtime_can_escape_egress() -> None:
    """§6 — a sysbox-runc container's inner root controls its own netns +
    iptables, so it can flush the OUTPUT chain the allowlist installs. The
    ``nsenter``-based egress restriction is NOT a trusted boundary for it —
    ``container_can_escape_egress`` must return True on the Runtime alone."""
    assert container_can_escape_egress({"Runtime": "sysbox-runc"}) is True
    # Even with no privileged / no NET_ADMIN, the runtime is enough.
    assert container_can_escape_egress(
        {"Runtime": "sysbox-runc", "Privileged": False, "CapAdd": []},
    ) is True
    # The ordinary runc runtime stays safe.
    assert container_can_escape_egress({"Runtime": "runc"}) is False
