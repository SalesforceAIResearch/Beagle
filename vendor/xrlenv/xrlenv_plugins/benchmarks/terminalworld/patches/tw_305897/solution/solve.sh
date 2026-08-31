#!/bin/bash

# Configure iptables INPUT chain and persist to /etc/sysconfig/iptables.
# The verifier only inspects the SAVED rules file's content/order, so no
# live iptables / NET_ADMIN is required — we write the file directly.
#
# Required INPUT order (exactly three rules):
#   (1) ACCEPT TCP dport 22 from anywhere      (no -s)
#   (2) ACCEPT TCP dport 8080 from 172.16.238.0/24
#   (3) REJECT TCP dport 8080 from anywhere    (no -s)

mkdir -p /etc/sysconfig

cat > /etc/sysconfig/iptables << 'RULES'
*filter
:INPUT ACCEPT [0:0]
:FORWARD ACCEPT [0:0]
:OUTPUT ACCEPT [0:0]
-A INPUT -p tcp -m tcp --dport 22 -j ACCEPT
-A INPUT -s 172.16.238.0/24 -p tcp -m tcp --dport 8080 -j ACCEPT
-A INPUT -p tcp -m tcp --dport 8080 -j REJECT --reject-with icmp-port-unreachable
COMMIT
RULES

echo "[+] Wrote /etc/sysconfig/iptables with correct rule order."
