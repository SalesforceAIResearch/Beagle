# echo_bench per-instance image.
#
# Identical Dockerfile across all 3 echo_bench instances — the
# instance-specific data (target string) flows through init_params,
# not the image. Single-stage build, no extra deps beyond the
# python:3.12-slim base.
#
# Tagged as ``echo-bench/<instance_id>:0.1`` by build-task-images.sh.
# The xrlenv stub-runtime layer adds python3 + xrlenv core as a
# wrapper layer; this image only carries the OS + Python runtime
# the stub needs.

FROM python:3.12-slim

# tini is provided by docker's --init flag at run time, no need to
# install it here. /tmp is writable by default on Linux containers,
# so the adapter's state-file writes need no setup.

# Workdir matters only as a courtesy default — the adapter writes
# to /tmp/echo_bench/ regardless.
WORKDIR /opt/echo_bench

# The xrlenv backend mounts xrlenv + xrlenv_plugins into
# /opt/xrlenv-pkg, runs ``python3 -m xrlenv.sandbox_stub``, which
# imports the EnvAdapter named in the manifest. Nothing more to do
# here.
