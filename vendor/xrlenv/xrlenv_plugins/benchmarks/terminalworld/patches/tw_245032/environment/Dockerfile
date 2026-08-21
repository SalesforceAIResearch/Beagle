# Fix (2026-07-08): build openal-soft at the 1.25.1 tag the verifier expects.
# Two coupled changes vs. the upstream Dockerfile:
#   1. Pin `--branch 1.25.1` (upstream clones `--depth 1` of master, which
#      drifted to build libopenal.so.1.25.2; the verifier hard-codes 1.25.1).
#   2. Base ubuntu:24.04 (default g++-13) instead of ubuntu:22.04 + g++-12.
#      openal-soft 1.25.1's alformat.hpp uses C++20 `#include <format>`, which
#      g++-12's libstdc++ lacks ("fatal error: format: No such file or
#      directory"); GCC 13 ships it. Master builds on g++-12 only via its fmt
#      fallback — so pinning 1.25.1 REQUIRES the newer toolchain. This is the
#      environment the task author must have used to author a 1.25.1 verifier.
# The BackendListEnd fix under test is present in 1.25.1. See STATUS.md.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        binutils-gold \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        g++ \
        gcc \
        gdb \
        git \
        libasound2-dev \
        libdbus-1-dev \
        libjack-dev \
        libpulse-dev \
        make \
        pkg-config \
        portaudio19-dev \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Clone repository at the 1.25.1 tag (builds libopenal.so.1.25.1).
RUN git clone --branch 1.25.1 --depth 1 https://github.com/kcat/openal-soft /home/user/openal-soft

WORKDIR /home/user/openal-soft

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
SHELL ["/bin/bash", "-c"]
CMD ["bash"]
