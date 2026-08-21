# Slice-1 smoke image. Baked-in deps: python + aiohttp + pyyaml.
# The xrlenv package itself is bind-mounted at /opt/xrlenv-pkg/xrlenv by the
# Docker backend at create time, so we never have to rebuild this image when
# the stub source changes during development.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "aiohttp>=3.9" "pyyaml>=6.0" "pydantic>=2.7"

RUN mkdir -p /sandbox /run/xrlenv \
 && useradd --create-home --shell /bin/bash --uid 1000 sandbox \
 && chown -R sandbox:sandbox /sandbox /run/xrlenv

WORKDIR /sandbox
USER sandbox

# The Docker backend overrides the command at create time:
#   python -m xrlenv.sandbox_stub --uds /run/xrlenv/stub.sock
# This CMD is a fallback for `docker run` debugging.
CMD ["python", "-c", "print('hello-shell base image; the node agent overrides CMD at create time')"]
