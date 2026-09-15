# Trusted, online preparation only; the resulting workload runs offline in Kata.
FROM ghcr.io/ggml-org/llama.cpp@sha256:b15cd87030b2eb8a284d398c095e03c6ba8baf386b68e1b83ab8b46e9555ffe8
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv git \
    && python3 -m venv /agent/.venv \
    && /agent/.venv/bin/pip install --no-cache-dir mini-swe-agent==2.4.6 \
    && rm -rf /var/lib/apt/lists/*
COPY mini-config.yaml /agent/kata-mini.yaml
COPY clamp.py /testbed/clamp.py
COPY clamp_checks.py /testbed/test_clamp.py
RUN cd /testbed && printf '__pycache__/\n' > .gitignore \
    && git init -q && git add .gitignore clamp.py test_clamp.py \
    && git -c user.name=beagle -c user.email=agent@beagle.local commit -qm baseline
WORKDIR /testbed
ENTRYPOINT []
CMD ["sleep", "infinity"]
