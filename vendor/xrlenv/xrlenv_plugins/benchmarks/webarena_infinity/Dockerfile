# webarena-infinity env-substrate image (xrlenv).
#
# The image bakes ONLY the environment: app servers + the eval runner + Python
# deps + Node. Build/analysis-only trees are pruned to keep it lean — apps/user-manuals/
# (doc source, never served) and the clone .git (see the clone RUN below, ~470 MB).
# It NEVER bakes the answers. The task verifiers
# (apps/*/real-tasks/*.py, apps/*/function-tasks/*.py) and the oracle solvers
# (apps/*/sanity_check_*.py) are STRIPPED from the `substrate` target. Answers
# are injected at runtime ONLY when/where needed (oracle validation, or
# post-agent grading) into a throwaway sandbox that is then destroyed — so an
# agent can never read them (audit H1 / D6). The task INSTRUCTIONS
# (real-tasks.json / function-tasks.json) are kept — they're the prompt.
#
# Targets:
#   substrate (default) — ANSWER-FREE; the image you run / distribute.
#   full                — answers present; used ONLY to extract the answers
#                         artifact at build time (see build-image.sh). NEVER
#                         run this as a rollout image.
#
# Pin WEBARENA_REF to a commit SHA for any real run. Base from AWS ECR Public
# (Docker Hub anonymous pulls 429 on the cluster; no pull-through mirror).
ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim

# ── full: clone + deps (answers present) — extraction source only ───────────
FROM ${BASE_IMAGE} AS full
# Pinned — build_plan.yaml carries no build-args, so the default IS the pin.
ARG WEBARENA_REF=1ca77813bf8d8e4011aa3fb1f740f70d78bd522d
ARG WEBARENA_REPO=https://github.com/web-arena-x/webarena-infinity
ARG BROWSER_USE_VERSION=0.11.9
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# git: clone. Node: the oracle runs the frontend's data.js through Node to
# compute seed state AND the Monet browser agent (--model monet) runs as a Node
# CLI in-container — Monet requires Node >= 20.5 (node:events addAbortListener),
# so install Node 20.x from NodeSource rather than Debian's older default.
# curl/ca-certs/gnupg for the NodeSource setup + tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version
RUN pip install --no-cache-dir uv

WORKDIR /opt
# Clone at the pin, then prune trees that are never served/read at runtime so the
# distributed image isn't bloated. CRITICAL: prune in THIS layer (the one that creates
# them). `substrate` is FROM `full`, so a later `rm` there only writes a whiteout — the
# bytes survive in this lower layer and nothing is reclaimed; it must happen here.
#   - apps/user-manuals/ (~264 MB): the doc SOURCE used by the build pipeline
#     (infra/pipeline.py) + offline analysis (bench/). App servers never serve it
#     (serve_all.py SKIP_DIRS), and each app's docs are already baked into its own files.
#   - .git (~206 MB): clone history — nothing at runtime shells out to git.
RUN git clone "${WEBARENA_REPO}" webarena-infinity \
    && git -C webarena-infinity checkout "${WEBARENA_REF}" \
    && rm -rf webarena-infinity/apps/user-manuals webarena-infinity/.git \
    && test ! -e webarena-infinity/apps/user-manuals -a ! -e webarena-infinity/.git \
    && echo "pruned apps/user-manuals + .git (~470 MB) from image"

WORKDIR /opt/webarena-infinity
# Deps installed explicitly: upstream runs as scripts by path, not an imported
# package. browser-use (pinned) pulls its CDP browser layer (cdp-use); requests
# is the oracle's HTTP client; pillow/rich/google-genai/openai = upstream's
# vision + dashboard extras.
RUN uv pip install --system \
        "browser-use==${BROWSER_USE_VERSION}" \
        requests pillow rich google-genai openai

# Chromium for the in-container browser agent. browser-use 0.11.9 launches a
# Playwright-managed Chromium by default (BrowserChannel.CHROMIUM) and drives it
# over CDP — `cdp-use` is only the protocol client, NOT a browser, so a real
# Chromium binary + its shared libraries must be present. The slim base ships
# neither. `playwright install --with-deps chromium` fetches the pinned Chromium
# build AND apt-installs the OS libs it needs (libnss3, libatk, fonts, …), the
# same provisioning the repo's local setup.sh does. Inherited by `substrate`.
# The vision agents (vision_agents.py) launch the same Playwright Chromium.
RUN uv pip install --system playwright \
    && playwright install --with-deps chromium

# ── substrate: strip the answers → the real, distributed image ──────────────
FROM full AS substrate
# Remove every verifier dir + oracle solver. Keep *-tasks.json (instructions).
# Build fails if any answer file survives.
RUN cd /opt/webarena-infinity \
    && find apps -type d \( -name real-tasks -o -name function-tasks \) -prune -exec rm -rf {} + \
    && find apps -name 'sanity_check_*.py' -delete \
    && REMAIN=$(find apps \( -name 'sanity_check_*.py' -o -type d -name real-tasks -o -type d -name function-tasks \) | wc -l) \
    && test "$REMAIN" -eq 0 \
    && echo "answers stripped from image (0 remain)"

# Build-time sanity: agent lib + oracle HTTP client import; Node present.
RUN python -c "import browser_use, requests; print('browser-use ok')" \
    && node --version

# No CMD. The answer-free image runs e.g. the app server / (W3) the browser
# agent. Answers are injected at runtime via put_archive, never present here.
