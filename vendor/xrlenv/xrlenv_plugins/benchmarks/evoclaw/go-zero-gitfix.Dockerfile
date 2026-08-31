# go-zero corrected BASE image (the ONE image xrlenv builds for EvoClaw).
#
# Why: EvoClaw's published `hyd2apse/go-zero:base-v0.9` ships WITHOUT `.git`
# (go-zero's `.dockerignore` excludes it, and — unlike the milestone Dockerfiles,
# which `COPY .git` back — the base build did not). With no repo, EvoClaw's own
# setup git-truncation + any agent's `git tag` fail and the run stalls at
# `Completed: 0`. This grafts `.git` back from a milestone image.
#
# Every OTHER EvoClaw image (milestone + agent-base `hyd2apse/*`) is EvoClaw's own,
# published to Docker Hub by EvoClaw's `scripts/pull_images.sh` and PULLED by the
# cluster mirror — xrlenv does not build those. This is the sole exception.
#
# Build + push to the private registry (run from the xrlenv repo; source .env for
# the registry host — the tag is registry-agnostic on purpose, host prepended at
# push time, so nothing bakes a registry IP into the cache):
#
#   docker build -f xrlenv_plugins/benchmarks/evoclaw/go-zero-gitfix.Dockerfile \
#     -t "${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}/go-zero:base-v0.9-gitfix" .
#   docker push "${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}/go-zero:base-v0.9-gitfix"
#
# Then point the sweep at it (README §Images): set
#   EVOCLAW_GOZERO_BASE_IMAGE=<registry-host>:<port>/go-zero:base-v0.9-gitfix
# in the EvoClaw checkout's .env_private. image_resolution.py redirects ONLY
# go-zero's `base` ref to this image; its milestone images are untouched.
#
# If a go-zero milestone needs the offline dependency closure (the quarantine forces
# GOPROXY=off), build FROM the offline base instead: `hyd2apse/go-zero:base-offline-v0.9`.
FROM hyd2apse/go-zero:base-v0.9

# Graft .git from a milestone image (any go-zero milestone image has it) so the
# base has a real repo again.
COPY --from=hyd2apse/go-zero:m005-v0.9 /testbed/.git /testbed/.git

RUN cd /testbed && git config --global --add safe.directory /testbed \
 && git checkout -f -B main v1.6.0 \
 && git for-each-ref --format='%(refname)' refs/heads | grep -vx refs/heads/main | xargs -r -n1 git update-ref -d \
 && git tag -l | xargs -r git tag -d \
 && git reflog expire --all --expire=now && git gc --prune=now
