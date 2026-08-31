# `scripts/onboard/` — stand up an agent's experiment copy

`onboard_agent.py` onboards an evolvable agent (monet today) from an upstream git
source. It's a thin shim for `python -m beagle.tools.onboard` and is **agent-agnostic**
— the inputs are the source, the ref, the GitHub repo to create, and the local dir for
the copy:

```bash
# token comes from the env (a credential) — source your .env first
set -a; source .env; set +a
# example — pin a COMMIT SHA, not a branch/tag (they drift; a re-onboard would snapshot a
# different baseline). This is upstream monet's monet_code_20260816 @ this commit.
.venv/bin/python scripts/onboard/onboard_agent.py \
    --upstream https://github.com/<your-org>/monet_code --ref 1261608f6530908e3a03218d8f4671b8c7b5b346 \
    --repo <your-org>/monet_code_20260816 --private --branch-name baseline \
    --dir ../beagle-experiments/monet_code_20260816
```

- `--repo <org/name>` — the **GitHub** repo it creates + pushes to (the experiment
  copy the evolver later pushes candidate branches to).
- `--dir <path>` — the **local** git-tracked working copy it checks out (`origin` =
  your GitHub repo, `upstream` = the source; default `.beagle/agents/<profile>`).

It resolves the upstream SHA, seeds the GitHub repo with a **single parentless commit** (that
upstream tree, no history) on a `baseline` branch (`--branch-name` to override — e.g. mirror the
snapshot `opencode_v1.18.16`), checks out the local copy, and **writes the agent-source pointer**
`{repo, ref, version[, token_env]}` to
`.beagle/agents/<profile>.json` (`--profile-name`, default the repo name; `ref` is the copy's baseline
commit, with `upstream_ref` + `branch` for provenance) — downstream tasks discover it by profile, no
hand-copying. That pointer is *all* onboarding produces; it deliberately does **not** take:

| Not an onboarding arg | Why | Where it lives |
|---|---|---|
| entrypoint (`bin/monet.js`) | agent-*intrinsic* | the agent adapter (`MonetAgent._default_source`) |
| `model`, benchmark selection | *eval-run* knobs | the run config |
| gateway creds/provider | *deployment* detail | the run config's `model` block |

See the repo `README.md` → "Onboard an agent and run a baseline" for the filled-in
config and the run command.

## Why no vendoring

The agent's code is **never committed into beagle**. The experiment-copy repo is the
single durable home for the agent code (θ):

- **Baseline eval** doesn't need a local copy — the trial container `git clone`s
  `<experiment-repo>@<sha>` at rollout (the M+N harbor path). The run config just
  points at that repo@sha.
- **Evolution** (Phase E) branches candidates from the local `--dir` copy (or a fresh
  clone) and pushes them to the experiment copy — never to upstream (the
  experiment-copy rule).

A `vendor/<agent>` dir in beagle would bloat the repo, couple git histories, and still
couldn't be cloned by a remote trial container without pushing it to a repo anyway —
which *is* the experiment copy.

## Auth & safety

Uses `gh` + git over HTTPS with the token in `--token-env` (default `GH_TOKEN`). Tokens
are injected into URLs only as command arguments (never written to `.git/config`) and
redacted from all output. `--reseed` re-seeds an existing repo from upstream and is
**destructive** (overwrites candidate branches) — omit it for a normal run.
