# Pruning the opencode experiment copy

opencode's experiment copy is the **whole monorepo** (34 packages), but a headless `opencode run`
only touches ~a dozen of them. The rest — web/desktop/marketing apps, demo videos, test fixtures — is
dead weight. Two sizes matter (don't conflate them):

| | `.git` objects (**git-fetch download**) | working tree (checkout) | **on disk** |
|---|---|---|---|
| full (current) | 79 MB | 184 MB | 262 MB |
| **after D** | **~12 MB** | **~73 MB** | **~86 MB** |

(Measured on a real fresh clone of the pruned seed: `.git` 13 MB + tree 73 MB.)

`du` on the copy shows the **262 MB** on-disk total (objects + checkout). The trial container runs
`git fetch --depth 1`, which transfers only the **`.git` packfile (~79 MB)** — dominated by
non-compressible binary assets — and checks out the working tree locally. That download per container
is the install slowdown.

**Patch-back safety (the hard constraint).** Pruning only ever **deletes whole files/dirs**; it never
edits a kept file. So every kept file stays **byte-identical to upstream**, an evolution diff
(`git diff base..HEAD`) touches only kept files and **applies cleanly to upstream**, and the deletions
never show up in that diff. (Removed files also aren't reachable from the evolver's edits, so it can't
depend on them.)

## D — the conservative prune (dependency-safe; done first)

Removes only what is outside the headless-run dependency closure:

| removed | size | what it is |
|---|---|---|
| `packages/console` | 55 MB | web dashboard (~40 MB of marketing `.mp4`) |
| `packages/web` | 17 MB | marketing website |
| `packages/app` **(except `vendor/`)** | ~15 MB | web app — **but keep `app/vendor/opencode-ai-client-1.17.13-v2.tgz` (67 KB)** |
| `packages/desktop` | 13 MB | Electron desktop app (1 MB `.icns` icons) |
| `artifacts/` | 7.4 MB | demo videos |
| `screenshot-uk.png` + `README.<lang>.md` (~20) | ~0.2 MB | marketing image + translated READMEs |

> **Gotcha (found empirically, not by static analysis):** `packages/session-ui` — which D keeps — depends
> on `@opencode-ai/client` via `file:../app/vendor/opencode-ai-client-1.17.13-v2.tgz`, a 67 KB vendored
> tarball *inside* `packages/app`. Delete all of `app` and `bun install` fails
> (`ENOENT extracting tarball from @opencode-ai/client`). So keep that one tarball; drop the rest of `app`.
> (This is a *different* `@opencode-ai/client` than the runtime one: `sdk-next` uses `workspace:*` →
> `packages/client`, unaffected.)

**Result:** per-container **download (git packfile) 79 MB → ~12 MB (~6.5×)**; on-disk 262 MB → ~86 MB
(the kept 67 KB tarball is noise).

### Verified (bun 1.3.14, the container's pinned version)

On the pruned tree (`packages/app` reduced to just `vendor/`):

- `bun install` → **1539 packages installed, exit 0** (plain `bun install`, no `--frozen-lockfile`, so
  the seed's `bun.lock` referencing the deleted packages is fine — bun re-resolves in-container; the
  shipped `bun.lock`/`package.json` stay byte-identical to upstream ⇒ patch-safe).
- `opencode … run --help` → exit 0, full run-command module graph loads, no import errors.
- `opencode … run --print-logs` → agent **fully boots**: loads tools + session, selects the ai-sdk LLM
  runtime, and makes the real model streaming call (only fails on a deliberately-dead endpoint). **No
  removed package is needed at runtime.**

The earlier segfault was purely a bun **1.2.14 vs required 1.3.14** mismatch — not pruning.

## Kept — the runtime closure

`opencode` + its `workspace:*` deps `core, server, tui, sdk, sdk-next, llm, schema, codemode, plugin,
http-recorder, protocol, script`; the transitive pull-ins `ui` (via `tui`), `client` (via `sdk-next`),
`effect-drizzle-sqlite` / `effect-sqlite-node` (via `core`); and root config (`package.json`,
`bun.lock`, `tsconfig*`, `bunfig.toml`) + `patches/` (bun `patchedDependencies`).

## A — deeper prune candidates (not yet applied; need a `bun install` + `run` check)

Bigger but riskier — verify `bun install` + `bun run packages/opencode/src/index.ts run --format json`
still work on the pruned tree first:

- non-dep packages `stats, session-ui, storybook, docs, enterprise, cli, containers, function,
  httpapi-codegen, identity, slack` (~8 MB; some are **explicit** workspace entries, so `bun.lock`
  may need regenerating — a baseline-only change, still patch-safe). Note: dropping `session-ui` also
  removes the only reason to keep `app/vendor/…tgz`, so the whole of `packages/app` could then go.
- `packages/opencode/test/` fixtures (~10 MB — a 5 MB base64 PNG, a 2.6 MB image, a 4.7 MB JSON).
- `packages/ui` assets (fonts/sprites, ~20 MB) — droppable **only if** headless `run --format json`
  never loads the TUI.

## How it's applied

Wired at **seed time** in `beagle/tools/onboard.py`. The set lives in `PRUNE_PROFILES["opencode"]`
(a list of git pathspecs); `seed_from_upstream(..., prune=…)` builds the reduced tree **in the index**
(`read-tree` → `git rm -r --cached --ignore-unmatch <pathspecs>` → `write-tree`) and commits that as the
orphan baseline — no working-tree checkout. Pathspecs that match nothing are ignored, so the profile
degrades gracefully if a future opencode version restructures.

Onboard with `--prune opencode`:

```
python -m beagle.tools.onboard --upstream https://github.com/anomalyco/opencode \
    --ref <sha> --repo <you>/opencode_v1.18.16 --private --version 1.18.16 \
    --prune opencode --profile-name opencode_v1.18.16
```

Patch-safety holds because pruning only *removes* whole paths: kept blobs are byte-identical to
upstream (the integration test asserts the kept `packages/core/…` blob hash equals upstream's), so an
evolution `git diff base..HEAD` touches only kept files and applies cleanly to upstream.
