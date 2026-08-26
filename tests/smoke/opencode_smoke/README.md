# opencode smoke — pure eval

Baseline **opencode** (the open-source Bun coding agent) on a few tasks, via `beagle evaluate`.

```bash
set -a; source .env; set +a                 # gateway creds; gateway up (scripts/gateway/)
cd /path/to/beagle

# terminal-bench-2.1 (2 tasks, harbor)
beagle evaluate --config tests/smoke/opencode_smoke/terminal_bench_2_1_smoke2.yaml --dry-run   # preview
beagle evaluate --config tests/smoke/opencode_smoke/terminal_bench_2_1_smoke2.yaml             # run

# swe-bench-verified (2 tasks, harbor)
beagle evaluate --config tests/smoke/opencode_smoke/swebench-verified_smoke2.yaml

# deep-swe (2 tasks, pier) — requires the extra: uv pip install -e '.[deep-swe]'
beagle evaluate --config tests/smoke/opencode_smoke/deep-swe_smoke2.yaml
```

Results → `<run.dir>/<run.name>/run.json`. Point `agent.harness.source.{repo,ref}` at your own
onboarded copy to evolve later (the configs default to the public upstream).

**Config shape.** First-level (top of every agent's block, uniform across agents): `model`,
`provider`, `effort`, `max_turns`, `forward_env`, `timeout`. An agent-harness's *own* args live under
`extra_args:`, keyed by `<agent>_args` — here `opencode_args:` (opencode's raw CLI, e.g. `--auto`) —
so a config names which args belong to which agent.

**How opencode runs** (cloned + built from `repo@ref` in-container, then its own headless CLI):
- INSTALL bootstraps Bun + ripgrep and `bun install`s the monorepo; RUN invokes
  `bun packages/opencode/src/index.ts run --format json --model <provider>/<model> [--variant <effort>]
  --auto --dir <repo>` with the prompt piped via stdin.
- The LLM gateway is injected as an `@ai-sdk/openai-compatible` provider through opencode's native
  `OPENCODE_CONFIG_CONTENT` (env-free, provider-neutral); `--model <provider>/<model>` selects it.
- `effort` → `--variant`; opencode has **no turn-cap flag**, so `max_turns` is accepted but a no-op.
- The `--format json` event stream (`opencode.stream.jsonl`) is converted to ATIF
  (`agent/trajectory.json`); the patch is `git diff <base>..HEAD`.

**First run may need tuning** (opencode is a large Bun monorepo):
- `bun install` cost + the `node-pty` native postinstall (needs a C/Python toolchain — the default
  `install_cmd` installs it best-effort across apt/apk),
- `--variant` behavior against a custom OpenAI-compatible provider (the reasoning lever),
- on deep-swe, a base-image apt/apk mirror not in `opencode.install_hosts` will 403 the Bun bootstrap
  through pier's proxy (add the host there).
