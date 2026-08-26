# trace_analyzer

A generic agent-trajectory analyzer, built on the
[agent-debugger](https://dawning-road.github.io/blog/agent-debugger) /
[`adb`](https://github.com/china-qijizhifeng/agentic-harness-engineering/tree/main/agents/evolve_agent/skills/agent-debugger-cli)
design. It normalizes any agent's run into one agent-agnostic model, then runs
**QC** (find issues) and **QA** (ask questions) over it.

## Pipeline

```
raw run ──[TrajectoryNormalizer]──▶ CanonicalTrajectory ──┬─▶ QC: proposers → filters → dedup   (check)
  (monet stream-json,                (turns / tool calls /  ├─▶ QA: ask an LLM, evidence-cited    (ask)
   openai messages, …)                usage / terminal)     └─▶ summarize()  (stats)
```

Everything downstream reads only `CanonicalTrajectory` (`model.py`); its
`messages()` gives the numbered `role`/`content` view proposers cite by
`message_index`. Adding an agent is just a new normalizer.

## QC = proposers → filters → mergers

Following the blog, QC is **not** a single agent or a fixed rules engine. It's a
fan-out of **proposers** (each proposes issues under a rubric), then
**post-processing** that removes known false positives and dedups:

- **Rule proposers** — deterministic, offline, high-precision, for *locally
  visible* pathologies: `tool_error`, `incomplete_run`, `early_truncation`,
  `behavioral_loop`, and `premature_completion` (the high-precision slice:
  declares success while its own last test run is red).
- **LLM proposers** — a rubric prompt applied to each chunk, in parallel, for
  *semantic* categories that can't be computed: `instruction_not_followed_llm`
  (drops the task), `premature_completion_llm` (stops/claims done without
  adequate verification), `fabricated_facts_llm`, and `no_progress`/should-replan.
  They run **only** when a model is configured.
- **Filters** — drop the false positives the blog names: failing unit tests are
  *normal* (not tool errors), models "stuck in 2024" mislabel 2025/26 dates as
  fake, irrelevant search results are expected. Then a `dedup` merger.

The taxonomy is the blog's general one, with two refinements that proved to be
distinct failures with distinct fixes: `premature_completion` (the agent thinks
it's done but isn't) is split out from `instruction_not_followed` (the agent
drops the task), and `incomplete_run` (the run never finished) is split from
`early_truncation` (the model truncated a reply mid-turn). It's **config-driven**:
a profile is just a list of proposers + filters + mergers (`configs/*.yaml`), so
bringing your own taxonomy or tuning rubrics per agent/benchmark is the intended
extension point.

## CLI

```bash
# QC — rule proposers only (offline); add --llm to also run the semantic proposers
python -m trace_analyzer check <trace> [--config default|monet|PATH] [--llm] [--format json]

# QA — ask a question; cites [trace_id #message_index]
python -m trace_analyzer ask <trace> [<trace> ...] -q "Why did this run fail?"

# stats / message export
python -m trace_analyzer summarize <trace> [--format json]
python -m trace_analyzer normalize <trace> [-o out.messages.jsonl]

# inspect a profile
python -m trace_analyzer profiles --config default
```

`--config default` is the general profile; `--config monet` is the same taxonomy
with rubrics tuned to monet-on-coding-benchmarks (no extra thresholds). Without
`--llm`, only the deterministic rule proposers run and the skipped LLM proposers
are reported.

### Example (rule-only, offline)

```
$ python -m trace_analyzer check samples/astropy__astropy-8872.trajectory.jsonl
astropy__astropy-8872: 1 issue(s)  [config=default]
  [HIGH  ] premature_completion (msg 18) declares success but the last test run shows failures
           ↳ Command exited with status 1 ... [ 80%] ...FFF.EEE.F...
           · proposer=premature_completion
  proposers not run: instruction_not_followed_llm (needs LLM), premature_completion_llm (needs LLM), fabricated_facts_llm (needs LLM), no_progress (needs LLM)
```

## Sample run

`samples/` holds 8 real monet trajectories — **6 graded-failed + 2 graded-passed**,
across SWE-bench Verified and Terminal-Bench v2.1 (`samples/manifest.json` records
each one's task id, source run, and grade). Reproduce the output offline:

```bash
for f in trace_analyzer/samples/*.trajectory.jsonl; do
  python -m trace_analyzer check "$f"
done
```

Rule-only QC (no `--llm`) over the eight, with each one's true grade:

| trajectory | grade | turns | rule-only `check` finds |
|---|---|---|---|
| `astropy__astropy-7336` | PASS | 13 | — none — |
| `astropy__astropy-12907` | PASS | 11 | — none — |
| `astropy__astropy-8872` | FAIL | 10 | `premature_completion` (HIGH) |
| `django__django-13112` | FAIL | 15 | `tool_error` (MEDIUM) |
| `astropy__astropy-14365` | FAIL | 9 | — none — *(needs `--llm`)* |
| `db-wal-recovery__s0` | FAIL | 6 | `incomplete_run` (HIGH) |
| `circuit-fibsqrt__s3` | FAIL | 15 | `tool_error` (MEDIUM) |
| `install-windows-3.11__s4` | FAIL | 46 | `tool_error`×4, `behavioral_loop` (MEDIUM) |

A clean pass and a premature-completion failure in detail:

```
$ python -m trace_analyzer check samples/astropy__astropy-7336.trajectory.jsonl
astropy__astropy-7336: 0 issue(s)  [config=default]
  proposers not run: instruction_not_followed_llm (needs LLM), premature_completion_llm (needs LLM), fabricated_facts_llm (needs LLM), no_progress (needs LLM)

$ python -m trace_analyzer check samples/astropy__astropy-8872.trajectory.jsonl
astropy__astropy-8872: 1 issue(s)  [config=default]
  [HIGH  ] premature_completion (msg 18) declares success but the last test run shows failures
           ↳ Command exited with status 1 ... [ 80%] ...FFF.EEE.F...
           · proposer=premature_completion
```

The `install-windows-3.11__s4` run (a 46-turn thrash) shows several categories at
once — repeated `image_read` calls that never converge, plus `SIGTERM`/timeout
tool errors:

```
$ python -m trace_analyzer check samples/install-windows-3.11__s4.trajectory.jsonl
install-windows-3.11__s4: 5 issue(s)  [config=default]
  [MEDIUM] tool_error      (msg 10) bash call returned an error
           ↳ Error: Killed by signal SIGTERM. Treat the output as incomplete...
  [MEDIUM] behavioral_loop (msg 43) recurring repetition of the same tool call ×9 (no apparent progress)
           ↳ image_read:{"detail": "low", "max_dimension": 1024, "path": "/tmp/win311.ppm"}
  ... (3 more tool_error)
```

**The instructive case is `astropy__astropy-14365`: it FAILED, but rule-only QC
finds nothing.** It's a "wrong-but-clean" run — the agent made the QDP command
parser case-insensitive only for one regex and validated on the issue's literal
example, so there's no *structural* symptom (no error, no loop, no truncation,
no red test). That's exactly the semantic residue the rule proposers can't see;
running with `--llm` lets the `premature_completion_llm` proposer flag the
shallow verification. Conversely, note the HIGH `premature_completion` hit fires
on a *failed* run here, but the same rule can fire on a passed run whose last
visible test was a transient red — proposers surface **candidates**, not verdicts,
which is why a filter/judge stage (and the real grade) sits downstream.

## Design notes

- **Rule vs LLM proposers** split the work along the right axis: deterministic
  rules are free, reproducible, and exhaustive for *structurally visible* issues
  (a red test, a missing terminal, a repeated command); the LLM handles the
  *semantic* residue (claim-vs-reality, did-it-actually-verify, no-progress) that
  rules can't see without overfitting. A rule proposer is the highest-precision
  proposer type; an LLM proposer is the most general.
- **Why not bake in monet's G1–G4 thresholds?** They're calibrated to one agent
  on coding benchmarks. The *generalizable essence* of those findings lives in
  the general categories (premature termination + shallow verification →
  `premature_completion`; thrash/no-replan → `behavioral_loop`/no-progress),
  detected by rubric, not by a hardcoded "30 turns" / "ran pytest". Long-run /
  big-context stats stay in `summarize` as numbers to bucket on, not as issues.
- **QA `ask`** map-reduces long traces (extract relevant evidence per chunk, then
  answer) to blunt the "context rot" the blog warns about; short traces answer in
  one shot.

## Adding a normalizer for another agent

```python
from trace_analyzer.normalizer import TrajectoryNormalizer, register
from trace_analyzer.model import CanonicalTrajectory, Turn, ToolCall

class MyAgentNormalizer(TrajectoryNormalizer):
    name = "myagent"
    def normalize(self, path) -> CanonicalTrajectory: ...      # parse → Turns/ToolCalls
    @classmethod
    def sniff(cls, path) -> bool: ...                          # cheap peek for --source auto

register(MyAgentNormalizer())
```

List it in `normalizers/__init__.py` so `import trace_analyzer` discovers it.

## Note on the monet normalizer

`normalizers/monet.py` reduces monet's `--output-format stream-json` directly
rather than reusing `agents/monet/trajectory.py`: current monet builds emit
`text_delta` events **without an `index`** field, which that reducer keys on, so
it drops all assistant free-text on today's traces. This normalizer captures
text whether or not an `index` is present.
