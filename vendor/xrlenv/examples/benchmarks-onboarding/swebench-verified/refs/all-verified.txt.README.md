# Generating `all-verified.txt`

The full 500-instance Verified ref list is generated from the
upstream dataset rather than vendored — the canonical source is
HuggingFace `SWE-bench/SWE-bench_Verified` and we don't want
the file to drift from the dataset.

To generate the full list (run on any host with `swebench` installed):

```bash
.venv/bin/python -c "
from swebench.harness.run_evaluation import load_swebench_dataset
for inst in load_swebench_dataset('SWE-bench/SWE-bench_Verified', 'test'):
    instance_id = inst['instance_id'].replace('__', '_1776_').lower()
    image = f'swebench/sweb.eval.x86_64.{instance_id}:latest'
    # 3 GiB conservative size hint
    print(f'{image}\\t3221225472')
" > examples/benchmarks-onboarding/swebench-verified/refs/all-verified.txt
```

Then the operator runs:

```bash
xrlenv images plan \
    --refs examples/benchmarks-onboarding/swebench-verified/refs/all-verified.txt \
    --eager-prefetch
```

`--eager-prefetch` is recommended for the full sweep so all 500
images arrive at their preferred-home nodes before the first
acquire (otherwise the first 500 acquires each pay a serial
cold-pull penalty).

For the default 8-instance smoke, `refs/smoke-8.txt` is shipped
in-repo — no generation step needed.
