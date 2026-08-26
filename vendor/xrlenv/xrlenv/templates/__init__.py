"""Built-in templates that ship with the platform.

Phase-0 carries one template here: ``hello_shell`` — the spine smoke
template that drives ``tests/smoke/cluster_bringup/single_rollout.py`` and the platform's
own integration tests. Benchmark plug-ins live under
:mod:`xrlenv_plugins.benchmarks` instead; the platform never edits
per-benchmark.

The directory was renamed from ``hello-shell`` (kebab) to
``hello_shell`` (snake) so it's a valid Python package and the
adapter can be co-located inside it
(``xrlenv.templates.hello_shell.adapter:ShellEnvAdapter``). The
template's public ``name:`` field stays ``hello-shell``.
"""
