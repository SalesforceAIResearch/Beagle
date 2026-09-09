# Live provider-routing integration tests

`generate_provider_matrix.py` creates small, operator-local evaluation configs under
`tests/integration/generated/`. Every config runs one DeepSWE task at parallelism one
with `max_turns: 3` and `run.timeout_multiplier: 0.1`.

```bash
python tests/integration/generate_provider_matrix.py --clean
for config in tests/integration/generated/*.yaml; do
  beagle evaluate --config "$config" &
done
```

Sources come from onboarded manifests in `.beagle/agents/`; no repository URL or
credential is baked into the generator. Override the task, model, runtime, agents, routes,
manifest directory, or output directory with the corresponding CLI flags.

OpenCode has no native max-turn/step-cap support. Its generated configs carry a warning and
record `max_turns: 3` for the shared schema, but OpenCode does not enforce it. The `0.1`
timeout multiplier shortens DeepSWE's declared task budget and is the actual bound.



The internal matrix has five runnable cells:

- Monet and OpenCode
- direct API: `provider: {type: direct, name: openai}`
- internal Gateway Express: `provider: {type: internal, name: llm-gateway-express-local-proxy}`
- SFR gateway: `provider: {type: gateway, name: SFR-GATEWAY, ...}`

Monet × SFR gateway is omitted: Monet's native provider interface cannot honor an arbitrary
`api_base` and `auth_header`, so presenting that config as runnable would be misleading.

The SFR route reads `SFR_GATEWAY_API_KEY` from the host and sends it through
`X-Api-Key`; the secret itself is never written to generated YAML.



The older `gateway_prompt_cache_*.py` programs are standalone prompt-cache diagnostics,
not evaluation-matrix tests.