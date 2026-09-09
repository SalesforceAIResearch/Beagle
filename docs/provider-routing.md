# Provider routing

`provider` tells an agent *how* to reach its model. It is a **discriminated union** in the
agent config block. `type` is always required; `name` is required for `gateway` and
`internal` and optional only for `direct`; `extra_args` is type-specific.

`forward_env` is a **separate**, independent sibling that generically forwards host
environment variables into the task container (API keys, tokens). The two serve different
concerns and compose freely.

---

## The three provider types

### `direct` — first-party model API (default)

The agent calls the named model provider directly. An explicit `name` makes routing and
network allowlisting unambiguous. `direct` is the default when `provider` is omitted
entirely; in that fallback form, LiteLLM infers the provider from the model name.

```yaml
agent:
  model:
    name: gpt-5.5
  provider: {type: direct, name: openai}
  forward_env: [OPENAI_API_KEY]     # first-party key forwarded into the container
```

The name pins a LiteLLM provider prefix (e.g. `openai`, `anthropic`), prepended to the
model name as `<name>/<model>`. If `name` is omitted, a recognized bare model name or
embedded LiteLLM prefix is used as a convenience fallback:

```yaml
  provider: {type: direct}
```

---

### `gateway` — OpenAI-compatible proxy (org-level gateway)

Use when your organisation fronts model providers with an internal OpenAI-compatible
proxy. The agent routes every model call through your endpoint — any model your proxy
serves works unchanged (the gateway routes by model name). An adapter whose native provider
does not accept an explicit endpoint rejects this type during preflight.

```yaml
agent:
  model:
    name: gpt-5.5                   # whatever model ID YOUR gateway routes
  provider:
    type: gateway
    name: my-org-gateway            # required gateway identifier
    extra_args:
      api_base: https://gateway.example.com/openai/v1
      api_key_env: MY_ORG_API_KEY   # NAME of the env var holding the key (not the key)
      auth_header: X-Api-Key        # optional; see below
  # forward_env is NOT needed for the gateway key — api_key_env already carries it.
  # Add forward_env only for OTHER credentials the agent needs inside the container.
```

**`extra_args` for `gateway`:**

| Field | Required | Description |
|---|---|---|
| `api_base` | yes | Endpoint prefix. beagle appends `/responses` or `/chat/completions` depending on the agent and `effort`. |
| `api_key_env` | no | **Name** of the environment variable holding the API key (set in `.env`, never in the config file). Omit when the gateway needs no key; then beagle uses a placeholder. |
| `auth_header` | no | Custom auth header (e.g. `X-Api-Key`). When set, the key is sent both as `Authorization: Bearer` and in this header — gateways accepting either work. |

The secret never enters the config file. `api_key_env` is the *variable name*; the value
lives in `.env` (which beagle loads at CLI start). The config stays committable.

Validate the wiring before spending:

```bash
beagle evaluate --config your-config.yaml --dry-run
```

---

### `internal` — named deployment provider

Use when a named provider (e.g. a team-internal deployment) is already registered in your
infrastructure. The agent is told which named provider to use; the actual endpoint is
resolved from the environment at run time.

```yaml
agent:
  model:
    name: gpt-5.5
  provider:
    type: internal
    name: my-deployment-provider    # required for internal; must not be empty
```

---

## `forward_env` — generic host-to-container forwarding

`forward_env` is a flat list of environment variable names (or rename pairs) that beagle
copies from the host into the task container before the agent step. It is independent of
the selected `provider` type — use it for API keys on the `direct` path, or for any other
credentials the agent needs (e.g. a search API key, a private registry token).

```yaml
forward_env: [OPENAI_API_KEY]                  # bare string: same name host→container
forward_env: [OPENAI_API_KEY, ANTHROPIC_API_KEY]

# Rename: container name on the left, host name on the right
forward_env:
  - [CONTAINER_VAR, HOST_VAR]                  # 2-element list
  - {CONTAINER_VAR: HOST_VAR}                  # mapping form (also accepted)
```

A missing host variable is silently skipped (the variable is not injected). `--dry-run`
reports which forwarded variables are unset, so you surface missing credentials before
spending.

---

## Agent support matrix

| Agent | `direct` | `gateway` | `internal` |
|---|:---:|:---:|:---:|
| `mini-swe` | ✓ | ✓ | ✓ |
| `opencode` | ✓ | ✓ | ✓ |
Adapters with an agent-native provider mechanism may support only `direct` and `internal`;
unsupported route types fail during preflight.

---

## Full runnable examples

### Direct access — OpenAI model, first-party key

```yaml
run:
  dir: ./results
  name: eval-direct
  runtime: xrlenv-cluster
  parallelism: 8

agent:
  harness:
    name: mini-swe
    version: v2.4.6
    source:
      repo: https://github.com/<your-org>/mini_swe_agent_v2.4.6
      ref: <baseline-commit-sha>
      token_env: GH_TOKEN
  model:
    name: gpt-5.5
  provider: {type: direct, name: openai}
  forward_env: [OPENAI_API_KEY]
  effort: high
  max_turns: 150

data:
  - benchmark: swe-rebench
```

### Org gateway — OpenAI-compatible proxy, custom header

```yaml
run:
  dir: ./results
  name: eval-gateway
  runtime: xrlenv-cluster
  parallelism: 8

agent:
  harness:
    name: mini-swe
    version: v2.4.6
    source:
      repo: https://github.com/<your-org>/mini_swe_agent_v2.4.6
      ref: <baseline-commit-sha>
      token_env: GH_TOKEN
  model:
    name: gpt-5.5          # whatever model your gateway routes
  provider:
    type: gateway
    name: my-org-gateway
    extra_args:
      api_base: https://gateway.example.com/openai/v1
      api_key_env: MY_ORG_API_KEY   # set MY_ORG_API_KEY in .env
      auth_header: X-Api-Key        # omit if your gateway uses only Authorization: Bearer
  effort: high
  max_turns: 150

data:
  - benchmark: swe-rebench
```

### Internal named provider

```yaml
run:
  dir: ./results
  name: eval-internal
  runtime: xrlenv-cluster
  parallelism: 4

agent:
  harness:
    name: mini-swe
    version: v1.0.0
    source:
      repo: https://github.com/<your-org>/mini_swe_agent_v1.0.0
      ref: <baseline-commit-sha>
      token_env: GH_TOKEN
  model:
    name: claude-opus-4
  provider:
    type: internal
    name: my-deployment-provider
  effort: high
  max_turns: 100

data:
  - benchmark: swe-rebench
```

---

## Migration note

The old scalar `provider:` and top-level `gateway:` shapes are no longer accepted. beagle
raises a descriptive error on startup if either is present:

| Old shape | Replacement |
|---|---|
| `provider: my-provider` | `provider: {type: internal, name: my-provider}` |
| `gateway: {api_base: …}` | `provider: {type: gateway, name: my-gateway, extra_args: {api_base: …, api_key_env: …}}` |
