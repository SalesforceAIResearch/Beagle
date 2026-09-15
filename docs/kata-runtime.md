# Experimental offline Kata runtime

`KataDockerRuntime` runs each acquired task container through an operator-configured
Kata runtime with `--network none`. It is a starting point for prebuilt, offline
workloads using Beagle's `Runnable` lifecycle and `DockerHarness`.

This work is motivated by Jeremy Canale's independent
[ExploitGym containment research](https://jeremycanale.com/incident-openai-hugging-face/en/):
containment should be enforced by execution infrastructure, with an explicit
runtime choice and verifiable cleanup. This adapter does not establish that
Beagle has the vulnerabilities discussed in that report.

## Prerequisites

Run Beagle on a Linux host with Docker, KVM and a configured Kata/QEMU runtime.
Follow the official [Kata Docker guide](https://github.com/kata-containers/kata-containers/blob/main/docs/how-to/how-to-use-kata-with-docker.md).
The host must register the runtime name before Beagle starts. Docker Desktop's
Windows engine is not the host used by the validation below.

The tested Windows setup uses Ubuntu 24.04 under WSL2 with accessible `/dev/kvm`,
a separate Linux Docker daemon and its own socket. Nested virtualization depends
on the machine and WSL configuration; WSL2 alone does not prove Kata can start.
Validated versions: Docker Engine 29.8.0, Kata Go runtime 4.1.0, QEMU 11.0.1,
WSL host kernel 6.6.87.1 and Kata guest kernel 6.18.35. The upstream Docker guide
currently uses the deprecated Go runtime; this is an experimental compatibility
target, not a claim of support for every Kata runtime or hypervisor.

## Explicit selection

Python callers can construct the runtime directly:

```python
from beagle.rollout.runtime import KataDockerRuntime

runtime = KataDockerRuntime(
    docker_host="unix:///run/beagle-kata/docker.sock",
    runtime_name="kata",
)
handle = runtime.acquire(image="alpine:3.22", command=["sleep", "infinity"])
try:
    result = runtime.exec(handle, ["uname", "-r"])
    assert result.ok, result.stderr
finally:
    runtime.destroy(handle)
```

For canonical evaluation YAML, `run.runtime` also accepts a mapping. This is a
configuration fragment; supply a compatible offline agent and DockerHarness
benchmark separately:

```yaml
run:
  runtime:
    kind: kata
    options:
      docker_host: unix:///run/beagle-kata/docker.sock
      runtime_name: kata
```

For `RunConfig` Python input, the same mapping is the top-level `runtime` field.
`beagle.evaluate` constructs this runtime when one is not supplied. The runner
rejects an incompatible explicit runtime and harnesses that do not use this
adapter. Only `InBandGrader` (which reduces rewards without launching an evaluator)
is supported. Harbor/Pier-owned environments, SWE-bench's external evaluator and
DarwinX evolution are unsupported. Host-side agent/harness Python code remains
trusted; it must execute task code through the supplied runtime.

## Boundaries

- Network access is disabled throughout installation and execution. Dependencies
  must be baked into the image. Existing agents that clone repositories or call
  external model APIs need additional integration before they can use this profile.
  Image pulls are performed by the Docker host, outside the guest's network policy.
- There is no automatic fallback to `runc`. The daemon must advertise a Linux Kata
  registration; after launch, Docker's reported runtime and network mode are checked.
  The registration name and daemon remain trusted operator configuration: Docker
  metadata is not proof that the advertised runtime actually starts a VM.
- Only an entrypoint override is accepted in `run_args`. Writable host mounts are
  rejected. Read-only mounts still expose their content: select inputs deliberately.
  CPU/memory limits requested on a host that does not advertise support are rejected.
- No host `nsenter`/iptables egress mechanism is reused for the guest. This version
  does not implement model-provider allowlists, shared Compose environments,
  attestation, or an external hypervisor watchdog.
- Cleanup force-removes the container, then independently checks Docker's listing.
  On uncertain cleanup, `KataRuntimeError.handle` retains the ID/name for retry.
  A partial launch is tracked by its generated name. Docker absence is a point-in-time
  observation; the adapter does not reconcile late daemon operations after a timeout
  or recover handles after a Beagle process crash.

## Real-host validation

Install Beagle and its vendored xrlenv in the Linux environment, pre-pull the image,
then run from the checkout:

```sh
docker --host unix:///run/beagle-kata/docker.sock pull alpine:3.22
python tests/smoke/kata_runtime.py --docker-host unix:///run/beagle-kata/docker.sock
```

The smoke makes no model calls. It runs a probe through `DockerHarness` and the
default `Runnable` lifecycle, checks file writes/reads, a distinct guest kernel,
loopback-only interfaces, an associated host QEMU process and timeout status 124.
Both normal completion and an intentional agent exception must remove the
container and terminate its associated QEMU process. A final probe exercises
`beagle.evaluate` with runtime construction from config, `Runner`, `InBandGrader`
and creation of `run.json` in a temporary directory. Run on the Docker host with
permission to inspect those processes. This is lifecycle validation, not an
adversarial escape-resistance evaluation.
