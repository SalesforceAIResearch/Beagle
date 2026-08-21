# Glossary

Short definitions for terms used in the public docs.

Admin panel
: Browser UI served by `xrlenv up`, usually at
  `http://127.0.0.1:8080/`. It shows nodes, raw rollouts, template
  rollouts, images, capacity, builds, and health.

Artifact path
: A directory written by your workflow or benchmark harness. Wrap work
  in `xrlenv.rollout_metadata(artifact_path=...)` so the admin panel
  can link the XRLEnv rollout to that directory.

Control plane
: The process started by `xrlenv up`. It accepts SDK requests, stores
  state, schedules work, serves the admin panel, and exposes metrics.

control plane
: See `Control plane`.

bootstrap
: The setup step that installs dependencies and configures
  `xrlenv-node` on a fresh VM.

Data plane
: The Docker-capable hosts running `xrlenv-node`. Nodes create,
  execute in, and destroy containers assigned by the control plane.

Docker SDK drop-in
: The `xrlenv.from_env()` compatibility layer for code that already
  uses docker-py. In cluster mode it routes docker-py operations
  through XRLEnv instead of the local Docker daemon.

Framework/harness adapter
: A small adapter that subclasses a benchmark framework's environment
  interface and replaces only the container operations. The Harbor
  adapter for terminal-bench-style workloads is the current example.

Image affinity
: Scheduler preference for nodes that already have the requested image
  cached. Affinity improves locality but does not override capacity or
  fairness rules.

Managed container
: A Docker container whose lifecycle is owned by XRLEnv. It is created
  by `Client.acquire_container(...)`, the Docker SDK drop-in, or a
  framework/harness adapter.

Node
: One running `xrlenv-node` daemon attached to the control plane.

Raw rollout
: The XRLEnv record for one managed-container lifecycle. Raw rollouts
  are shown under `/rollouts/raw` in the admin panel.

Rollout id
: XRLEnv's unique id for one unit of sandbox work. For managed
  containers it identifies the acquire-to-destroy lifecycle. For
  template rollouts it identifies one environment episode.

Scheduler
: Control-plane component that chooses a node for work based on
  capacity, image state, task fairness, and node connectivity.

sandbox
: An isolated execution environment managed by XRLEnv. In the shipped
  backend this is a Docker container.

State store
: SQLite-backed metadata store used by the control plane and admin
  panel. Large artifacts stay on disk; the store holds metadata and
  pointers.

systemd
: The Linux service manager used by the cloud bootstrap scripts to run
  `xrlenv-node` as a background service.

Task key
: Optional fairness key passed with related work. It helps the
  scheduler avoid packing too many attempts for the same logical task
  onto one node. It is not identity.

Template rollout
: Advanced SDK path driven by `Client.rollout(template=...)`. It uses
  an `EnvAdapter` inside the sandbox and produces a sealed trajectory.

Trajectory
: Ordered record of observations, actions, rewards, and terminal
  state for a template rollout. Trajectories can be replayed after the
  rollout is sealed.
