---
orphan: true
---

# Overview

Use this section when you are writing code that needs XRLEnv-managed
containers directly.

The recommended path is the managed-container API. It gives you a
remote container session with `exec`, streaming `exec`, archive
upload/download, lifecycle cleanup, and admin-panel metadata hooks.

If your code already uses docker-py, use the Docker SDK drop-in
instead of rewriting the harness.

- {doc}`work_with_xrlenv_managed_containers/index`
