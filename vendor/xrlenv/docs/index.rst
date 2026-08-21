XRLEnv
======

XRLEnv runs containerized agent and evaluation workloads on local
machines or a small fleet of cloud VMs. Your code asks for an
isolated container, XRLEnv chooses a node, tracks lifecycle and
artifacts, and gives operators a single place to inspect what
happened.

Use it when you need Docker-backed sandboxes that are easier to run
at cluster scale than hand-rolled SSH scripts or one-off Docker
daemons. XRLEnv is not a trainer, model server, or benchmark grader:
it manages sandbox execution and leaves task logic in your code or
the upstream benchmark harness.

Pick Your Path
--------------

.. container:: landing-grid

   .. container:: landing-card

      **Try one local container**

      Start with :doc:`getting_started/quickstart` to boot ``xrlenv
      up``, acquire one container, run ``exec``, and destroy it
      cleanly.

   .. container:: landing-card

      **Run existing Docker SDK code**

      If your harness already calls ``docker.from_env()``, use the
      :doc:`Docker SDK drop-in
      <build_with_xrlenv/work_with_xrlenv_managed_containers/docker_py_dropin>`.

   .. container:: landing-card

      **Use a framework/harness adapter**

      For Harbor-format workloads (terminal-bench-2, seta-env, …), use
      the :doc:`framework adapter
      <supported_benchmarks_and_harnesses/harbor_framework>` and keep
      the benchmark's own driver in charge.

   .. container:: landing-card

      **Build a custom workflow**

      For new async Python code, use
      :doc:`Client.acquire_container
      <build_with_xrlenv/work_with_xrlenv_managed_containers/direct_api>`
      directly.

   .. container:: landing-card

      **Operate a cluster**

      Read :doc:`deploy/index` for deployment shapes and
      :doc:`observability/index` for the admin panel, metrics, logs,
      and image capacity views.

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Getting started

   getting_started/installation
   getting_started/quickstart
   getting_started/architecture

.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: Deploy

   deploy/single_node_deployment
   deploy/multi_node_deployment/index
   deploy/multi_tenancy

.. toctree::
   :maxdepth: 1
   :numbered:
   :caption: benchmarks & harnesses

   supported_benchmarks_and_harnesses/index
   supported_benchmarks_and_harnesses/swe_bench
   supported_benchmarks_and_harnesses/harbor_framework
   supported_benchmarks_and_harnesses/pier_framework
   supported_benchmarks_and_harnesses/deep_swe
   supported_benchmarks_and_harnesses/frontier_swe
   supported_benchmarks_and_harnesses/webarena_infinity
   supported_benchmarks_and_harnesses/evoclaw
   supported_benchmarks_and_harnesses/writing_your_own_adapter

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Build with XRLEnv

   build_with_xrlenv/index
   build_with_xrlenv/work_with_xrlenv_managed_containers/index



.. toctree::
   :maxdepth: 1
   :numbered:
   :caption: Observability

   observability/admin_panel
   observability/admin_auth
   observability/metrics
   observability/logs
   observability/capacity
   observability/tracing

.. toctree::
   :maxdepth: 4
   :numbered:
   :caption: Technical details

   technical_details/scheduling
   technical_details/resource_isolation
   technical_details/images/index


.. toctree::
   :maxdepth: 1
   :numbered:
   :caption: Developer guide

   developer_guide/api_reference
   developer_guide/cli_reference
   developer_guide/run_config
   developer_guide/security
   developer_guide/tokens
   developer_guide/timeouts

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Reference

   reference/configuration
   reference/glossary
   reference/cheatsheets/index
