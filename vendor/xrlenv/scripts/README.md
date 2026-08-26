# `scripts/`

Developer helper scripts. **Operator / deployment scripts live under
[`deploy/`](../deploy/)** — registry operations in
[`deploy/registry/`](../deploy/registry/README.md), node provisioning in
[`deploy/node/`](../deploy/README.md).

| Script | Used when | Cross-referenced / used in |
|---|---|---|
| `gen_protos.sh` | After editing any `xrlenv/api/proto/*.proto`: regenerate the gRPC stubs into `xrlenv/api/_pb2/`. | `README.md`; `xrlenv/api/__init__.py` |
