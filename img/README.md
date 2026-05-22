# Architecture diagrams

Rendered diagrams for the design docs at the repo root.

## Inference (already rendered)

Referenced by [`inference-2000-users.md`](../inference-2000-users.md):

- `inference-arch-overview.jpeg` — six-layer architecture top-down
- `inference-request-flow.jpeg` — numbered request lifecycle overlay
- `inference-pod-detail.jpeg` — single NIM pod internals
- `inference-cost-overlay.jpeg` — cost-annotated architecture

## Inference + Training (pending)

Referenced by [`inference-and-training.md`](../inference-and-training.md). Six prompts inline in that doc — render via nano-banana and drop the JPEGs here:

- `foundation.jpeg` — Figure 1: shared GCP foundation (two clusters, shared services)
- `inference-architecture.jpeg` — Figure 2: inference architecture with dense 8× H100 path (can reuse `inference-arch-overview.jpeg` or re-render)
- `training-cluster-topology.jpeg` — Figure 3: multi-node training with LWS + GPUDirect TCPXO
- `multi-nic-topology.jpeg` — Figure 4: single pod with 9 NIC interfaces
- `storage-hierarchy.jpeg` — Figure 5: 4-tier storage with workload mapping
- `promotion-lifecycle.jpeg` — Figure 6: horizontal timeline of train → eval → serve

Once rendered, update the image references in `inference-and-training.md` to point at these filenames.
