# nemotron-architecture

> Architecture design documents and diagrams for deploying NVIDIA Nemotron-3 on Google Cloud.

Documentation-only repo. No code, no infrastructure. The design docs here describe topology, sizing, cost models, and decision rationale; the implementation lives in [nemotron-platform-gcp](https://github.com/adnanfida/nemotron-platform-gcp) (production IaC) and [nemotron-on-gke](https://github.com/adnanfida/nemotron-on-gke) (the YAML generator exploration tool).

## Documents

| Document | Scope | Status |
|---|---|---|
| [`inference-2000-users.md`](inference-2000-users.md) | Production inference deployment of Nemotron-3 Super 120B on GKE for 2,000 concurrent users. Six-layer architecture, ~$178K/month. | Stable. Rendered diagrams in `img/`. |
| [`inference-and-training.md`](inference-and-training.md) | Both inference AND training architectures on GCP. Updated sizing math accounting for Mamba-2 SSM cache reduction. Covers DWS Flex-start, GPUDirect TCPXO multi-NIC topology, training data flow, promotion lifecycle. | Draft. Diagrams pending — nano-banana prompts inline. |

## Relationship to the implementation repos

```
                   nemotron-architecture  (this repo — design docs)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      nemotron-on-gke              nemotron-platform-gcp
      (exploration tool,           (production IaC,
       YAML generator UI)           Terraform + Helm + Cloud Run)
```

Decisions documented here drive what gets implemented in the platform repo. When a design changes, the design doc is updated first, then a follow-up PR lands the implementation.

## License

Apache-2.0
