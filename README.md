# nemotron-architecture

> Architecture design documents and diagrams for deploying NVIDIA Nemotron-3 on the public cloud (GCP, AWS, and Azure).

Documentation-only repo. No code, no infrastructure. The design docs describe topology, sizing, cost models, and decision rationale. Implementation lives in separate repos per cloud.

## Documents — indexed by cloud

### Google Cloud (GKE)

| Document | Scope | Status |
|---|---|---|
| [`inference-2000-users.md`](inference-2000-users.md) | GCP inference-only design for 2,000 concurrent users on GKE. Six-layer architecture, ~$178K/month. | Stable. Diagrams in `img/`. |
| [`inference-and-training.md`](inference-and-training.md) | GCP inference + training. Mamba-2 cache math, DWS Flex-start, GPUDirect TCPXO multi-NIC topology, promotion lifecycle via Vertex AI Model Registry. | Stable. Diagrams embedded inline via GitHub user-attachments. |

### AWS (EKS)

| Document | Scope | Status |
|---|---|---|
| [`inference-and-training-aws.md`](inference-and-training-aws.md) | AWS inference + training on EKS, single region multi-AZ. Mirrors the GCP doc structure with AWS service mappings: Karpenter, EC2 Capacity Blocks for ML, EFA UltraClusters, FSx for Lustre with S3 linkage, SageMaker Model Registry, Argo Rollouts. Includes a GCP→AWS component mapping appendix. | Stable. Diagrams embedded inline via GitHub user-attachments. |

### Microsoft Azure (AKS)

| Document | Scope | Status |
|---|---|---|
| [`inference-and-training-azure.md`](inference-and-training-azure.md) | Azure inference + training on AKS, single region multi-AZ. Mirrors the GCP and AWS doc structures with Azure service mappings: ND_H100_v5, Karpenter on AKS, Azure On-Demand Capacity Reservations, NDR InfiniBand + GPUDirect RDMA, Azure Managed Lustre with Blob integration + GDS, Azure ML Model Registry, Container Apps for the Model Gateway, Argo Rollouts. Includes a three-cloud (GCP/AWS/Azure) component mapping appendix. | Draft. Diagrams pending — nano-banana prompts inline. |

### Cross-cloud

Appendix A in [`inference-and-training-azure.md`](inference-and-training-azure.md#appendix-a--three-cloud-component-mapping) is the canonical three-cloud component reference (GCP / AWS / Azure side-by-side, 40+ services). Supersedes the GCP↔AWS mapping in the AWS doc.

## Relationship to implementation repos

```
                  nemotron-architecture  (this repo — design docs)
                   ┌──────────┴──────────┐
                  GCP                    AWS
                   │                      │
        ┌──────────┴──────────┐          (no impl repo yet)
        ▼                     ▼
 nemotron-on-gke      nemotron-platform-gcp
 (exploration tool,   (production IaC,
  YAML generator UI)   Terraform + Helm + Cloud Run)
```

Decisions documented here drive what gets implemented. When a design changes, the design doc is updated first, then a follow-up PR lands the implementation. An AWS implementation repo would mirror `nemotron-platform-gcp` (Terraform modules for VPC, EKS, IAM, data-plane, observability, API Gateway, plus Helm charts and ECS Fargate gateway).

## License

Apache-2.0
