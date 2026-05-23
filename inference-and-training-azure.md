# Architecture: Nemotron-3 Super 120B on AKS — Inference and Training

**Status:** Design proposal
**Last updated:** 2026-05-22
**Audience:** Platform engineering, ML platform, FinOps
**Region target:** single region, multi-AZ (e.g., East US 2 with Zones 1, 2, 3)

This document mirrors [`inference-and-training.md`](inference-and-training.md) (the GCP version) and [`inference-and-training-aws.md`](inference-and-training-aws.md) (the AWS version) for Microsoft Azure / AKS. Same two workloads, same model, same target scale (2,000 concurrent users for inference; 32–128 GPU training jobs), same architectural principles. Azure-specific service mappings throughout. A side-by-side comparison appendix maps GCP / AWS / Azure components.

The two workloads share an Azure subscription, Shared VNet, IAM model (AKS Workload Identity via Microsoft Entra ID), observability stack, and Blob-based model artifact plane. They run on **separate AKS clusters** because their node SKUs, schedulers, networking, and lifecycles are fundamentally different.

The seam between training and inference is the **Azure ML Model Registry**: training writes versioned checkpoints to Azure Blob Storage and registers them; inference consumes approved checkpoints via a promotion pipeline (Argo Rollouts).

---

## 1. Why Nemotron-3 changes the sizing math

This is the same on Azure as GCP and AWS — the model architecture is cloud-agnostic. Key consequences:

- **Mamba-2 SSM cache is fixed-size**, not context-length-dependent. Per-replica concurrent capacity is enormous compared to a dense 120B Transformer.
- **MoE with 12B active params**: all 120B must be in VRAM, but effective compute is ~12B-dense per token.
- **Native NVFP4 (4-bit)**: weights ~60GB instead of ~240GB at BF16.

**Concrete consequence for Azure sizing:** at 200 concurrent streams × 3,000-token context in FP8 KV, total runtime cache is ~15 GB. A single `Standard_ND96isr_H100_v5` node (8× H100 80GB, 640 GB total VRAM) holds 120 GB of weights distributed across 8 GPUs and has ~440 GB of cache headroom. **One node handles 1000+ concurrent streams** before hitting memory; compute is the bottleneck long before VRAM.

This makes the deployment dramatically smaller than equivalent 120B-Transformer architectures.

### Sizing math (for engineers verifying)

Static weight memory ($M_{weights}$): $N$ parameters × bytes per parameter

| Precision | Bytes/param | Weight VRAM (120B) | Per-GPU at TP=8 |
|---|---|---|---|
| BF16 | 2 | 240 GB | 30 GB |
| FP8 | 1 | 120 GB | 15 GB |
| NVFP4 | 0.5 | 60 GB | 7.5 GB |

KV cache for the attention layers (Grouped Query Attention):
$M_{KV} = 2 × B × C × L_{attn} × H_{kv} × D_{head} × P$
where $B$ = batch, $C$ = context length, $L_{attn}$ = number of attention layers (~16), $H_{kv}$ = KV heads (2), $D_{head}$ = head dim (128), $P$ = precision bytes.

At $B$=200, $C$=3000, FP8 KV: $M_{KV} ≈ 9.83$ GB.

SSM cache for Mamba-2 layers is fixed size per batch, independent of context length.

Cumulative runtime cache (KV + SSM + framework overhead) at 200 concurrent ≈ 15 GB, vs ~440 GB per-node headroom after weights → ~30× headroom. Compute, not memory, bounds concurrency on Nemotron-3.

---

## 2. Shared Azure foundation

| Component | Purpose | Shared by |
|---|---|---|
| **Azure Subscription** `nemotron-platform` (or management-group child) | One billing/audit boundary | Both |
| **Shared VNet** (East US 2, 3 AZs) | One network, Azure NAT Gateway for egress | Both |
| **Private Endpoints + Private Link** | Private access to Blob, ACR, Key Vault, etc. without traversing the internet | Both |
| **Azure Key Vault** | NGC API key, HF token (secrets); CMK for Blob + Cosmos DB | Both |
| **Azure Key Vault Managed HSM** | FIPS 140-2 Level 3 keys for high-compliance scenarios | Optional |
| **Azure Blob Storage `nemotron-weights`** (RA-GRS or RA-GZRS, CMK-encrypted) | Source of truth for model weights. Training writes; inference reads | Both (write/read separated by role assignments) |
| **Azure Container Registry (ACR)** | NIM containers + training framework images | Both |
| **Azure ML Model Registry** | Versioned checkpoints with eval metadata — the promotion gate | Both |
| **Azure Monitor Managed Service for Prometheus (AMP)** | Metrics federation across both clusters | Both |
| **Azure Managed Grafana (AMG)** | Dashboards | Both |
| **Azure Monitor Log Analytics Workspace** | Container logs + diagnostic logs; routed to Azure Data Explorer for analytics | Both |
| **Application Insights** | Distributed tracing | Both |

### [Figure 1] Shared Azure foundation

```
Create a clean technical cloud architecture diagram in flat-design vector
illustration style, 16:9 landscape, soft cream/beige background (#F5F0E8)
with subtle grid pattern. Title at top: "Nemotron-3 on Azure — Shared
Platform Foundation".

A single large rounded rectangle in the center labeled "Azure Subscription:
nemotron-platform (East US 2)" contains the entire scene. Inside that, a
slightly smaller rounded rectangle labeled "Shared VNet — 3 Availability
Zones (Zone 1, Zone 2, Zone 3)".

Inside the VNet, two side-by-side Kubernetes cluster blocks in K8s blue
(#326CE5):
  - LEFT: "AKS Inference Cluster" with badge "always-on 24/7"
  - RIGHT: "AKS Training Cluster" with badge "on-demand bursts"

Below the VNet, three horizontal "shared services" strips in Azure blue
(#0078D4):

Strip 1 — Artifacts:
  - Blob icon "Azure Blob Storage (nemotron-weights, RA-GRS, CMK)"
  - Container icon "Azure Container Registry (ACR)"
  - Database icon "Azure ML Model Registry"

Strip 2 — Identity & Secrets:
  - Chain-link icon "AKS Workload Identity (Microsoft Entra)"
  - Key icon "Azure Key Vault"
  - Lock icon "Key Vault Managed HSM (CMK)"

Strip 3 — Observability:
  - Flame icon "Azure Monitor Managed Prometheus"
  - Chart icon "Azure Managed Grafana"
  - Stacked-logs icon "Log Analytics → Azure Data Explorer"

Dashed arrows from both cluster blocks down to each strip with small
labels: "writes / reads" on artifacts, "binds" on identity, "emits" on
observability. Subtle corner annotation: "Training writes weights;
Inference reads them. Azure ML Model Registry is the promotion gate."

Visual rules: flat 2D, soft drop shadows, rounded corners, Azure blue
(#0078D4) for Azure service icons, K8s blue (#326CE5) for cluster
boundaries, NVIDIA green (#76B900) only on GPU chips (none in this
figure). Render each label exactly once.
```

---

## 3. Inference architecture

**Sized for:** 2,000 active users → ~200 concurrent streams at 10% peak concurrency, TTFT < 500 ms, ITL < 25 ms, > 99.9% monthly availability.

### Cluster shape

**Node SKU choice — recommended path is dense:**

| Path | Per-replica | Replicas for 200 concurrent | Total GPUs | Notes |
|---|---|---|---|---|
| **Dense** (recommended) | `Standard_ND96isr_H100_v5`, 8× H100 80GB, TP=8 | 2–3 (HA across AZs) | 16–24 | Single-node NVLink. Mamba math = each replica handles 1000+ concurrent. |
| **Smaller GPUs** | Not viable on Azure for 120B | — | — | Azure doesn't offer a 2-GPU H100 SKU. NV-series (A10) is too weak for 120B. |

**Stick with the dense path on Azure** — same logic as the AWS doc. The H100 SKU is the standard pick for 120B inference; Mamba's per-replica capacity makes 2–3 replicas across AZs sufficient.

**`Standard_ND96isr_H100_v5` full specs (for reference):**

| Component | Spec |
|---|---|
| GPUs | 8× NVIDIA H100 80GB SXM5 |
| GPU interconnect | NVLink + 4th-gen NVSwitch, 900 GB/s bidirectional per GPU |
| vCPUs | 96 (Intel Xeon Platinum) |
| System memory | 1,900 GiB |
| Local NVMe SSD | 3,800 GiB |
| InfiniBand | 8× NDR (400 Gbps each) |
| InfiniBand aggregate | 3,200 Gbps (3.2 Tbps) |

### Edge → routing → inference

```
Client → Azure DNS → Azure Front Door (CDN + WAF + DDoS) → Application Gateway →
Azure API Management → Model Gateway (Container Apps × 3) → AKS Service (App
Gateway backend pool) → NIM pods (8× H100)
```

**Each layer's job:**

- **Azure DNS + Azure Front Door + DDoS Protection** — DNS, global CDN/edge with TLS, WAF rules (OWASP top-10), L7 DDoS protection. Front Door supports HTTP/2 + HTTP/3.
- **Azure Application Gateway** — regional L7 LB. Terminates TLS from Front Door, routes by hostname/path. Integrates with AKS via the **Application Gateway for Containers** (the modern AGIC successor), using pod-IP backend pools (analog to GCP NEG / AWS ALB target type IP).
- **Azure API Management (APIM)** — per-tenant subscription keys, products + tiers, throttle policies, request logging to Log Analytics. *Caveat: APIM has request timeout limits; for SSE streaming, configure long-running policies or bypass APIM for the streaming path and use it for the metadata/admin endpoints.*
- **Microsoft Entra ID + App Gateway OIDC** — alternate auth path for human-user OAuth/SSO on internal tools. **AWS Verified Access** equivalent for zero-trust application access.
- **Model Gateway on Azure Container Apps (3 always-on replicas)** — SSE streaming proxy, session-affinity routing via Azure Cache for Redis, prompt shaping, token-count cost emit to Event Hubs → Azure Data Explorer, cancellation propagation. **Container Apps is the right runtime** — managed serverless (built on KEDA, Envoy, Dapr), TLS termination, scale-to-zero, supports SSE. Not Azure Functions (15-min timeout), not AKS-hosted (couples blast radius).
- **AKS Service exposed via Application Gateway for Containers** — gateway targets pod IPs directly
- **NIM pods** — `nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b`, TP=8 on a single `Standard_ND96isr_H100_v5` node

### Inference workload spec

| Setting | Value | Why |
|---|---|---|
| Deployment replicas (baseline) | 2 (one per AZ across 2 of 3) | Bare minimum HA |
| Deployment replicas (with HPA headroom) | 3–4 | N+1 redundancy across all 3 AZs |
| HPA metric | `vllm:num_requests_waiting` queue depth via AMP + KEDA | CPU-based HPA never triggers on GPU pods |
| Target | queue depth < 4 per replica | |
| PDB | `minAvailable: 2` | Tolerates rolling updates |
| Topology spread | 3-AZ, maxSkew=1 via `topologySpreadConstraints` | At least one replica per AZ |
| Weights mount | **Azure Blob CSI driver (BlobFuse2) + local NVMe SSD cache** (3.8 TB local SSD per ND_v5) | Warm replicas don't re-read from Blob on restart |
| Probes | startupProbe 30-min grace, readiness, liveness | Weight load takes 5–15 min on first boot |
| Service | ClusterIP + Application Gateway for Containers backend reference | Container-native LB targeting |
| Session affinity | App Gateway cookie OR header-based via gateway | KV-cache prefix reuse on attention layers |
| Pod security context | `capabilities: add: ["IPC_LOCK"]` | Required to lock memory pages for InfiniBand RDMA (only matters if multi-node, but harmless on single-node) |

### Inference cluster topology details

- **Cluster:** AKS with private API server endpoint (or public + authorized IP ranges), 3 AZs, Azure CNI Overlay with Cilium dataplane, KMS envelope encryption for etcd via Key Vault
- **Node management:** Karpenter on AKS (in preview/GA — verify current status; alternate is Cluster Autoscaler)
- **Node groups:**
  - `system-pool`: `Standard_D4s_v5` × 3 across AZs for system pods (CoreDNS, App Gateway for Containers controller, AMP collector, Karpenter itself)
  - `gpu-pool`: `Standard_ND96isr_H100_v5`, autoscale 0→4 — 0 baseline so idle dev clusters don't bill
- **CNI:** Azure CNI Overlay (each pod gets a routable IP via overlay, doesn't consume VNet IPs)
- **Networking:** Private subnets for nodes, NAT Gateway for egress (image pulls before ACR cache), Private Endpoints for Blob / ACR / Key Vault to avoid public traversal

### State plane (production app concerns)

| Component | Purpose | Sizing |
|---|---|---|
| **Azure Cache for Redis (Premium)** | Session-affinity hints, per-tenant rate counters, idempotency keys | P1 (6 GB), zone-redundant |
| **Azure Cosmos DB (NoSQL API)** | Durable conversation history per tenant | Serverless or provisioned with autoscale |
| **Event Hubs → Azure Data Explorer (ADX)** | Request logs + token usage for cost analytics | ADX = AWS Athena / GCP BigQuery equivalent |
| **ACR (Premium tier)** | Intra-region cached copy of NIM container | Critical for fast pod startup on scale-up |

### Observability

- **Azure Monitor Managed Service for Prometheus (AMP)** scrapes `vllm:*` and DCGM-exporter (GPU util, HBM, NVLink throughput, temperature). Pods use the **Azure Monitor agent** or **Prometheus Operator with remote_write** to ship metrics.
- **Azure Managed Grafana (AMG)** — queue depth, GPU util, KV-cache fill, p50/p95/p99 first-token and inter-token latency, cost per 1K tokens (joined with ADX). Optional CloudWatch-equivalent: Azure Monitor dashboards.
- **Application Insights** — request traces from gateway through pod with token counts in span attributes.
- **Container Insights** — node and pod metrics, complement to AMP.
- **SLOs:** p95 first token < 500 ms, p99 inter-token < 25 ms, monthly availability > 99.9%.
- **Alerts:** queue-depth-high via AMP alerting → Action Group (PagerDuty / email / Teams); pod restart rate; GPU OOM; daily cost overrun via Azure Cost Management Budgets.

### [Figure 2] Inference architecture on Azure

```
Create a clean technical cloud architecture diagram in flat-design vector
illustration style, 16:9 landscape, soft cream background (#F5F0E8) with
subtle grid. Title at top: "Nemotron-3 Inference — 2,000 Users on AKS".

Five horizontal layers stacked top to bottom with solid arrows between
layers (request flow) and dashed arrows to state stores on the side.

Layer 1 (edge / security, green tones):
Laptop icon "Client" → globe "Azure DNS" → shield "Azure Front Door
(CDN + WAF + DDoS Protection)" → LB icon "Application Gateway"

Layer 2 (API management, purple):
Hexagonal block "Azure API Management — subscription keys, products,
throttle policies"

Layer 3 (application gateway, Azure blue #0078D4):
Three Container Apps task tiles side by side labeled "Model Gateway × 3
(Azure Container Apps) — SSE proxy, session-affinity routing, cost tracking"

Layer 4 (AKS inference plane, large K8s blue rectangle taking 50% of
canvas):
Top of rectangle: "AKS Cluster — East US 2 (3 AZs, private nodes,
Karpenter-managed)"
Inside: three vertical AZ columns. In each column, ONE LARGE pod card
labeled "NIM Pod" containing 8 small NVIDIA-green (#76B900) H100 chips
in a 2x4 grid with label "8× H100 TP=8 (NVLink)". Small badge above
the row: "Nemotron-3 Super 120B-A12B (NVFP4) · 2 baseline → 4 HPA".
Left side of cluster: BlobFuse2 CSI icon with sidecar label "+ 3.8 TB
instance NVMe cache". Right side: chain icon labeled "AKS Workload
Identity → Entra App: nemotron-inference". Bottom: small gauge "HPA:
vllm queue depth via AMP" and circle "PDB minAvail=2".

Layer 5 (state + storage, Azure blue row):
Six Azure service icons: Key Vault (key), Blob Storage (weights),
ACR (NIM cache), Azure Cache for Redis (sessions), Cosmos DB (chat
history), Azure Data Explorer (logs + analytics).

Right-side observability rail (vertical, dashed connections to gateway
and AKS): Azure Monitor MSP → Azure Managed Grafana → Log Analytics →
Application Insights.

Visual rules: K8s blue (#326CE5) for cluster, Azure blue (#0078D4) for
state and Azure services, purple for API Management, Azure blue for
Container Apps gateway, NVIDIA green only on GPU chips. Solid bold
arrows for request flow (top to bottom); dashed arrows for state and
observability. Render each label exactly once.
```

---

## 4. Training architecture

### Workload types and recommended patterns

| Workload | Compute | Duration | Recommended pattern |
|---|---|---|---|
| LoRA / QLoRA / adapter fine-tuning | 1× `Standard_ND96isr_H100_v5` (8 H100) | Hours | Single-node K8s `Job`, NeMo PEFT or HF TRL |
| Full SFT | 4× `Standard_ND96isr_H100_v5` (32 H100) | 1–3 days | Multi-node `LeaderWorkerSet` + Kueue, NeMo |
| DPO / RLHF alignment | 4–16× nodes (32–128 H100) | 3–7 days | Multi-node LWS + Kueue, NeMo-Aligner |
| Continued pre-training | 16–64× nodes (128–512 H100) | Weeks | Multi-node LWS + Kueue + Capacity Reservations |
| Full pre-training from scratch | 1000+ H100 | Months | **Consider Azure Machine Learning managed compute** |

### Training memory math

Identical to GCP and AWS docs (cloud-agnostic). 1.92 TB static state for full-parameter BF16 training. 4× nodes × 8 H100 = 32× H100 = 2,560 GB cluster VRAM → ~60 GB per GPU after FSDP/ZeRO-3 sharding → ~20 GB headroom for activations.

### Cluster topology

- **Cluster:** AKS, private nodes, AKS Workload Identity, separate from inference cluster
- **Node management:** Karpenter on AKS (or Cluster Autoscaler)
- **Node groups:**
  - `system-pool`: `Standard_E4s_v5` × 3 for Kueue controller, JobSet API, custom metrics adapter
  - `gpu-pool`: `Standard_ND96isr_H100_v5` with InfiniBand enabled, **Azure On-Demand Capacity Reservation** OR on-demand. Karpenter scales 0 → 16 nodes
  - Optional `gpu-pool-spot`: same SKU with Spot priority, for burst with aggressive checkpointing
- **Placement:** **Azure Proximity Placement Group (PPG) with Compact policy** on the GPU pool — all nodes for one job on the same InfiniBand leaf switch, required for full 3.2 Tbps inter-node bandwidth
- **InfiniBand driver:** installed via DaemonSet (NVIDIA GPU Operator with InfiniBand support, or Mellanox OFED)

### Orchestration stack

| Layer | Tool | Purpose |
|---|---|---|
| **Azure GPU reservation** | **Azure On-Demand Capacity Reservation** | Reserve ND_v5 capacity for scheduled training runs. Direct equivalent of GCP DWS Flex-start and AWS Capacity Blocks for ML. |
| **K8s job queueing** | **Kueue** | Same as GCP/AWS — fair sharing across teams, priority preemption, gang scheduling |
| **Multi-job topology** | **JobSet API** (sig-apps) | Same as GCP/AWS |
| **Multi-node single-replica** | **LeaderWorkerSet (LWS)** | Same as GCP/AWS |
| **Single-node single-replica** | K8s `Job` | LoRA, eval, data prep |
| **Alternative: managed training** | **Azure Machine Learning** | Managed alternative to running training on AKS. Handles cluster reconfig, fault detection, abstracts InfiniBand drivers. Evaluate before going DIY for long-running pre-training. |

### Framework choice — NeMo for Nemotron specifically

Same recommendation as GCP and AWS docs: **NVIDIA NeMo Framework + NeMo-Aligner**. First-party support for Nemotron's MoE + Mamba-2 hybrid. HuggingFace + DeepSpeed/FSDP works for LoRA on attention layers but doesn't handle Mamba layers cleanly for full-parameter training.

### Networking — NDR InfiniBand + GPUDirect RDMA

This is where Azure differs from both GCP (TCPXO over standard VPC) and AWS (EFA with custom SRD transport). **Azure uses real, bare-metal NDR InfiniBand** — the same fabric used in NVIDIA reference designs.

| SKU | InfiniBand | Bandwidth | Use |
|---|---|---|---|
| `Standard_ND96isr_H100_v5` | **8× NDR InfiniBand HCAs** | **3,200 Gbps** aggregate | Standard for 120B-class training |
| `Standard_ND96isr_H200_v5` (when GA in region) | Same 8× NDR | Same | H200 141GB variant |
| `Standard_NC40ads_H100_v5` | Limited | — | Smaller H100 SKU, not for multi-node |

**GPUDirect RDMA over InfiniBand**: each H100 GPU is mapped 1:1 to its own NDR HCA. NCCL writes packets directly from GPU HBM to the HCA, bypassing host CPU + RAM. Configure via:
- `NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,...,mlx5_ib7` (one HCA per GPU)
- `NCCL_SOCKET_IFNAME=eth0` (control plane only)
- `NCCL_IB_GID_INDEX=3` (typical for Azure NDR)

**Pod-to-NIC binding pattern:**
- AKS uses **Multus CNI** with the AKS InfiniBand device plugin to expose the 8 IB HCAs as `rdma/hca` extended resources
- Pods request via `resources.limits.rdma/hca: "8"` plus `securityContext.capabilities.add: ["IPC_LOCK"]`
- For most one-pod-per-node training jobs, the standard pattern is one pod claiming all 8 HCAs and all 8 GPUs

**No exotic VPC architecture required** — InfiniBand traffic flows over the Mellanox fabric, completely separate from the Azure VNet (which carries only management traffic). Operationally simpler than GCP's Multus + 8 secondary VPCs.

### [Figure 3] Training cluster topology on Azure

```
Create a flat-design vector technical diagram, 16:9 landscape, soft cream
background (#F5F0E8) with subtle grid. Title at top: "Nemotron-3 Training
Cluster — multi-node distributed training on AKS".

A single large rounded rectangle in K8s blue (#326CE5) labeled at top
"AKS Training Cluster — East US 2, single Proximity Placement Group,
private nodes".

Inside, two horizontal node-pool sections stacked:

Top section — "system-pool (Standard_E4s_v5 × 3, Karpenter-provisioned)"
with three small pills:
  - "Kueue (queue + gang scheduling)"
  - "JobSet API + LeaderWorkerSet"
  - "Custom metrics adapter (HPA on AMP metrics)"

Bottom section — "gpu-pool (Standard_ND96isr_H100_v5, Karpenter 0 → 16
nodes via Azure On-Demand Capacity Reservations)" takes most of canvas.
FOUR node boxes side by side, each containing 8 small NVIDIA-green
(#76B900) H100 chips in a 2x4 grid, labeled "Node 1: 8× H100", etc.
THICK neon-cyan/blue double-arrow bars between adjacent nodes labeled
"NDR InfiniBand  3,200 Gbps (8× NDR HCAs per node)" — these are the
visual centerpiece, glowing softly. A small badge above the interconnect
bars: "Proximity Placement Group — same InfiniBand leaf switch".

Overlaid on the gpu-pool, semi-transparent rectangles representing one
active training job (LeaderWorkerSet pattern):
  - "leader pod (rank 0)" overlapping leftmost GPU on Node 1
  - "worker pods (ranks 1-7, 8-15, 16-23, 24-31)" spanning across nodes
  - Badge above: "NeMo + Megatron, FSDP / ZeRO-3, TP=8 PP=4"

Outside the cluster on the right edge, Azure-blue icons connected by
dashed arrows to the leader pod:
  - Blob bucket "training dataset (streaming via BlobFuse2)"
  - Cylinder "Azure Managed Lustre (Hot Checkpoints, with Blob integration)"
  - Blob bucket "checkpoint cold archive"

Below the cluster: Azure ML Model Registry icon with dashed arrow from
checkpoint archive labeled "promotion path → inference cluster".

Right edge observability fan-out: dashed arrows from leader to small
icons labeled "Azure Monitor MSP + TensorBoard" and "Weights & Biases /
Azure ML Experiments".

Visual rules: K8s blue for cluster, NVIDIA green ONLY on GPU chips,
Azure blue (#0078D4) for storage and Azure services, InfiniBand bars in
soft glowing cyan/blue (matching Azure brand). Compact placement implied
by the dense node arrangement. Render each label exactly once.
```

### [Figure 4] InfiniBand networking topology detail

```
Create a flat-design vector technical diagram, 16:9 landscape, soft cream
background (#F5F0E8) with subtle grid. Title at top: "Azure NDR InfiniBand
Topology — Standard_ND96isr_H100_v5".

Center the diagram on a SINGLE large rounded rectangle labeled "AKS Pod:
PyTorch Training Worker (rank N) on Standard_ND96isr_H100_v5". Inside the
pod box, on the left, stack 9 horizontal labeled lines representing
network interfaces:

  eth0   "Primary VNet ENI → Cluster control, ACR pull, Blob endpoint,
          NAT for egress"
  ib0    "NDR HCA 0 (400 Gbps) → GPU 0"
  ib1    "NDR HCA 1 (400 Gbps) → GPU 1"
  ib2    "NDR HCA 2 (400 Gbps) → GPU 2"
  ib3    "NDR HCA 3 (400 Gbps) → GPU 3"
  ib4    "NDR HCA 4 (400 Gbps) → GPU 4"
  ib5    "NDR HCA 5 (400 Gbps) → GPU 5"
  ib6    "NDR HCA 6 (400 Gbps) → GPU 6"
  ib7    "NDR HCA 7 (400 Gbps) → GPU 7"

On the right side of the pod, 8 NVIDIA-green (#76B900) H100 GPU chips
in a 2x4 grid. Each GPU has a thick green arrow connecting to its
matching IB HCA on the left labeled "GPUDirect RDMA over InfiniBand
(kernel-bypass)". Between the GPUs, a thin label "NVLink + 4th-gen
NVSwitch 900 GB/s intra-node mesh".

Above the pod, a small "Multus CNI + AKS InfiniBand Device Plugin
(rdma/hca: 8)" badge.

Below the pod, dashed thick arrows fanning out to peer training pods on
other nodes, all converging through a faint cloud labeled "NDR
InfiniBand Leaf Switch — Proximity Placement Group, 3.2 Tbps per node".

A small annotation panel on the side:
  "Azure uses real Mellanox NDR InfiniBand hardware, not RoCE or TCP.
  NCCL_IB_HCA=mlx5_ib0..7 maps GPUs 1:1 to HCAs. Traffic flows over the
  IB fabric separately from the VNet — simpler architecture than GCP's
  Multus + 8 VPC pattern."

Visual rules: K8s blue for pod boundary, NVIDIA green for GPU chips and
the 8 GPUDirect arrows, Azure blue (#0078D4) for the management ENI
and the IB fabric callouts, neutral gray for the InfiniBand HCAs.
Render each label once. Sans-serif, readable.
```

---

## 5. Storage hierarchy

Different model than GCP and AWS but similar tiering. **Killer feature: Azure Managed Lustre + GPUDirect Storage over NDR InfiniBand** (announced in 2025) — direct path from H100 HBM to Lustre NVMe controllers over IB, bypassing host CPU and RAM.

| Tier | Service | Throughput | Cost | Used for |
|---|---|---|---|---|
| **Cold archive** | Azure Blob Storage (Hot or Cool tier) | Moderate | Cheapest | Source datasets, checkpoint archive, model weights distribution |
| **Hot read cache** | **BlobFuse2 CSI driver + local NVMe SSD** (3.8 TB per ND_v5) | High | Near-free | Inference weight cache, training dataset shards (read-mostly) |
| **Hot read/write** | **Azure Managed Lustre + GDS over NDR IB, with Blob integration** | Up to ~150+ GB/s aggregate | Medium-high | Multi-node training checkpoint writes; dataset hot tier |
| **Premium shared** | Azure NetApp Files (Premium / Ultra) | Very high | Expensive | If Managed Lustre is unavailable or you need NFS semantics across jobs |

### The killer features layered together

**1. Azure Managed Lustre + Blob integration** — files in Blob appear as lazy-loaded entries in the Lustre filesystem. First read pulls from Blob into Lustre; subsequent reads hit Lustre at full throughput. Writes flush back to Blob periodically. Datasets effectively live in Blob while training I/O hits Lustre speeds.

**2. Azure Managed Lustre + NDR InfiniBand + GPUDirect Storage (GDS)** — GDS establishes a direct data path between the Lustre filesystem and H100 GPU HBM via the same NDR IB fabric. Bypasses host CPU + RAM. Multi-node checkpoint writes happen in seconds.

**Inference uses:** Blob for weights + instance NVMe cache via BlobFuse2 CSI driver. Each ND_v5 has 3.8 TB local NVMe — plenty to cache the ~60 GB weight set.

**Training uses:** Blob for dataset shards (lazy-loaded into Managed Lustre via Blob integration), Managed Lustre with NDR IB + GDS for checkpoint writes and dataset hot tier.

### [Figure 5] Azure storage hierarchy

```
Create a flat-design vector diagram, 16:9 landscape, soft cream background
(#F5F0E8) with subtle grid. Title at top: "Azure Storage Hierarchy —
Inference and Training I/O Profiles".

A vertical four-tier stack on the left, each tier a horizontal rounded
rectangle in increasingly warm colors (cool at top = cold storage,
warm at bottom = hot storage):

Tier 1 (top, gray-blue, "COLD"):
  Azure Blob Storage icon (Hot or Cool tier) labeled "Source datasets,
  model weights, checkpoint archive". Annotation: "moderate throughput,
  cheapest, 11 9s durability with RA-GZRS"

Tier 2 (light blue, "WARM READ"):
  BlobFuse2 CSI + instance NVMe icon labeled "3.8 TB local NVMe SSD
  per Standard_ND96isr_H100_v5". Annotation: "high throughput,
  read-mostly, automatic cache for weights and dataset shards"

Tier 3 (light orange, "HOT READ/WRITE"):
  Azure Managed Lustre cylinder labeled "Lustre + NDR InfiniBand + GDS,
  with Blob integration". Annotation: "150+ GB/s aggregate -
  GPUDirect Storage bypasses host CPU and system RAM, 1.92 TB
  checkpoint dump in ~13 seconds"

Tier 4 (bottom, light red, "PREMIUM SHARED"):
  Azure NetApp Files (Ultra tier) icon. Annotation: "NFS RWX, premium
  pricing, fallback when Lustre doesn't fit your access pattern"

On the RIGHT half of the canvas, two columns showing which workload
uses which tier:

LEFT column (K8s blue header "INFERENCE"):
  Solid arrows from a "NIM inference pod" icon to:
    - Tier 1 (initial weight pull, label "1× at first boot")
    - Tier 2 (warm read cache, label "subsequent restarts")

RIGHT column (K8s blue header "TRAINING"):
  Solid arrows from a "Training pod (LWS leader)" icon to:
    - Tier 1 (dataset source + checkpoint archive, label "via Lustre
      Blob integration")
    - Tier 3 (active checkpoints + dataset hot tier, label "GDS over
      NDR IB, every 1000 steps")

A small callout annotation: "Managed Lustre + Blob integration + GDS
over IB is the Azure killer combo — datasets live in cheap Blob while
training I/O hits Lustre speeds, with GPUs talking directly to Lustre
controllers."

Visual rules: Azure blue (#0078D4) for storage services, K8s blue for
pod icons, NVIDIA green only if showing GPU chips in pods (optional).
Solid arrows for primary I/O, dashed for tier-to-tier promotion.
Render each label exactly once.
```

---

## 6. The seam: how training feeds inference

The Azure ML Model Registry is the source of truth for "which checkpoint is production-ready." Promotion is gated by eval + manual approval; rollout uses **Argo Rollouts** for K8s-native progressive delivery (App Gateway weighted backend pools instead of an AWS-style weighted target group).

### Promotion flow

1. ML Engineer submits NeMo SFT JobSet to training cluster
2. Kueue + Azure Capacity Reservations: wait for 32× H100, admit when ready
3. Training runs ~3 days with async checkpointing every 1000 steps to Azure Managed Lustre → flushed to Blob via integration
4. Final checkpoint `v23` archived in Blob
5. ML Engineer triggers eval pipeline (separate small GPU pool, runs MMLU + HellaSwag + internal evals)
6. Checkpoint `v23` registered in Azure ML Model Registry with eval scores (MMLU, GSM8K, MT-Bench)
7. ML Engineer reviews scores; tags v23 with `prod-candidate` label
8. Event Grid rule on registry change → triggers Argo Rollouts on the inference cluster
9. Argo Rollouts: canary pod with `v23` takes 5% traffic via App Gateway weighted backend for 30-min soak
10. Blue/green rollout to 100% via backend pool swap; previous `v22` kept warm for instant rollback

### [Figure 6] Promotion lifecycle on Azure

```
Create a flat-design vector horizontal timeline diagram, 16:9 landscape,
soft cream background (#F5F0E8) with subtle grid. Title at top:
"Nemotron-3 Model Promotion on Azure — train → eval → register → serve".

Horizontal swim-lane structure with 5 rows stacked top to bottom (each
row gently tinted):

Row 1 (light blue): "ML Engineer" with person icon on far left
Row 2 (light green): "Kueue + Azure On-Demand Capacity Reservations"
Row 3 (K8s blue): "AKS Training Cluster (NeMo + Megatron, 32× H100, NDR IB)"
Row 4 (Azure blue #0078D4): "Azure Blob + Azure ML Model Registry"
Row 5 (bottom, K8s blue): "AKS Inference Cluster (Argo Rollouts canary)"

A THICK CURVED ORANGE RIBBON weaves through these rows left to right
with 10 numbered white circles (bold orange numbers) representing steps:

  1 (Row 1)  ML Engineer submits JobSet (NeMo SFT)
  2 (Row 2)  Kueue + Capacity Reservations: wait for 32× H100, admit
  3 (Row 3)  SFT runs 3 days, async checkpointing every 1000 steps
  4 (Row 4)  Final checkpoint v23 written to Blob via Managed Lustre
  5 (Row 1)  ML Engineer triggers eval pipeline
  6 (Row 3)  Eval (small GPU pool): MMLU + HellaSwag + internal evals
  7 (Row 4)  v23 registered in Azure ML Model Registry with scores
  8 (Row 1)  Manual approval gate — tag v23 as "prod-candidate"
  9 (Row 5)  Argo Rollouts: canary pod takes 5% via App Gateway weighted
             backend, 30-min soak
 10 (Row 5)  Blue/green rollout to 100% (backend pool swap)

Below swim lanes, small legend: "Orange ribbon = promotion flow. White
numbered circles = ordered steps. Time progresses left to right
(~4 days end-to-end). Event Grid triggers Argo Rollouts on registry
state change."

Annotations off to the side at relevant steps:
  - Near step 2: small Azure logo + "gang-scheduled, ND_v5 reservation"
  - Near step 3: small NVIDIA green GPU chip cluster
  - Near step 8: small lock icon "manual approval gate"
  - Near step 9: small chart "App Gateway weighted backends + AMP
    metrics monitored"

Visual rules: orange ribbon dominant; numbered white circles standing
out; swim lanes are subtle background tints. No 3D, no neon. Sans-serif
labels, readable from a slide projection. Render each label exactly once.
```

---

## 7. Inference vs Training — side-by-side

| Dimension | Inference | Training |
|---|---|---|
| **Workload primitive** | `Deployment` + `HPA` + `PDB` | `LeaderWorkerSet` admitted by `Kueue` |
| **Node scheduler** | Karpenter | Karpenter + Azure On-Demand Capacity Reservations |
| **Scaling signal** | `vllm:num_requests_waiting` via AMP + KEDA | Queue admission (Kueue) — no in-job scaling |
| **Node SKU** | `Standard_ND96isr_H100_v5` (8× H100, TP=8, single-node NVLink) | `Standard_ND96isr_H100_v5` (8× H100, multi-node NDR IB) |
| **Cluster size** | Steady 2–4 nodes | 0 idle → bursts to 4–16 for a job |
| **Cluster autoscaling** | Mild (HPA-driven) | Aggressive (0 ↔ N) via Capacity Reservation + Karpenter |
| **Spot eligibility** | Burst only; baseline on Reserved Instances | Risky for long jobs without aggressive checkpointing |
| **Networking** | Single-node NVLink only | NDR InfiniBand 3.2 Tbps cross-node; Proximity Placement Group |
| **Storage I/O** | BlobFuse2 + instance NVMe | Azure Managed Lustre (with Blob integration) + GDS over NDR IB |
| **Lifecycle** | 24/7 service, blue/green via Argo Rollouts | Episodic jobs, days-to-weeks duration |
| **Failure recovery** | Pod restart + HPA backfill + Karpenter | Checkpoint restore + Kueue re-admit; Azure ML for resilience on long jobs |
| **Observability focus** | Latency, throughput, queue depth, cost/1K-tokens | Loss, throughput, GPU util, IB bandwidth |
| **CI/CD trigger** | Push to main → Container Apps revision → Argo Rollouts | Manual job submission or scheduled (Event Grid → Azure ML Pipelines / Airflow on Azure) |
| **Cost predictability** | Predictable monthly baseline | Variable per-experiment |
| **Right-sizing question** | "How many replicas for our QPS?" | "How many GPU-hours for this experiment?" |

---

## 8. Cost shapes

| Workload | Driver | Azure discount strategy |
|---|---|---|
| **Inference** | Baseline replicas × hours × per-VM rate | **3-yr Reserved VM Instance** OR **Azure Savings Plan for Compute** on baseline. 1-yr RI for less commitment risk. Burst on on-demand or Spot. |
| **Training** | GPU-hours per experiment | **Azure On-Demand Capacity Reservation** for predictable batch (guarantees ND_v5 availability for scheduled runs). 1-yr RI on steady minimum if continuous. Spot VMs for short LoRA runs. **Azure ML** for resilience-critical long jobs. |

### Realistic annual cost example

Assumptions: 2 inference replicas × 8 H100 with HPA to 4 replicas at peak; one SFT every 2 weeks (32× H100 × 3 days); one LoRA per week (8× H100 × 8 hours).

| Line item | Calc | Annual |
|---|---|---|
| Inference baseline (2 nodes × ND_v5, 3-yr RI ~50% off list ~$100/hr) | 2 × 50 × 8760 | **~$876K** |
| Inference burst (avg 1 node extra × 200 hr/mo, Spot ~$30) | 1 × 30 × 2400 | **~$72K** |
| Training SFT (4 nodes × 72 hr × 26 runs, Capacity Reservation pricing) | 4 × 95 × 1872 | **~$711K** |
| Training LoRA (1 node × 8 hr × 52, on-demand) | 1 × 100 × 416 | **~$42K** |
| Storage (Blob + Managed Lustre + Cache for Redis + Cosmos DB) | | **~$50K** |
| Networking (NAT, Private Link, Front Door, App Gateway, egress) + observability + APIM + Container Apps | | **~$65K** |
| **Total platform** | | **~$1.82M / year** |

Azure ND_v5 list pricing varies by region; East US 2 and South Central US tend to have capacity. Numbers above are illustrative — use Azure Pricing Calculator for actual quotes.

---

## 9. Open decisions

1. **Single cluster vs separate clusters?** Recommend separate (same reasoning as GCP/AWS).

2. **Karpenter vs AKS Cluster Autoscaler?** **Karpenter on AKS** if it's GA in your region — better diversity, spot handling, scale-to-zero. Cluster Autoscaler is the conservative fallback.

3. **Self-host on AKS vs Azure Machine Learning vs Azure HPC SKUs directly?**
   - **Azure ML** — managed alternative, handles cluster reconfig, auto-recovery, abstracts InfiniBand drivers. Best for multi-week pre-training where node failures would otherwise abort. Tradeoff: less flexibility, ~10-20% management premium.
   - **Self-host on AKS** — most flexibility, most operational burden.
   - **For most teams, evaluate Azure ML first** for long jobs; use AKS for shorter/experimental work.

4. **NeMo vs Megatron-LM directly vs HuggingFace?** NeMo for Nemotron (same as GCP/AWS). HuggingFace + Accelerate works for LoRA but not the MoE + Mamba hybrid in full-param training.

5. **API Management vs App Gateway-direct for SSE?** **App Gateway-direct** for the streaming path (APIM has request timeout concerns). APIM is fine for non-streaming endpoints. The Container Apps gateway sits in front of AKS regardless.

5a. **Human-user auth: Microsoft Entra ID vs Azure AD B2C vs Verified Access pattern?**
   - **Microsoft Entra ID** (formerly Azure AD) + App Gateway OIDC — internal tools, employee SSO
   - **Azure AD B2C** — B2C user signup, social login, custom user journeys
   - **Microsoft Entra Verified ID** — verifiable credentials for zero-trust scenarios

5b. **Kueue vs Azure Batch?** **Kueue** for K8s-native (matches GCP/AWS for cross-cloud consistency). Azure Batch is the older Azure-native batch service; reasonable if your org already runs it.

6. **AKS Workload Identity vs deprecated AAD Pod Identity v1?** **Always Workload Identity** (Pod Identity v1 deprecated in 2023). Uses OIDC federation: AKS issues signed service account tokens, Microsoft Entra ID trusts them and swaps for Azure access tokens. No local secret storage, no node-level identity agent.

7. **Container Apps vs AKS-hosted Model Gateway?**
   - **Container Apps (recommended)** — managed serverless (KEDA + Envoy + Dapr), TLS termination, scales to zero, doesn't add load to GPU cluster
   - **AKS-hosted** — lower cross-service latency (same VNet, local pod-to-pod), no Container Apps resource limits
   - **Decision: Container Apps for v1.** Refactor to AKS-hosted only if p95 cross-service latency > 5ms.

8. **Azure Managed Lustre vs Azure NetApp Files for training checkpoints?** **Managed Lustre with GDS** is the better fit for 120B-class training due to higher aggregate throughput and direct-from-GPU writes. NetApp Files is the fallback for NFS RWX semantics or if Managed Lustre isn't available in your region.

9. **Where to host the Model Gateway?** Three options:
    - **Container Apps** (recommended) — serverless containers, supports SSE, autoscales
    - **Azure Functions** — 15-min timeout makes SSE awkward; only viable for non-streaming
    - **AKS pods** — coupling concern; not recommended for baseline

---

## Appendix A — Three-cloud component mapping

Quick reference for engineers translating between GCP, AWS, and Azure.

| Component | GCP | AWS | Azure |
|---|---|---|---|
| Project / account | GCP Project | AWS Account | Azure Subscription |
| Network | VPC + subnet | VPC + subnets | VNet + subnets |
| Kubernetes | GKE Standard | Amazon EKS | AKS |
| Node autoscaler | Cluster Autoscaler | **Karpenter** | Karpenter on AKS (preview/GA) |
| H100 GPU SKU | `a3-highgpu-8g` / `a3-megagpu-8g` | `p5.48xlarge` | `Standard_ND96isr_H100_v5` |
| GPU cohort scheduling | DWS Flex-start | EC2 Capacity Blocks for ML | Azure On-Demand Capacity Reservations |
| Multi-node fabric | GPUDirect TCPXO (custom OS-bypass) | EFA + SRD | **NDR InfiniBand (real RDMA)** |
| Multi-node bandwidth | 1.6 Tbps (A3-Mega) | 3.2 Tbps (p5) | 3.2 Tbps (ND_v5) |
| Compact placement | Compact Placement Policy | Cluster Placement Group / UltraCluster | Proximity Placement Group |
| Workload Identity | Workload Identity (KSA → GSA) | IRSA | AKS Workload Identity (KSA → Entra App) |
| Object storage | GCS | S3 | Blob Storage |
| Object storage CSI | GCS-FUSE CSI | Mountpoint for S3 CSI | Azure Blob CSI (BlobFuse2) |
| HPC filesystem | Filestore High Scale / Parallelstore | FSx for Lustre | **Azure Managed Lustre** |
| Container registry | Artifact Registry | ECR | Azure Container Registry (ACR) |
| Secrets | Secret Manager | AWS Secrets Manager | Azure Key Vault |
| Key management | Cloud KMS | AWS KMS | Azure Key Vault Managed HSM |
| Managed Prometheus | Managed Service for Prometheus | Amazon MSP | Azure Monitor MSP |
| Dashboards | Cloud Monitoring | CloudWatch + Amazon Managed Grafana | Azure Monitor + Azure Managed Grafana |
| Logs | Cloud Logging → BigQuery | CloudWatch Logs → S3 → Athena | Log Analytics → Azure Data Explorer |
| Tracing | Cloud Trace | AWS X-Ray | Application Insights |
| Model registry | Vertex AI Model Registry | SageMaker Model Registry | Azure ML Model Registry |
| Managed training | Vertex AI Training | SageMaker Training / HyperPod | Azure Machine Learning |
| Edge / CDN | Global HTTPS LB (anycast) | CloudFront | Azure Front Door |
| WAF / DDoS | Cloud Armor | AWS WAF + Shield | Azure WAF + DDoS Protection |
| API management | Apigee X | Amazon API Gateway | Azure API Management |
| Human-auth gateway | Identity-Aware Proxy (IAP) | IAM Identity Center + ALB OIDC | Microsoft Entra ID + App Gateway OIDC |
| Serverless containers (Model Gateway) | Cloud Run | **ECS Fargate** | **Azure Container Apps** |
| Session cache | Memorystore (Redis) | ElastiCache for Redis | Azure Cache for Redis |
| Document DB | Firestore | DynamoDB | Cosmos DB (NoSQL API) |
| Serverless analytics SQL | BigQuery | Athena | Azure Data Explorer |
| Progressive rollout | Cloud Deploy | **Argo Rollouts** | **Argo Rollouts** |
| DNS | Cloud DNS | Route 53 | Azure DNS |
| L7 LB | Global HTTPS LB | Application Load Balancer (ALB) | Application Gateway / Front Door |
| K8s service to LB binding | Network Endpoint Group (NEG) | ALB target type IP | Application Gateway for Containers |
| Private service access | Private Service Connect (PSC) | VPC Endpoints (PrivateLink) | Azure Private Link |
| Cross-region replication | Multi-regional bucket | S3 Cross-Region Replication | RA-GRS / RA-GZRS Blob |
| Cost commitment | 1-yr or 3-yr CUD | Compute / EC2 Savings Plan | Reserved VM Instances / Savings Plan for Compute |
| Burst pricing | Spot VMs | EC2 Spot Instances | Azure Spot VMs |

---

## Appendix B — Decision quick reference

| If you need to... | Use |
|---|---|
| Deploy the model for production serving | Inference AKS cluster, Section 3, dense path (8× H100 TP=8 on `Standard_ND96isr_H100_v5`) |
| Fine-tune Nemotron with your data (LoRA) | Training AKS cluster, single-node Job, NeMo PEFT |
| Full SFT or DPO alignment | Training AKS cluster, multi-node LWS + Kueue + Capacity Reservations, NeMo |
| Pre-train from scratch | **Azure Machine Learning** (managed cluster, fault recovery) |
| Add a new tenant with custom rate limits | API Management products + the Container Apps gateway |
| Test the architecture without GPU quota | AKS mock profile (Helm `values-mock.yaml` adapted) running on E-series instances |
| Promote a new model to production | Eval pipeline → Azure ML Model Registry → Event Grid → Argo Rollouts canary → blue/green |

---

## Appendix C — What's NOT in this design

Documented as future increments, same as the GCP and AWS docs:

- Multi-region active-active (trigger: 10,000+ concurrent users or strict cross-continent latency SLA)
- Disaggregated prefill/decode (trigger: > 1,000 RPS sustained)
- Per-tenant LoRA variants (trigger: enterprise tenants demanding fine-tuned model behavior)
- Azure OpenAI / Azure AI Foundry comparison — Azure offers Nemotron-class models as managed endpoints; TCO comparison worth running before committing to self-hosted on AKS at this scale

---

*This document mirrors `inference-and-training.md` (GCP) and `inference-and-training-aws.md` (AWS) for Microsoft Azure. The Nemotron-specific sizing math (Mamba SSM cache, MoE 12B active params, NVFP4 precision) is identical across clouds. Operational details differ — see Appendix A for component mappings.*
