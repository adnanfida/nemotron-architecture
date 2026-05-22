# Architecture: Nemotron-3 Super 120B on GCP — Inference and Training

**Status:** Design proposal
**Last updated:** 2026-05-22
**Audience:** Platform engineering, ML platform, FinOps

This document specifies a complete GCP architecture for both **serving** and **training** Nemotron-3 Super 120B-A12B. The two workloads share a foundation (one GCP project, VPC, IAM, observability, model artifact plane) but run on **separate GKE clusters** with different node SKUs, schedulers, networking, and lifecycles.

The seam between them is the **Vertex AI Model Registry**: training writes versioned checkpoints to GCS and registers them; inference consumes approved checkpoints via a promotion pipeline.

---

## 1. Why Nemotron-3 changes the sizing math

Most 120B-class models would saturate a multi-node cluster just to fit weights + KV cache. Nemotron-3 Super is different on three counts:

**Hybrid Mamba-Transformer (Mamba-2 SSM + select attention layers)** — most layers are Mamba-2 state-space, not attention. Mamba uses a **fixed-size recurrent state cache** that does *not* grow with context length. Only the ~16 attention layers contribute to a per-token KV cache. Net effect: KV-cache memory is roughly 3–5× smaller than a pure-Transformer 120B at equivalent context.

**MoE with 12B active parameters** — 120B total parameters must be in VRAM (experts can't swap on the fly without destroying latency), but only 12B activate per token. Effective inference compute is closer to a 12B dense model than a 120B dense model.

**Native NVFP4 (4-bit) precision** — weights are ~60GB instead of ~240GB at BF16.

**Concrete consequence for sizing:** at 200 concurrent streams × 3,000-token context in FP8 KV, total runtime cache is ~15GB. A single `a3-highgpu-8g` node (8× H100, 640GB VRAM total) holds 120GB of weights distributed across 8 GPUs (15GB per GPU) and has ~440GB of cache headroom across the node. **One node can support thousands of concurrent streams before hitting memory limits**; the bottleneck shifts from memory to compute throughput long before VRAM.

This makes the deployment dramatically smaller than equivalent 120B-Transformer architectures. It does *not* eliminate the need for multi-replica HA, scale-out for compute, or the surrounding production architecture.

---

## 2. Shared GCP foundation

| Component | Purpose | Shared by |
|---|---|---|
| **GCP project** `nemotron-platform` | One billing/audit boundary | Both |
| **Shared VPC** (us-central1, 3 zones) | One network, Cloud NAT for egress | Both |
| **VPC Service Controls** | Exfil prevention from compromised pods | Both |
| **Secret Manager** | NGC API key, HF token | Both |
| **Cloud KMS** | CMEK keys for weights, checkpoints, Firestore | Both |
| **GCS `nemotron-weights`** (multi-regional) | Source of truth for model weights. Training writes; inference reads | Both (write/read separated by IAM) |
| **Artifact Registry** | NIM containers + training framework images | Both |
| **Vertex AI Model Registry** | Versioned checkpoints with eval metadata — the promotion gate | Both |
| **Managed Prometheus + Cloud Monitoring** | Metrics federation across both clusters | Both |
| **Cloud Logging → BigQuery sink** | Long-term log analysis | Both |

### [Figure 1] Nano-banana prompt — Shared foundation

```
Create a clean technical cloud architecture diagram in flat-design vector
illustration style, 16:9 landscape, soft cream/beige background (#F5F0E8)
with subtle grid pattern. Title at top: "Nemotron-3 on GCP — Shared
Platform Foundation".

A single large rounded rectangle in the center labeled "GCP Project:
nemotron-platform" contains the entire scene. Inside that, a slightly
smaller rounded rectangle labeled "Shared VPC — us-central1, 3 zones".

Inside the VPC, two side-by-side Kubernetes cluster blocks in K8s blue
(#326CE5):
  - LEFT: "Inference Cluster (GKE Standard)" with badge "always-on 24/7"
  - RIGHT: "Training Cluster (GKE Standard)" with badge "on-demand bursts"

Below the VPC, three horizontal "shared services" strips in GCP red/coral:

Strip 1 — Artifacts (left to right):
  - GCS bucket "nemotron-weights (multi-region)"
  - Container icon "Artifact Registry"
  - Database icon "Vertex AI Model Registry"

Strip 2 — Identity & Secrets:
  - Chain-link "Workload Identity"
  - Key "Secret Manager"
  - Lock "Cloud KMS (CMEK)"

Strip 3 — Observability:
  - Flame "Managed Prometheus"
  - Chart "Cloud Monitoring"
  - Stacked-logs "Cloud Logging → BigQuery"

Dashed arrows from both cluster blocks down to each strip with small
labels: "writes / reads" on artifacts, "binds" on identity, "emits" on
observability. Subtle corner annotation: "Training writes weights;
Inference reads them. Vertex AI Model Registry is the promotion gate."

Visual rules: flat 2D, soft drop shadows, rounded corners, NVIDIA green
only on GPU chips (none in this figure). Render each label exactly once.
```

---

## 3. Inference architecture

**Sized for:** 2,000 active users → ~200 concurrent streams at 10% peak concurrency, TTFT < 500ms, inter-token latency < 25ms, > 99.9% monthly availability.

### Cluster shape

**Node SKU choice — two viable paths:**

| Path | Per-replica | Replicas for 200 concurrent | Total GPUs | Notes |
|---|---|---|---|---|
| **Dense** (recommended for this model) | `a3-highgpu-8g`, 8× H100, TP=8 | 2–3 (HA across zones) | 16–24 | Single-node NVLink only. One replica handles 1000+ concurrent given Mamba math. Spark's architecture. |
| **Distributed** | `a3-highgpu-2g`, 2× H100, TP=2 | 6–10 | 12–20 | More granular scaling. Closer to my original recommendation. Less efficient for Nemotron specifically. |

**For Nemotron specifically, the Dense path wins.** Mamba's fixed-size SSM cache means per-replica concurrent capacity is enormous; you want fewer, larger replicas. Distributed makes sense for dense Transformer 120Bs where each replica saturates faster.

### Edge → routing → inference

```
Client → Cloud DNS → Cloud Armor → Global HTTPS LB → Apigee X →
Model Gateway (Cloud Run × 3) → GKE Service (NEG) → NIM pods (8× H100)
```

**Each layer's job:**

- **Cloud DNS + Cloud Armor + Global HTTPS LB** — anycast TLS termination, WAF, per-IP DDoS rate limiting
- **Apigee X** — per-tenant API keys, quotas, spike arrest, request logging to BigQuery for cost attribution. (Substitute: Cloud API Gateway if <10 tenants and SLA differentiation is shallow)
- **Identity-Aware Proxy (IAP)** — optional alternative path for human-user OAuth/SSO, e.g., internal tools
- **Cloud Run Model Gateway (3 always-on instances)** — SSE streaming proxy, session-affinity routing via Memorystore (`(tenant_id, session_id) → replica hint`), prompt shaping, token-count cost emit to Pub/Sub → BigQuery, cancellation propagation. Long-lived SSE connections live here, not on GPU pods.
- **GKE Service exposed via NEG** — Cloud Run gateway targets pod IPs directly
- **NIM pods** — `nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b`, TP=8 on a single a3-highgpu-8g node

### Inference workload spec

| Setting | Value | Why |
|---|---|---|
| Deployment replicas (baseline) | 2 (one per zone) | Bare minimum HA across 2 of 3 zones |
| Deployment replicas (with HPA headroom) | 3–4 | N+1 redundancy so a zone loss doesn't push remaining to 100% |
| HPA metric | `vllm:num_requests_waiting` queue depth | CPU-based HPA never triggers on GPU pods |
| Target | queue depth < 4 per replica | |
| PDB | `minAvailable: 2` | Tolerates rolling updates without dropping below HA floor |
| Topology spread | 3-zone, maxSkew=1 | At least one replica per zone |
| Weights mount | GCS-FUSE CSI driver + local NVMe SSD cache (6.2TB available per a3 node) | Warm replicas don't re-read from GCS on restart |
| Probes | startupProbe 30-min grace, readiness, liveness | Weight load takes 5–15 min on first boot |
| Service | ClusterIP + NEG annotation | Container-native LB targeting |
| Session affinity | ClientIP or header-based | KV-cache prefix reuse on attention layers |

### Inference cluster topology details

- **Cluster:** Regional GKE Standard, private nodes (no public IPs), Workload Identity, Managed Prometheus enabled, GCS-FUSE CSI driver enabled at cluster level
- **Node pools:** `system-pool` (3× e2-standard-4 for gateway proxies, daemonsets); `gpu-pool` (a3-highgpu-8g, autoscaling 0→4) — 0 baseline so idle dev clusters don't bill
- **Networking:** Private cluster needs Cloud NAT for image pulls (until Artifact Registry caches them intra-region)

### State plane (production app concerns)

| Component | Purpose | Sizing |
|---|---|---|
| **Memorystore Redis** | Session-affinity hints, per-tenant rate counters, idempotency keys | Standard tier 5GB |
| **Firestore (Native)** | Durable conversation history per tenant | Per-tenant document model |
| **BigQuery** | Request logs from Apigee + token usage from gateway → cost analytics | Long-term sink |
| **Artifact Registry** | Intra-region cached copy of NIM container | Critical for fast pod startup on scale-up |

### Observability

- **Managed Prometheus** scrapes `vllm:*` and DCGM-exporter (GPU util, HBM, NVLink throughput, temperature)
- **Cloud Monitoring dashboards** — queue depth, GPU utilization, KV-cache fill, p50/p95/p99 first-token and inter-token latency, cost per 1K tokens (joined with BigQuery billing)
- **Cloud Trace** — request traces from gateway through pod with token counts in span attributes
- **SLOs:** p95 first token < 500ms, p99 inter-token < 25ms, monthly availability > 99.9%
- **Alerts:** queue-depth-high (precedes saturation), pod restart rate, p99 latency breach, GPU OOM, daily cost overrun

### [Figure 2] Nano-banana prompt — Inference architecture

```
Create a clean technical cloud architecture diagram in flat-design vector
illustration style, 16:9 landscape, soft cream background (#F5F0E8) with
subtle grid. Title at top: "Nemotron-3 Inference — 2,000 Users on GKE".

Five horizontal layers stacked top to bottom with solid arrows between
layers (request flow) and dashed arrows to state stores on the side.

Layer 1 (edge / security, green tones):
Laptop icon "Client" → globe "Cloud DNS" → shield "Cloud Armor (WAF +
DDoS)" → LB icon "Global HTTPS LB (TLS, anycast)"

Layer 2 (API management, purple):
Hexagonal block "Apigee X — API keys, per-tenant quotas, billing tags"

Layer 3 (application gateway, Cloud Run orange):
Three Cloud Run service tiles side by side labeled "Model Gateway × 3
(Cloud Run) — SSE proxy, session-affinity routing, cost tracking"

Layer 4 (GKE inference plane, large K8s blue rectangle taking 50% of
canvas):
Top of rectangle: "GKE Standard Cluster — us-central1 (3 zones, private)"
Inside: three vertical zone columns. In each column, ONE LARGE pod card
labeled "NIM Pod" containing 8 small NVIDIA-green (#76B900) H100 chips
arranged in a 2x4 grid with label "8× H100 TP=8 (NVLink)". A small badge
above the row: "Nemotron-3 Super 120B-A12B (NVFP4) · 2 baseline → 4 HPA".
Left side of cluster: GCS-FUSE CSI icon with sidecar file-cache label
"+ 6.2TB local NVMe cache". Right side: chain icon labeled "Workload
Identity → GSA: nemotron-gsa". Bottom: small gauge "HPA: vllm queue
depth" and circle "PDB minAvail=2".

Layer 5 (state + storage, GCP red row):
Six cylinder/bucket icons: Cloud KMS (key), GCS Multi-Region (weights),
Artifact Registry (NIM cache), Memorystore (Redis), Firestore (chat
history), BigQuery (logs + analytics).

Right-side observability rail (vertical, dashed connections to gateway
and GKE): Managed Prometheus → Cloud Monitoring → Cloud Logging →
Cloud Trace.

Visual rules: K8s blue (#326CE5) for cluster, GCP red/coral for state,
purple for Apigee, Cloud Run orange for gateway, NVIDIA green only on
GPU chips. Solid bold arrows for request flow (top to bottom); dashed
arrows for state and observability. Render each label exactly once.
```

---

## 4. Training architecture

### Workload types and recommended patterns

| Workload | Compute | Duration | Recommended pattern |
|---|---|---|---|
| LoRA / QLoRA / adapter fine-tuning | 1× a3-highgpu-8g (8 H100) | Hours | Single-node `Job`, NeMo PEFT or HF TRL |
| Full SFT (Supervised Fine-Tuning) | 4× a3-megagpu-8g (32 H100) | 1–3 days | Multi-node `LeaderWorkerSet` + Kueue, NeMo |
| DPO / RLHF alignment | 4–16× a3-megagpu-8g (32–128 H100) | 3–7 days | Multi-node LWS + Kueue, NeMo-Aligner |
| Continued pre-training | 16–64× a3-megagpu-8g (128–512 H100) | Weeks | Multi-node LWS + Kueue + DWS Flex-start |
| Full pre-training from scratch | 1000+ H100 | Months | **Consider Vertex AI Training instead** |

### Training memory math (full parameter, BF16 mixed precision with AdamW)

For Full SFT or continued pre-training of the 120B parameter model:

| Component | Size | Calculation |
|---|---|---|
| Model weights (BF16) | 240 GB | 120B × 2 bytes |
| Gradients (BF16) | 240 GB | 120B × 2 bytes |
| Optimizer states (FP32 AdamW: master weights + momentum + variance) | 1,440 GB | 120B × 12 bytes |
| **Static state total** | **1,920 GB** | |
| + Activation memory (during backprop) | varies | Mitigated by activation checkpointing + FlashAttention-2 |

**Cluster sizing:** 4× a3-megagpu-8g = 32× H100 = 2,560 GB cluster VRAM. Sharded via ZeRO-3 / FSDP → 60 GB static per GPU, leaves 20 GB for activations.

### Cluster topology

- **Cluster:** Regional GKE Standard, separate from inference cluster, private nodes, Workload Identity, GPU-Direct CSI driver enabled
- **Node pools:**
  - `system-pool` (3× e2-standard-4) — hosts Kueue controller, JobSet API, custom-metrics adapter
  - `gpu-pool-mega` (a3-megagpu-8g, autoscaling 0→16) — provisioned only when a job is admitted by Kueue
  - Optional `gpu-pool-spot` (same SKU, --spot) — burst capacity for non-critical fine-tunes with aggressive checkpointing
- **Placement:** Compact placement policy on GPU pool (all nodes for one job on the same network spine — required for GIB bandwidth)

### Orchestration stack

| Layer | Tool | Purpose |
|---|---|---|
| **GCP cohort allocation** | **Dynamic Workload Scheduler (DWS) Flex-start** | Allocates all N GPU nodes atomically; cheaper than on-demand for batch workloads. Billing clock starts only when full cohort is available |
| **K8s job queueing** | **Kueue** | Fair sharing across teams, priority preemption, gang scheduling at the K8s layer |
| **Multi-job topology** | **JobSet API** (sig-apps) | Defines groups of related jobs (trainer + data loader + evaluator) |
| **Multi-node single-replica** | **LeaderWorkerSet (LWS)** | The leader + N workers pattern that distributed training needs |
| **Single-node single-replica** | K8s `Job` | LoRA, eval, data prep |

### Framework choice — NeMo for Nemotron specifically

**Recommended:** NVIDIA **NeMo Framework** + **NeMo-Aligner** (for RLHF).

**Why NeMo, not generic HF + DeepSpeed:** Nemotron's MoE + Mamba-2 hybrid architecture requires expert parallelism and SSM-aware training. NeMo has first-party support; it wraps Megatron-LM (which provides TP / PP / EP / DP) with PyTorch Lightning ergonomics. HuggingFace + DeepSpeed/FSDP works for LoRA on the attention layers but doesn't handle Mamba layers cleanly for full-parameter training.

### Networking — Multi-NIC + GPUDirect TCPXO

Multi-node distributed training is bottlenecked by gradient sync (NCCL `All-Reduce`). Single-NIC VMs can't sustain this. GCP's solution:

| SKU | NIC topology | Bandwidth | Use |
|---|---|---|---|
| a3-highgpu-8g | 4+1 (4 GPU NICs + 1 management) | 800 Gbps | Smaller multi-node jobs |
| **a3-megagpu-8g** | **8+1 (one NIC per GPU + 1 management)** | **1.6 Tbps** | **Standard for 120B-class training** |
| a3-ultragpu-8g | 8+1 (GIB) | 3.2 Tbps | Bigger optimizer states (H200 141GB) |

**GPUDirect TCPXO** lets the GPU write packets directly to the NIC ring buffer, bypassing host kernel and CPU — near-RDMA speeds over standard VPC.

**VPC architecture required:** one primary VPC + 8 secondary VPCs (one per GPU NIC). GKE configures this via Multus CNI and `NetworkAttachmentDefinitions`. The scheduler only places GPU pods on nodes where all 9 interfaces are healthy.

### [Figure 3] Nano-banana prompt — Training cluster topology

```
Create a flat-design vector technical diagram, 16:9 landscape, soft cream
background (#F5F0E8) with subtle grid. Title at top: "Nemotron-3 Training
Cluster — multi-node distributed training on GKE".

A single large rounded rectangle in K8s blue (#326CE5) labeled at top
"GKE Training Cluster — us-central1, single placement group, private".

Inside, two horizontal node-pool sections stacked:

Top section — "system-pool (3× e2-standard-4)" with three small pills:
  - "Kueue (queue + gang scheduling)"
  - "JobSet API + LeaderWorkerSet"
  - "Custom metrics adapter"

Bottom section — "gpu-pool-mega (a3-megagpu-8g, autoscale 0 → 16 nodes
via DWS Flex-start)" takes most of the canvas. FOUR node boxes side by
side, each containing 8 small NVIDIA-green (#76B900) H100 chips in a
2x4 grid, labeled "Node 1: 8× H100", "Node 2: 8× H100", etc. THICK
neon-green double-arrow bars between adjacent nodes labeled "GPUDirect
TCPXO  1.6 Tbps (8+1 multi-NIC)" — these are the visual centerpiece,
showing the high-bandwidth interconnect glowing softly.

Overlaid on the gpu-pool, semi-transparent rectangles representing one
active training job (LeaderWorkerSet pattern):
  - "leader pod (rank 0)" overlapping leftmost GPU on Node 1
  - "worker pods (ranks 1-7)" spanning rest of Node 1
  - "worker pods (ranks 8-15)" spanning Node 2
  - "worker pods (ranks 16-23)" spanning Node 3
  - "worker pods (ranks 24-31)" spanning Node 4
  - Badge above: "NeMo + Megatron, FSDP / ZeRO-3, TP=8 PP=4"

Outside the cluster on the right edge, GCP-red icons connected by
dashed arrows to the leader pod:
  - GCS bucket "training dataset (streaming via Mosaic / WebDataset)"
  - Cylinder "Filestore High Scale (RWX NFS, 25 GB/s write)"
  - GCS bucket "checkpoint cold archive"

Below the cluster: Vertex AI Model Registry icon with dashed arrow from
checkpoint archive labeled "promotion path → inference cluster".

Right edge observability fan-out: dashed arrows from leader to small
icons labeled "Managed Prometheus + TensorBoard" and "W&B / Vertex AI
Experiments".

Visual rules: K8s blue for cluster boundaries, NVIDIA green ONLY on GPU
chips and the interconnect bars (which should glow softly), GCP red/coral
for storage and registry, soft drop shadows. Compact placement implied
by the dense node arrangement. Render each label exactly once.
```

### [Figure 4] Nano-banana prompt — Multi-NIC topology detail

```
Create a flat-design vector technical diagram, 16:9 landscape, soft cream
background (#F5F0E8) with subtle grid. Title at top: "GPUDirect TCPXO
Multi-NIC Topology — a3-megagpu-8g node".

Center the diagram on a SINGLE large rounded rectangle labeled "GKE Pod:
PyTorch Training Worker (rank N)". Inside the pod box, on the left,
stack 9 horizontal labeled lines representing network interfaces:

  eth0   "Primary VPC (10.0.0.0/16) → Cluster control, git, Docker pull"
  net1   "GPU VPC 1 (192.168.1.0/24) → GPU 0 / gVNIC 1"
  net2   "GPU VPC 2 (192.168.2.0/24) → GPU 1 / gVNIC 2"
  net3   "GPU VPC 3 (192.168.3.0/24) → GPU 2 / gVNIC 3"
  net4   "GPU VPC 4 (192.168.4.0/24) → GPU 3 / gVNIC 4"
  net5   "GPU VPC 5 (192.168.5.0/24) → GPU 4 / gVNIC 5"
  net6   "GPU VPC 6 (192.168.6.0/24) → GPU 5 / gVNIC 6"
  net7   "GPU VPC 7 (192.168.7.0/24) → GPU 6 / gVNIC 7"
  net8   "GPU VPC 8 (192.168.8.0/24) → GPU 7 / gVNIC 8"

On the right side of the pod, 8 NVIDIA-green (#76B900) H100 GPU chips
in a 2x4 grid. Each GPU has a thick green arrow connecting to its
matching gVNIC interface on the left (GPU 0 → net1, GPU 1 → net2, etc.)
labeled "GPUDirect TCPXO (kernel-bypass)".

Above the pod, a small "Multus CNI + NetworkAttachmentDefinitions"
badge.

Below the pod, dashed thick arrows fanning out to peer training pods
on other nodes, all converging through a faint cloud labeled "Compact
Placement — same network spine, 1.6 Tbps aggregate".

A small annotation in the corner: "Each gVNIC = 200 Gbps. 8 NICs × 200
Gbps = 1.6 Tbps inter-node clustering bandwidth. Bypasses host kernel
and CPU."

Visual rules: K8s blue for pod boundary, NVIDIA green for GPU chips and
the 8 high-bandwidth arrows, GCP coral/orange for the multus + CNI
labels, neutral gray for the primary management NIC. Render each label
once. Sans-serif, readable.
```

---

## 5. Storage hierarchy

Training and inference have different I/O profiles. The storage hierarchy serves both.

| Tier | Service | Throughput | Cost | Used for |
|---|---|---|---|---|
| **Cold archive** | GCS Multi-regional | Moderate | Cheapest | Source datasets, checkpoint archive, model weights distribution |
| **Hot read cache** | GCS-FUSE + local NVMe SSD (6.2TB per a3 node) | High | Near-free | Inference weight cache, training dataset shards |
| **Hot read/write** | **Filestore High Scale (pNFS)** | 25 GB/s write / 40 GB/s read | Medium | Multi-node training checkpoint writes (RWX NFS mount) |
| **Hot read/write extreme** | **Parallelstore (managed Lustre)** | 100+ GB/s | Expensive | Only needed if Filestore is the bottleneck (rare for 120B SFT) |

**Inference uses:** GCS multi-regional for weights + local NVMe cache via GCS-FUSE CSI. ~60GB weight set means each pod can pre-cache all weights to local SSD on first warm; subsequent restarts are fast.

**Training uses:** GCS for dataset shards (streamed via Mosaic / WebDataset), Filestore High Scale for checkpoints. A 1.92TB full state dump completes in under 80 seconds at 25 GB/s — async checkpointing keeps GPU utilization above 95%.

### [Figure 5] Nano-banana prompt — Storage hierarchy

```
Create a flat-design vector diagram, 16:9 landscape, soft cream background
(#F5F0E8) with subtle grid. Title at top: "Storage Hierarchy — Inference
and Training I/O Profiles".

A vertical four-tier stack on the left, each tier a horizontal rounded
rectangle in increasingly warm colors (cool at top = cold storage, warm
at bottom = hot storage):

Tier 1 (top, gray-blue, "COLD"):
  GCS Multi-regional buckets labeled "Source datasets, model weights
  distribution, checkpoint archive". Annotation: "moderate throughput,
  cheapest tier"

Tier 2 (light blue, "WARM READ"):
  GCS-FUSE CSI + local NVMe SSD icon labeled "6.2 TB local SSD cache
  per a3 node". Annotation: "high throughput, near-free, automatic
  read cache for weights and dataset shards"

Tier 3 (light orange, "HOT READ/WRITE"):
  Filestore High Scale icon labeled "Parallel NFS (pNFS) - RWX mount".
  Annotation: "25 GB/s write, 40 GB/s read - multi-node checkpoints
  in under 80 seconds"

Tier 4 (bottom, light red, "EXTREME"):
  Parallelstore (managed Lustre) icon. Annotation: "100+ GB/s, expensive,
  use only when Filestore is the bottleneck"

On the RIGHT half of the canvas, two columns showing which workload
uses which tier:

LEFT column (K8s blue header "INFERENCE"):
  Solid arrows from a "NIM inference pod" icon to:
    - Tier 1 (initial weight pull, label "1x at first boot")
    - Tier 2 (warm read cache, label "subsequent restarts")

RIGHT column (K8s blue header "TRAINING"):
  Solid arrows from a "Training pod (LWS leader)" icon to:
    - Tier 1 (dataset source + checkpoint archive, label "stream + sync")
    - Tier 2 (dataset shard cache, label "streaming reads")
    - Tier 3 (active checkpoints, label "async write every 1000 steps")

Visual rules: GCP red/coral for storage cylinders, K8s blue for pod
icons, NVIDIA green only if showing GPU chips in pods (optional).
Solid arrows for primary I/O, dashed for tier-to-tier promotion. Render
each label exactly once.
```

---

## 6. The seam: how training feeds inference

The Vertex AI Model Registry is the source of truth for "which checkpoint is production-ready." Promotion is gated by eval + manual approval; rollout is blue/green at the Service label level (not rolling, because model weight changes are higher-risk than container changes).

### Promotion flow

1. ML Engineer submits NeMo SFT JobSet to training cluster
2. Kueue + DWS waits for 32× H100, admits when full cohort available
3. Training runs ~3 days with async checkpointing every 1000 steps to Filestore → backed to GCS
4. Final checkpoint `v23` archived in GCS
5. ML Engineer triggers eval pipeline (separate small GPU pool, runs MMLU + HellaSwag + internal evals)
6. Checkpoint `v23` registered in Vertex AI Model Registry with eval scores
7. ML Engineer reviews scores; promotes `v23` to `prod-candidate`
8. Cloud Deploy: canary pod with `v23` takes 5% traffic for 30-min soak
9. Blue/green rollout to 100% via Service label switch
10. `v23` serving production; previous `v22` Deployment kept alive for instant rollback

### [Figure 6] Nano-banana prompt — Promotion lifecycle (horizontal timeline)

```
Create a flat-design vector horizontal timeline diagram, 16:9 landscape,
soft cream background (#F5F0E8) with subtle grid. Title at top:
"Nemotron-3 Model Promotion — train → eval → register → serve".

Horizontal swim-lane structure with 5 rows stacked top to bottom (each
row a different gently tinted background band):

Row 1 (light blue): "ML Engineer" — person icon on far left
Row 2 (light green): "Kueue + DWS (training queue)"
Row 3 (K8s blue): "Training Cluster (NeMo + Megatron on 32× H100)"
Row 4 (GCP red/coral): "GCS weights + Vertex AI Model Registry"
Row 5 (bottom, K8s blue): "Inference Cluster (Cloud Deploy canary)"

A THICK CURVED ORANGE RIBBON weaves through these rows left to right,
with 10 numbered white circles (bold orange numbers) representing steps:

  1 (Row 1)  ML Engineer submits JobSet (NeMo SFT)
  2 (Row 2)  Kueue + DWS waits for 32× H100, admits when cohort ready
  3 (Row 3)  SFT runs 3 days, async checkpointing every 1000 steps
  4 (Row 4)  Final checkpoint v23 archived in GCS
  5 (Row 1)  ML Engineer triggers eval pipeline
  6 (Row 3)  Eval (small GPU pool): MMLU + HellaSwag + internal evals
  7 (Row 4)  v23 registered in Vertex AI Model Registry with scores
  8 (Row 1)  Manual approval gate — promote v23 to "prod-candidate"
  9 (Row 5)  Canary: v23 pod takes 5% traffic for 30 min soak
 10 (Row 5)  Blue/green rollout to 100% (Service label switch)

Below swim lanes, small legend: "Orange ribbon = promotion flow. White
numbered circles = ordered steps. Time progresses left to right
(~4 days end-to-end)."

Annotations off to the side at relevant steps:
  - Near step 2: small Kueue + DWS logos with "gang-scheduled, Flex-start"
  - Near step 3: small NVIDIA green GPU chip cluster
  - Near step 8: small lock icon "manual approval gate"
  - Near step 9: small chart "p95 latency + completion quality monitored"

Visual rules: orange ribbon dominant; numbered white circles standing
out; swim lanes are subtle background tints. No 3D, no neon. Sans-serif
labels, readable from a slide projection. Render each label exactly once.
```

---

## 7. Inference vs Training — side-by-side

| Dimension | Inference | Training |
|---|---|---|
| **Workload primitive** | `Deployment` + `HPA` + `PDB` | `LeaderWorkerSet` admitted by `Kueue` |
| **Scheduler** | Default K8s scheduler | Kueue (gang scheduling) + DWS Flex-start |
| **Scaling signal** | `vllm:num_requests_waiting` (HPA) | Queue admission (Kueue) — no in-job scaling |
| **Node SKU** | `a3-highgpu-8g` (8× H100, TP=8, single-node NVLink) | `a3-megagpu-8g` (8× H100, multi-node GPUDirect TCPXO) |
| **Cluster size** | Steady 2–4 nodes | 0 idle → bursts to 4–16 for a job |
| **Cluster autoscaling** | Mild (1× HPA factor) | Aggressive (0 ↔ N) via DWS |
| **Spot eligibility** | Burst only; baseline on-demand | Risky for long jobs unless checkpoint-aggressive |
| **Networking** | Single-node NVLink only | GPUDirect TCPXO 1.6 Tbps, Multus CNI, 8 secondary VPCs |
| **Storage I/O** | Read-mostly weights via GCS-FUSE + NVMe cache | Streaming dataset reads + parallel checkpoint writes (Filestore) |
| **Lifecycle** | 24/7 service, rolling/blue-green updates | Episodic jobs, days-to-weeks duration |
| **Failure recovery** | Pod restart + HPA backfill | Checkpoint restore + Kueue re-admit |
| **Observability focus** | Latency, throughput, queue depth, cost/1K-tokens | Loss, throughput, GPU util, comm efficiency |
| **CI/CD trigger** | Push to main → image build → Cloud Deploy | Manual job submission or scheduled (Vertex Pipelines / Cloud Composer) |
| **Cost predictability** | Predictable monthly baseline | Variable per-experiment |
| **Right-sizing question** | "How many replicas for our QPS?" | "How many GPU-hours for this experiment?" |

---

## 8. Cost shapes

| Workload | Driver | CUD strategy |
|---|---|---|
| **Inference** | Baseline replicas × hours × per-GPU rate | **3-yr CUD on baseline** (2 replicas × 8 H100 = 16 H100). Burst on on-demand or spot |
| **Training** | GPU-hours per experiment | **DWS Flex-start** for batch (cheaper than on-demand). 1-yr CUD only on steady minimum if running continuously. Avoid CUD for episodic work |

### Cost example for a realistic year

Assumptions: 2 inference replicas × 8 H100 with HPA to 4 replicas at peak; one SFT every 2 weeks (32× H100 × 3 days); one LoRA per week (8× H100 × 8 hours).

| Line item | Calc | Annual |
|---|---|---|
| Inference baseline (16 H100, 3-yr CUD ~55% off $11/hr) | 16 × 5 × 8760 | **~$700K** |
| Inference burst (avg 8 H100 extra × 200 hr/mo, spot ~$3) | 8 × 3 × 2400 | **~$58K** |
| Training SFT (32 H100 × 72 hr × 26 occurrences, DWS Flex ~$8/hr) | 32 × 8 × 1872 | **~$480K** |
| Training LoRA (8 H100 × 8 hr × 52 occurrences, on-demand) | 8 × 11 × 416 | **~$37K** |
| Storage (GCS + Filestore + Memorystore + Firestore) | | **~$30K** |
| Networking + observability + Apigee + Cloud Run | | **~$40K** |
| **Total platform** | | **~$1.35M / year** |

Inference dominates (~56%). Training is meaningful but bounded by experiment frequency. Storage and networking are rounding error.

---

## 9. Open decisions

1. **Single cluster vs separate clusters?** Recommend separate. Single-cluster with taints + Kueue is possible but the scheduling complexity and blast-radius coupling outweigh ~$220/month per control plane in savings.

2. **GKE Standard vs Autopilot for training?** Standard. Autopilot doesn't support compact placement, granular node pool config, or local NVMe SSD attachment — all critical for distributed training.

3. **Self-host on GKE vs Vertex AI Training?** Vertex AI Training handles a lot of this (DWS abstracted, GIB networking automated, checkpoint handling built-in). Tradeoff: less flexibility, locked into Vertex's job spec, ~10–20% premium per GPU-hour. **For most teams, Vertex AI Training is the right answer until you outgrow it.** Evaluate before committing to self-host.

4. **NeMo vs Megatron-LM directly vs HuggingFace?** NeMo for Nemotron. HuggingFace doesn't support the MoE+Mamba hybrid cleanly. Megatron-LM is what NeMo wraps — drop down only if NeMo doesn't expose a needed feature.

5. **Inference: Dense (TP=8 on a3-highgpu-8g) vs Distributed (TP=2 on a3-highgpu-2g)?** Dense for Nemotron (Mamba math makes per-replica capacity huge). Distributed for dense Transformers.

6. **Apigee X vs Cloud API Gateway vs IAP?** Use case decides:
   - Apigee X — B2B, multi-tenant, per-tenant SLA differentiation, billing tags
   - Cloud API Gateway — fewer than 10 tenants, simple quotas
   - IAP — internal tools, human OAuth/SSO

7. **Filestore High Scale vs Parallelstore for training checkpoints?** Filestore High Scale is sufficient for 120B SFT (under 80s for full state dump). Parallelstore only if Filestore proves to be the bottleneck (unlikely except for huge models or very frequent checkpointing).

8. **RLHF location: training cluster, separate cluster, or production inference cluster for rollouts?** RLHF combines training with rollout inference. Three options with different tradeoffs (interference risk vs cost). Decide based on RLHF cadence and inference SLO.

---

## Appendix — Decision quick reference

| If you need to... | Use |
|---|---|
| Deploy the model for production serving | Inference cluster, Section 3, dense path (8× H100 TP=8) |
| Fine-tune Nemotron with your data (LoRA) | Training cluster, single-node Job, NeMo PEFT |
| Full SFT or DPO alignment | Training cluster, multi-node LWS + Kueue + DWS, NeMo |
| Pre-train from scratch | Evaluate Vertex AI Training first |
| Add a new tenant with custom rate limits | Apigee X (already in inference path) |
| Test the architecture without GPU quota | Mock profile in nemotron-platform-gcp (Helm values-mock.yaml) |
| Promote a new model to production | Eval pipeline → Vertex AI Model Registry → Cloud Deploy canary → blue/green |

---

*Document generated synthesizing the original `architecture-2000-users.md` (inference focus) and Gemini Spark's training-focused architecture, with corrections for Nemotron-3's Mamba-aware sizing, DWS Flex-start scheduling, Multi-NIC GPUDirect TCPXO networking, and NeMo framework recommendation.*
