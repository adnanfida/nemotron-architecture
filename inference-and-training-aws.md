# Architecture: Nemotron-3 Super 120B on AWS EKS — Inference and Training

**Status:** Design proposal
**Last updated:** 2026-05-22
**Audience:** Platform engineering, ML platform, FinOps
**Region target:** single region, multi-AZ (e.g., us-east-1 with a, b, c)

This document mirrors [`inference-and-training.md`](inference-and-training.md) (the GCP version) for AWS EKS. Same two workloads, same model, same target scale (2,000 concurrent users for inference; 32–128 GPU training jobs), same architectural principles. AWS-specific service mappings throughout. A side-by-side comparison appendix maps GCP → AWS components.

The two workloads share an AWS account, VPC, IAM model (IRSA), observability stack, and S3-based model artifact plane. They run on **separate EKS clusters** because their node SKUs, schedulers, networking, and lifecycles are fundamentally different.

The seam between training and inference is the **SageMaker Model Registry**: training writes versioned checkpoints to S3 and registers them; inference consumes approved checkpoints via a promotion pipeline (Argo Rollouts).

---

## 1. Why Nemotron-3 changes the sizing math

This is the same on AWS as GCP — the model architecture (hybrid Mamba-2 + select attention layers + MoE with 12B active) is cloud-agnostic. Key consequences:

- **Mamba-2 SSM cache is fixed-size**, not context-length-dependent. Per-replica concurrent capacity is enormous compared to a dense 120B Transformer.
- **MoE with 12B active params**: all 120B must be in VRAM (experts can't swap without destroying latency), but effective compute is ~12B-dense per token.
- **Native NVFP4 (4-bit)**: weights ~60GB instead of ~240GB at BF16.

**Concrete consequence for AWS sizing:** at 200 concurrent streams × 3,000-token context in FP8 KV, total runtime cache is ~15GB. A single `p5.48xlarge` node (8× H100 80GB, 640GB total VRAM) holds 120GB of weights distributed across 8 GPUs and has ~440GB of cache headroom. **One node handles 1000+ concurrent streams** before hitting memory; compute is the bottleneck long before VRAM.

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
where $B$ = batch (concurrent streams), $C$ = context length, $L_{attn}$ = number of attention layers (~16 in Nemotron-3), $H_{kv}$ = KV heads (2), $D_{head}$ = head dim (128), $P$ = precision bytes.

At $B$=200, $C$=3000, FP8 KV: $M_{KV} ≈ 9.83$ GB.

SSM cache for Mamba-2 layers (does NOT scale with $C$):
$M_{SSM} = B × L_{mamba} × 2 × D_{model} × D_{state} × P$ — fixed size per batch, independent of context length.

Cumulative runtime cache (KV + SSM + framework overhead) at 200 concurrent ≈ 15 GB, vs ~440 GB per-node headroom after weights → ~30× headroom. Compute, not memory, bounds concurrency on Nemotron-3.

---

## 2. Shared AWS account foundation

| Component | Purpose | Shared by |
|---|---|---|
| **AWS account** `nemotron-platform` (or sub-account in an Org) | One billing/audit boundary | Both |
| **Shared VPC** (us-east-1, 3 AZs) | One network, NAT Gateway for egress | Both |
| **VPC Endpoints (PrivateLink)** | Private access to S3, ECR, KMS, Secrets Manager, etc. without traversing the internet | Both |
| **AWS Secrets Manager** | NGC API key, HF token | Both |
| **AWS KMS** | Customer-managed keys (CMKs) for S3, EBS, FSx, Secrets Manager | Both |
| **S3 `nemotron-weights`** (with optional Cross-Region Replication) | Source of truth for model weights. Training writes; inference reads | Both (write/read separated by bucket policy + IAM) |
| **ECR (Elastic Container Registry)** | NIM containers + training framework images | Both |
| **SageMaker Model Registry** | Versioned checkpoints with eval metadata — the promotion gate | Both |
| **Amazon Managed Service for Prometheus (AMP)** | Metrics federation across both clusters | Both |
| **Amazon Managed Grafana (AMG)** | Dashboards (alternative or complement to CloudWatch) | Both |
| **CloudWatch Logs + Kinesis Firehose → S3 → Athena** | Long-term log analysis (Athena = AWS equivalent of BigQuery on S3) | Both |
| **AWS X-Ray** | Distributed tracing | Both |

### [Figure 1] Nano-banana prompt — Shared AWS foundation

<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/ea237d8b-5244-43f2-a599-d21bef5d74cf" />

---

## 3. Inference architecture

**Sized for:** 2,000 active users → ~200 concurrent streams at 10% peak concurrency, TTFT < 500ms, inter-token latency < 25ms, > 99.9% monthly availability.

### Cluster shape

**Node SKU choice — recommended path is dense:**

| Path | Per-replica | Replicas for 200 concurrent | Total GPUs | Notes |
|---|---|---|---|---|
| **Dense** (recommended for this model) | `p5.48xlarge`, 8× H100 80GB, TP=8 | 2–3 (HA across AZs) | 16–24 | Single-node NVLink only. Mamba math = each replica handles 1000+ concurrent. |
| **Distributed** | smaller GPU instances + TP=2 | many replicas | similar GPU count | No equivalent of `a3-highgpu-2g` on AWS. The two-GPU AWS option is `g5.12xlarge` (4× A10G, weaker GPU); not viable for 120B-class. |

**Stick with the dense path on AWS.** AWS doesn't offer a 2-GPU H100 instance — the H100 SKUs jump from "none" to "p5.48xlarge (8× H100)." This is fine because Mamba's per-replica capacity is huge.

**p5.48xlarge full specs (for reference):**

| Component | Spec |
|---|---|
| GPUs | 8× NVIDIA H100 80GB SXM5 |
| GPU interconnect | NVLink + 3rd-gen NVSwitch, 3.6 TB/s bidirectional per GPU |
| vCPUs | 192 (AMD EPYC 7R13) |
| System memory | 2,048 GiB |
| Instance-store NVMe | 30.4 TB (8× 3.84 TB local SSD) |
| EFA cards | 32 |
| EFA aggregate bandwidth | 3,200 Gbps |
| EBS bandwidth | 80 Gbps |

### Edge → routing → inference

```
Client → Route 53 → AWS WAF + Shield → CloudFront → ALB →
API Gateway (or direct ALB) → Model Gateway (ECS Fargate × 3) →
EKS Service (ALB target type IP) → NIM pods (8× H100)
```

**Each layer's job:**

- **Route 53 + AWS WAF + Shield Advanced** — DNS, WAF rules (OWASP top-10), L7 DDoS protection. Shield Advanced (paid) gives 24/7 response team for major attacks.
- **CloudFront** — global CDN/edge with TLS termination, HTTP/2 + HTTP/3. Origin is the ALB. CloudFront also fronts API Gateway and reduces origin load.
- **ALB (Application Load Balancer)** — regional L7 load balancer. Terminates TLS from CloudFront, routes by hostname/path. Integrates with EKS via the AWS Load Balancer Controller using **target type: IP** (analog to GCP's NEG — bypasses kube-proxy, targets pod IPs directly).
- **API Gateway (REST or HTTP API)** — per-tenant API keys, usage plans, throttle limits, request logging to CloudWatch. *Caveat: API Gateway HTTP APIs have a 30-second timeout, REST APIs have 29 seconds. SSE streaming requires the Cloud Gateway path below, NOT API Gateway, OR you need an ALB-direct path with a sticky longer timeout.*
- **AWS IAM Identity Center (formerly SSO) + ALB OIDC** — alternate auth path for human users on internal tools. Equivalent to GCP IAP.
- **Model Gateway on ECS Fargate (3 always-on tasks)** — SSE streaming proxy, session-affinity routing via ElastiCache, prompt shaping, token-count cost emit to Kinesis Firehose → S3 → Athena, cancellation propagation. Fargate is the right runtime here (not Lambda — Lambda has a 15-minute timeout and doesn't do persistent SSE cleanly; not EC2 — too much undifferentiated heavy lifting).
- **EKS Service exposed via ALB target type IP** — gateway targets pod IPs directly
- **NIM pods** — `nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b`, TP=8 on a single p5.48xlarge node

### Inference workload spec

| Setting | Value | Why |
|---|---|---|
| Deployment replicas (baseline) | 2 (one per AZ across 2 of 3) | Bare minimum HA |
| Deployment replicas (with HPA headroom) | 3–4 | N+1 redundancy across all 3 AZs |
| HPA metric | `vllm:num_requests_waiting` queue depth via AMP + adapter | CPU-based HPA never triggers on GPU pods |
| Target | queue depth < 4 per replica | |
| PDB | `minAvailable: 2` | Tolerates rolling updates without HA floor breach |
| Topology spread | 3-AZ, maxSkew=1 via `topologySpreadConstraints` | At least one replica per AZ |
| Weights mount | **Mountpoint for S3 CSI driver + instance-store NVMe** (30.4 TB local SSD per p5.48xlarge) | Warm replicas don't re-read from S3 on restart |
| Probes | startupProbe 30-min grace, readiness, liveness | Weight load takes 5–15 min on first boot |
| Service | ClusterIP + ALB target type IP annotation | Container-native LB targeting |
| Session affinity | ALB sticky cookie OR header-based via gateway | KV-cache prefix reuse on attention layers |
| Pod security context | `capabilities: add: ["IPC_LOCK"]` | Required to lock memory pages for EFA RDMA |
| Container image alternative | AWS Deep Learning Container (`763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:latest`) | Alternative to pulling NIM from nvcr.io directly |

### Inference cluster topology details

- **Cluster:** EKS with a private API endpoint (or public + authorized CIDRs), 3 AZs, KMS envelope encryption for secrets, IRSA enabled
- **Node management:** **Karpenter** (not the older Cluster Autoscaler) — better diversity handling, scale-to-zero, spot interruption awareness
- **Node groups (via Karpenter NodePool CRDs):**
  - `system-pool`: M6i / M7i medium instances for system pods (CoreDNS, AWS LB controller, ExternalDNS, AMP collector, Karpenter itself)
  - `gpu-pool`: `p5.48xlarge`, Karpenter min=0 → max=4 — 0 baseline so idle dev clusters don't bill
- **CNI:** **AWS VPC CNI** with **prefix delegation** enabled (each pod gets a routable VPC IP; without prefix delegation, max pods per node is limited)
- **Networking:** Private subnets for nodes, NAT Gateway in public subnet for egress (image pulls before ECR cache), VPC Endpoints for S3 / ECR / KMS / Secrets Manager / STS to avoid NAT data charges

### State plane (production app concerns)

| Component | Purpose | Sizing |
|---|---|---|
| **ElastiCache for Redis** | Session-affinity hints, per-tenant rate counters, idempotency keys | `cache.r7g.large` × 2 nodes (Multi-AZ with replica) |
| **DynamoDB** | Durable conversation history per tenant | Per-tenant partition key; on-demand pricing or provisioned with autoscaling |
| **Kinesis Firehose → S3 → Athena (or Redshift Serverless)** | Request logs + token usage for cost analytics | Athena = serverless SQL on S3, BigQuery equivalent |
| **ECR (private)** | Intra-region cached copy of NIM container | Critical for fast pod startup on scale-up |

### Observability

- **Amazon Managed Prometheus (AMP)** scrapes `vllm:*` and DCGM-exporter (GPU util, HBM, NVLink throughput, temperature). EKS pods use the **AWS Distro for OpenTelemetry (ADOT) Collector** or **Prometheus Operator + remote_write** to ship metrics
- **CloudWatch Dashboards + Amazon Managed Grafana (AMG)** — queue depth, GPU utilization, KV-cache fill, p50/p95/p99 first-token and inter-token latency, cost per 1K tokens (joined with Athena)
- **AWS X-Ray** — request traces from gateway through pod with token counts in span attributes
- **CloudWatch Container Insights** — node and pod metrics, complement to AMP
- **SLOs:** p95 first token < 500ms, p99 inter-token < 25ms, monthly availability > 99.9% (defined via CloudWatch Synthetics canaries + SLO API)
- **Alerts:** queue-depth-high via AMP alerting → SNS → PagerDuty; pod restart rate via CloudWatch alarm; GPU OOM via DCGM; daily cost overrun via AWS Budgets

### [Figure 2] Inference architecture on AWS

<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/98508406-9ee1-4311-a2db-b2ac30cf97cb" />

---

## 4. Training architecture

### Workload types and recommended patterns

| Workload | Compute | Duration | Recommended pattern |
|---|---|---|---|
| LoRA / QLoRA / adapter fine-tuning | 1× `p5.48xlarge` (8 H100) | Hours | Single-node K8s `Job`, NeMo PEFT or HF TRL |
| Full SFT (Supervised Fine-Tuning) | 4× `p5.48xlarge` (32 H100) | 1–3 days | Multi-node `LeaderWorkerSet` + Kueue, NeMo |
| DPO / RLHF alignment | 4–16× `p5.48xlarge` (32–128 H100) | 3–7 days | Multi-node LWS + Kueue, NeMo-Aligner |
| Continued pre-training | 16–64× `p5.48xlarge` (128–512 H100) | Weeks | Multi-node LWS + Kueue + Capacity Blocks |
| Full pre-training from scratch | 1000+ H100 | Months | **Consider SageMaker HyperPod instead** |

### Training memory math

Identical to GCP doc (the model and AdamW optimizer math don't care which cloud). 1.92 TB static state for full-parameter BF16 training. 4× `p5.48xlarge` = 32× H100 = 2,560 GB cluster VRAM → ~60 GB per GPU after FSDP/ZeRO-3 sharding → ~20 GB headroom for activations.

### Cluster topology

- **Cluster:** EKS, private nodes, IRSA, separate from inference cluster
- **Node management:** Karpenter (better than Cluster Autoscaler for bursty GPU workloads)
- **Node groups (via Karpenter NodePool):**
  - `system-pool`: M-series instances for Kueue controller, JobSet API, custom-metrics adapter
  - `gpu-pool`: `p5.48xlarge` with EFA enabled, **EC2 Capacity Block reservation** OR on-demand. Karpenter scales 0 → 16 nodes
  - Optional `gpu-pool-spot`: same SKU with `--capacity-type=spot`, for burst with aggressive checkpointing
- **Placement:** **EC2 Cluster Placement Group** on the GPU pool — all nodes for one job on the same rack, required for EFA peak bandwidth. UltraCluster placement (a managed alternative) places nodes in a preconfigured high-bandwidth rack
- **EFA driver** installed via DaemonSet (`aws-efa-k8s-device-plugin`)

### Orchestration stack

| Layer | Tool | Purpose |
|---|---|---|
| **AWS GPU reservation** | **EC2 Capacity Blocks for ML** | Reserve P5 capacity for 1–14 days at a discount vs on-demand. Atomic cohort allocation — all nodes available at start time. Direct equivalent of GCP DWS Flex-start |
| **K8s job queueing** | **Kueue** | Same as GCP — fair sharing across teams, priority preemption, gang scheduling |
| **Multi-job topology** | **JobSet API** (sig-apps) | Same as GCP |
| **Multi-node single-replica** | **LeaderWorkerSet (LWS)** | Same as GCP |
| **Single-node single-replica** | K8s `Job` | LoRA, eval, data prep |
| **Alternative: managed batch** | **AWS Batch with EKS as execution environment** | Older pattern; Kueue is now preferred for K8s-native scheduling |
| **Alternative: managed training** | **SageMaker HyperPod** | Managed alternative to running training on EKS. Handles cluster reconfig, fault detection (auto-replaces failed nodes), resilience layer for long jobs. Evaluate this seriously before going DIY. |

### Framework choice — NeMo for Nemotron specifically

Same recommendation as GCP doc: **NVIDIA NeMo Framework + NeMo-Aligner**. First-party support for Nemotron's MoE + Mamba-2 hybrid. HuggingFace + DeepSpeed/FSDP works for LoRA on attention layers but doesn't handle Mamba layers cleanly for full-parameter training.

### Networking — EFA (Elastic Fabric Adapter)

This is where AWS is **operationally much simpler than GCP**. GCP requires Multus CNI + 8 secondary VPCs for the multi-NIC GPUDirect TCPXO setup. AWS uses a single ENI per node with EFA hardware acceleration.

| SKU | EFA | Bandwidth | Use |
|---|---|---|---|
| `p5.48xlarge` | **32 EFA cards** | **3,200 Gbps** aggregate | Standard for 120B-class training |
| `p5e.48xlarge` | 32 EFA cards | 3,200 Gbps | H200 141GB variant (more VRAM headroom) |
| `p5en.48xlarge` | 32 EFA cards | Same | Enhanced networking variant (additional EBS bandwidth) |
| `p4d.24xlarge` | 4 EFA cards | 400 Gbps | A100 generation; only if H100 capacity is unavailable |

**EFA + SRD (Scalable Reliable Datagram)** lets the GPU write packets directly to the EFA device, bypassing host kernel — like RDMA over standard networking. NCCL has first-class EFA support; configure via `FI_PROVIDER=efa` and the rest is automatic.

**UltraClusters** are AWS's pre-configured P5 racks with all-to-all 3,200 Gbps EFA. Requesting capacity via Capacity Blocks routinely lands you in an UltraCluster automatically.

**Pod-to-NIC binding pattern:**

- Each p5.48xlarge has **1 primary ENI** (for cluster traffic) + **32 EFA-only network interfaces** (for GPU comms)
- The **AWS EFA Kubernetes Device Plugin** exposes EFA cards as the `vpc.amazonaws.com/efa` extended resource
- Pods request EFA via `resources.limits.vpc.amazonaws.com/efa: "32"` in the pod spec
- Pods also need `securityContext.capabilities.add: ["IPC_LOCK"]` to lock memory pages
- **Multus CNI** is used when a pod needs multiple EFA interfaces explicitly assigned (e.g., multi-pod-per-node patterns); for the standard one-training-pod-per-node case, the EFA Device Plugin alone handles it

Compared to GCP's GPUDirect TCPXO setup (which requires Multus + 8 secondary VPCs with per-GPU subnet bindings regardless of pod density), AWS is simpler: standard VPC subnets, EFA handled as a device-plugin resource. Multus only enters when you need fine-grained per-pod NIC assignment.

### [Figure 3] Training cluster topology on AWS

<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/d82f3ffd-d74e-4c33-960d-fb7a9d5caeb6" />


### [Figure 4] EFA networking topology (much simpler than GCP)

<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/087864c1-adf6-40b0-a6a1-63a422d7795f" />

---

## 5. Storage hierarchy

Different model than GCP but similar tiering principle. **The killer feature: FSx for Lustre + GPUDirect Storage (GDS) over EFA** — added in Nov 2024 and a major performance advantage over the GCP path.

| Tier | Service | Throughput | Cost | Used for |
|---|---|---|---|---|
| **Cold archive** | Amazon S3 (Standard or Intelligent-Tiering) | Moderate | Cheapest | Source datasets, checkpoint archive, model weights distribution |
| **Hot read cache** | **Mountpoint for S3 CSI driver + instance-store NVMe** (30.4 TB per p5.48xlarge) | High | Near-free | Inference weight cache, training dataset shards (read-mostly) |
| **Hot read/write** | **Amazon FSx for Lustre (Scratch SSD) + EFA + GDS, with S3 linkage** | **Up to 1,200 Gbps per client (~150 GB/s)** | Medium-high | Multi-node training checkpoint writes; dataset hot tier |
| **Hot read/write extreme** | FSx for Lustre Persistent SSD (largest sizes) | Even higher aggregate | Expensive | Only for the largest jobs |

### Two killer features layered together

**1. FSx for Lustre + S3 linkage** — files in S3 appear as lazy-loaded entries in the FSx filesystem. First read pulls from S3 into FSx; subsequent reads hit FSx at full Lustre throughput. Writes flush back to S3 periodically or on-demand. Datasets effectively live in S3 while training I/O happens at Lustre speeds.

**2. FSx for Lustre + EFA + NVIDIA GPUDirect Storage (GDS)** — announced Nov 2024. GDS establishes a direct data path between the parallel FSx filesystem and H100 GPU memory, **bypassing host CPU and system RAM**. Delivers up to **1,200 Gbps per client instance** (~150 GB/s). For a p5.48xlarge with 32 EFA cards, this lights up the full network fabric for storage I/O.

**Performance implication:** a 1.92 TB full training state dump completes in ~13 seconds at 1.2 Tbps. Async checkpointing keeps GPU utilization above 95% even on frequent checkpoint cadence. This is a meaningful advantage over the GCP Filestore High Scale path (25 GB/s write, ~80 seconds for the same dump).

**Inference uses:** S3 for weights + instance-store NVMe cache via Mountpoint for S3 CSI driver. Each p5.48xlarge has 30.4 TB local NVMe, plenty to cache the ~60GB weight set with massive room to spare for ephemeral state. GDS-on-FSx isn't needed for inference (read-mostly, smaller volumes).

**Training uses:** S3 for dataset shards (lazy-loaded into FSx Lustre via linkage), FSx for Lustre Scratch with EFA+GDS for active checkpoint writes and dataset hot tier.

### [Figure 5] AWS storage hierarchy

<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/fa0d0bc8-62c3-46d0-a3f0-a700ca92e622" />


---

## 6. The seam: how training feeds inference

The SageMaker Model Registry is the source of truth for "which checkpoint is production-ready." Promotion is gated by eval + manual approval; rollout uses **Argo Rollouts** for K8s-native progressive delivery (AWS CodeDeploy doesn't speak Kubernetes natively).

### Promotion flow

1. ML Engineer submits NeMo SFT JobSet to training cluster
2. Kueue + EC2 Capacity Blocks reservation waits for 32× H100, admits when cohort is ready
3. Training runs ~3 days with async checkpointing every 1000 steps to FSx Lustre → flushed to S3 via linkage
4. Final checkpoint `v23` archived in S3
5. ML Engineer triggers eval pipeline (separate small GPU pool, runs MMLU + HellaSwag + internal evals)
6. Checkpoint `v23` registered in SageMaker Model Registry with eval scores
7. ML Engineer reviews scores; promotes `v23` to `Approved` status
8. EventBridge rule on registry change → triggers Argo Rollouts on the inference cluster
9. Argo Rollouts: canary pod with `v23` takes 5% traffic via ALB weighted target groups for 30-min soak
10. Blue/green rollout to 100% via target group swap; previous `v22` kept warm for instant rollback

### [Figure 6] Promotion lifecycle on AWS

<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/6692b0f8-1e68-40dc-a99a-d9e2d39ecda2" />

---

## 7. Inference vs Training — side-by-side

| Dimension | Inference | Training |
|---|---|---|
| **Workload primitive** | `Deployment` + `HPA` + `PDB` | `LeaderWorkerSet` admitted by `Kueue` |
| **Node scheduler** | Karpenter | Karpenter + EC2 Capacity Blocks for ML |
| **Scaling signal** | `vllm:num_requests_waiting` via AMP custom-metrics adapter | Queue admission (Kueue), no in-job autoscaling |
| **Node SKU** | `p5.48xlarge` (8× H100, TP=8, single-node NVLink) | `p5.48xlarge` (8× H100, multi-node EFA UltraCluster) |
| **Cluster size** | Steady 2–4 nodes | 0 idle → bursts to 4–16 for a job |
| **Cluster autoscaling** | Mild (HPA-driven, Karpenter provisions) | Aggressive (0 ↔ N) via Capacity Block reservation + Karpenter |
| **Spot eligibility** | Burst only; baseline on Savings Plans | Risky for long jobs without aggressive checkpointing |
| **Networking** | Single-node NVLink only | EFA + SRD 3,200 Gbps cross-node; Cluster Placement Group |
| **Storage I/O** | Mountpoint for S3 + instance-store NVMe | FSx for Lustre Scratch (with S3 linkage) for hot path |
| **Lifecycle** | 24/7 service, blue/green via Argo Rollouts | Episodic jobs, days-to-weeks duration |
| **Failure recovery** | Pod restart + HPA backfill + Karpenter | Checkpoint restore + Kueue re-admit; SageMaker HyperPod auto-replaces failed nodes |
| **Observability focus** | Latency, throughput, queue depth, cost/1K-tokens | Loss, throughput, GPU util, comm efficiency |
| **CI/CD trigger** | Push to main → CodeBuild image build → Argo Rollouts | Manual job submission or scheduled (EventBridge → SageMaker Pipelines / Airflow on MWAA) |
| **Cost predictability** | Predictable monthly baseline | Variable per-experiment |
| **Right-sizing question** | "How many replicas for our QPS?" | "How many GPU-hours for this experiment?" |

---

## 8. Cost shapes

| Workload | Driver | AWS discount strategy |
|---|---|---|
| **Inference** | Baseline replicas × hours × per-GPU rate | **3-yr Compute Savings Plan** or **EC2 Instance Savings Plan** on baseline (2 replicas × 8 H100 = 16 H100). 1-yr SP for less commitment risk. Burst on on-demand or spot |
| **Training** | GPU-hours per experiment | **EC2 Capacity Blocks for ML** for predictable batch (cheaper than on-demand, gang allocation). 1-yr Savings Plan only on steady minimum if running continuously. Avoid SP for episodic work. **SageMaker HyperPod** for resilience-critical long jobs (managed alternative). |

### Realistic annual cost example

Assumptions: 2 inference replicas × 8 H100 with HPA to 4 replicas at peak; one SFT every 2 weeks (32× H100 × 3 days); one LoRA per week (8× H100 × 8 hours).

| Line item | Calc | Annual |
|---|---|---|
| Inference baseline (16 H100, 3-yr Compute SP ~50% off ~$98/hr for p5.48xlarge with 8 H100 ÷ 8 = ~$12.30/hr/GPU) | 16 × 6.15 × 8760 | **~$862K** |
| Inference burst (avg 8 H100 extra × 200 hr/mo, spot ~$4) | 8 × 4 × 2400 | **~$77K** |
| Training SFT (32 H100 × 72 hr × 26 occurrences, Capacity Block ~$9/hr/GPU) | 32 × 9 × 1872 | **~$540K** |
| Training LoRA (8 H100 × 8 hr × 52, on-demand) | 8 × 12.30 × 416 | **~$41K** |
| Storage (S3 + FSx Lustre + ElastiCache + DynamoDB) | | **~$45K** |
| Networking (NAT GW, VPC endpoints, ALB, CloudFront, egress) + observability + API GW + Fargate | | **~$60K** |
| **Total platform** | | **~$1.625M / year** |

Note: AWS P5 list pricing is roughly comparable to GCP a3-highgpu-8g on-demand, but the realized cost depends heavily on Savings Plan term, Capacity Block vs on-demand mix, and Spot interruption tolerance. Run actual quotes with AWS Pricing Calculator before committing.

---

## 9. Open decisions

1. **Single cluster vs separate clusters?** Recommend separate (same reasoning as GCP). Single-cluster with taints + Kueue is possible but scheduling complexity and blast-radius coupling outweigh the savings.

2. **EKS managed node groups vs Karpenter?** **Karpenter** for both clusters. Better for diverse instance types, spot interruption, scale-to-zero. The older managed node groups + Cluster Autoscaler model is fading.

3. **Self-host training on EKS vs SageMaker HyperPod vs SageMaker Training?**
   - **HyperPod** — managed cluster with auto-recovery for failed nodes during long training runs. Best for multi-week jobs where node failures would otherwise abort training. Tradeoff: less flexibility than DIY EKS, premium pricing.
   - **SageMaker Training** — fully managed (no cluster to manage), good for shorter jobs. Limited to specific frameworks SageMaker supports.
   - **DIY on EKS** — most flexibility, most operational burden.
   - **For most teams, evaluate HyperPod first** for long jobs; use DIY EKS for shorter/experimental work.

4. **NeMo vs Megatron-LM vs HuggingFace?** NeMo for Nemotron (same as GCP doc). HuggingFace + Accelerate works for LoRA but not the MoE + Mamba hybrid in full-parameter training.

5. **API Gateway vs ALB-direct for SSE?** **ALB-direct** for the streaming path (no 29-second timeout limitation). API Gateway is fine for non-streaming endpoints (e.g., model metadata, admin APIs). The ECS Fargate gateway sits in front of EKS regardless.

5a. **Human-user auth: IAM Identity Center vs Cognito vs AWS Verified Access?** Use case decides:
    - **IAM Identity Center** (formerly SSO) — internal tools, employee OAuth via your corporate IdP, ALB OIDC integration
    - **Amazon Cognito** — B2C user signup, social login, user pools, hosted UI
    - **AWS Verified Access** — zero-trust application access without VPN, integrates with IdP + device trust providers; newest of the three, best for sensitive internal apps

5b. **Kueue vs AWS Batch for training queue?** **Kueue** for new builds — K8s-native, more flexibility, matches the GCP path for cross-cloud consistency. **AWS Batch** if your org already runs Batch and wants unified queue management across EKS + EC2; it can use EKS as an execution environment.

6. **ElastiCache (Redis) vs DynamoDB for sessions?** ElastiCache for hot session-affinity hints (sub-millisecond, in-memory). DynamoDB for durable conversation history. Use both, different jobs.

7. **Argo Rollouts vs AWS CodeDeploy?** Argo Rollouts for the inference cluster — K8s-native, supports canary with weighted ALB target groups, automatic rollback on metric thresholds. CodeDeploy is fine for Fargate/Lambda but not the EKS path.

8. **FSx for Lustre Scratch vs Persistent?** Scratch is much cheaper and faster but data is lost when the filesystem is deleted. Persistent if you need the same hot tier across multiple jobs. **Scratch for most cases** with S3 linkage providing durability.

9. **VPC Service Controls equivalent?** AWS has no direct VPC-SC analog. Closest pattern: combination of VPC Endpoints with endpoint policies + S3 bucket policies + KMS key policies + AWS Organizations SCPs. More fragmented than GCP's perimeter model; document carefully.

10. **Where to host the Model Gateway?** Three options:
    - **ECS Fargate** (recommended) — serverless containers, autoscales, supports SSE
    - **Lambda** — 15-min timeout makes SSE awkward; only viable for non-streaming
    - **EKS pods** — coupling concern (gateway and inference share blast radius); not recommended

---

## Appendix A — Decision quick reference

| If you need to... | Use |
|---|---|
| Deploy the model for production serving | Inference EKS cluster, Section 3, dense path (8× H100 TP=8 on `p5.48xlarge`) |
| Fine-tune Nemotron with your data (LoRA) | Training EKS cluster, single-node Job, NeMo PEFT |
| Full SFT or DPO alignment | Training EKS cluster, multi-node LWS + Kueue + Capacity Blocks, NeMo |
| Pre-train from scratch (multi-week) | **SageMaker HyperPod** (resilience matters at scale) |
| Add a new tenant with custom rate limits | API Gateway usage plans + the Fargate gateway |
| Test the architecture without GPU quota | EKS mock profile (Helm `values-mock.yaml` adapted) running on M-series instances |
| Promote a new model to production | Eval pipeline → SageMaker Model Registry → EventBridge → Argo Rollouts canary → blue/green |

---

## Appendix B — GCP → AWS component mapping

Quick reference for engineers translating between clouds.

| Concept | GCP | AWS |
|---|---|---|
| Project / account | GCP Project | AWS Account |
| Network | VPC + subnet per region (auto multi-zone) | VPC + 3 AZ subnets (manual) |
| Kubernetes | GKE Standard | EKS |
| Node autoscaler | Cluster Autoscaler | **Karpenter** (modern) |
| H100 instance | `a3-highgpu-8g` (8× H100, TP=8) | `p5.48xlarge` (8× H100, TP=8) |
| H100 multi-node | `a3-megagpu-8g` (GPUDirect TCPXO) | `p5.48xlarge` with EFA |
| GPU cohort scheduling | DWS Flex-start | EC2 Capacity Blocks for ML |
| Multi-node networking | Multus CNI + 8 secondary VPCs | EFA (single ENI, simpler) |
| Compact placement | Compact Placement Policy | Cluster Placement Group / UltraCluster |
| Workload Identity (K8s SA ↔ cloud SA) | Workload Identity (KSA → GSA) | IRSA (KSA annotated with IAM role) |
| Object storage | GCS | S3 |
| Object storage CSI | GCS-FUSE CSI driver | Mountpoint for S3 CSI driver |
| HPC filesystem | Filestore High Scale / Parallelstore | FSx for Lustre (with S3 linkage + EFA + GDS) |
| Container registry | Artifact Registry | ECR |
| Secrets | Secret Manager | AWS Secrets Manager |
| Key management | Cloud KMS | AWS KMS |
| Managed Prometheus | Managed Service for Prometheus | Amazon Managed Service for Prometheus (AMP) |
| Dashboards | Cloud Monitoring | CloudWatch + Amazon Managed Grafana |
| Logs | Cloud Logging → BigQuery | CloudWatch Logs → S3 → Athena |
| Tracing | Cloud Trace | AWS X-Ray |
| Model registry | Vertex AI Model Registry | SageMaker Model Registry |
| Managed training | Vertex AI Training | SageMaker Training / **SageMaker HyperPod** |
| Edge / CDN | Global HTTPS LB (anycast) | CloudFront |
| WAF / DDoS | Cloud Armor | AWS WAF + AWS Shield |
| API management | Apigee X | Amazon API Gateway |
| Human-auth gateway | Identity-Aware Proxy (IAP) | IAM Identity Center + ALB OIDC |
| Serverless containers (Model Gateway) | Cloud Run | **ECS Fargate** |
| Session cache | Memorystore (Redis) | ElastiCache for Redis |
| Document DB | Firestore | DynamoDB |
| Serverless analytics SQL | BigQuery | Athena (on S3) |
| Progressive rollout | Cloud Deploy | **Argo Rollouts** (K8s-native) |
| DNS | Cloud DNS | Route 53 |
| L7 LB | Global HTTPS LB | Application Load Balancer (ALB) |
| K8s service to LB binding | Network Endpoint Group (NEG) | ALB target type IP |
| Private service access | Private Service Connect (PSC) | VPC Endpoints (PrivateLink) |
| Cross-region replication | Multi-regional bucket | S3 Cross-Region Replication |
| Cost commitment | 1-yr or 3-yr CUD | Compute / EC2 Savings Plan |
| Burst pricing | Spot VMs | EC2 Spot Instances |

---

## Appendix C — What's NOT in this design

Documented as future increments, same as the GCP doc:

- Multi-region active-active (trigger: 10,000+ concurrent users or strict cross-continent latency SLA)
- Disaggregated prefill/decode (trigger: > 1,000 RPS sustained)
- Per-tenant LoRA variants (trigger: enterprise tenants demanding fine-tuned model behavior)
- Bedrock comparison — AWS Bedrock offers Nemotron as a fully managed endpoint; TCO comparison worth running before committing to self-hosted on EKS at this scale

---

*This document mirrors `inference-and-training.md` (GCP version) for AWS. The Nemotron-specific sizing math (Mamba SSM cache, MoE 12B active params, NVFP4 precision) is identical across clouds. Operational details differ — see Appendix B for component mappings.*
