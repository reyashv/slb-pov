# SLB PoC — TrueFoundry Model Serving & Fine-Tuning

This repository demonstrates TrueFoundry's capabilities for deploying, optimizing, and fine-tuning seismic foundation models at scale. It is organized into 4 modules, each independently deployable on TrueFoundry.

---

## Repository Structure

```
slb-pov/
├── encoders/
│   ├── sfm/                       # SFM encoder server, fine-tuning, batch inference, TRT optimization
│   │   ├── sfm_server.py          # FastAPI inference server (single + batch)
│   │   ├── sfm_server_trt.py      # PyTriton TRT inference server
│   │   ├── finetune.py            # Fine-tuning with checkpoint save/resume
│   │   ├── batch_infer.py         # Sustained batch inference with progress checkpointing
│   │   ├── preprocess_dataset.py  # One-time dataset preprocessing → ML Repo artifact
│   │   ├── convert_trt.py         # TensorRT FP16 conversion job
│   │   └── requirements.txt
│   └── vjepa/                     # V-JEPA encoder server and probe training
│       ├── vjepa_encoder_server.py
│       ├── probe_train.py
│       ├── probe_train_segment.py
│       ├── joint_finetune.py
│       └── requirements.txt
├── decoder/                       # Decoder heads + unified inference service
│   ├── server.py                  # Decoupled decoder (calls encoder over internal network)
│   ├── fused_server.py            # Fused encoder+decoder in single pod
│   ├── inference_service.py       # Unified entry point — fans out to all decoders
│   ├── heads/
│   │   ├── classify.py
│   │   ├── detect.py
│   │   └── segment.py
│   └── requirements.txt
├── locust_service/                # Load testing service
│   ├── locust_benchmark.py        # Supports FastAPI and PyTriton, configurable batch size
│   └── requirements.txt
└── yaml/                          # Optional YAML configs for tfy apply
```

---

## Modules

### 1. Encoder Serving (`encoders/sfm/`, `encoders/vjepa/`)

Deploys foundation model encoders as live HTTP endpoints.

| Feature | Detail |
|---|---|
| Single slice inference | `POST /infer` — returns 1024-dim embedding |
| Batch inference | `POST /batch_infer` — N slices in one GPU forward pass |
| TRT optimization | `sfm_server_trt.py` — TensorRT FP16 + PyTriton dynamic batching |
| Autoscaling | RPS-based, min 1 → max 3 replicas, configurable trigger |
| Auto-shutdown | Scales to 0 when idle — saves GPU cost |
| Model loading | Weights downloaded from TrueFoundry ML Repo on startup via `artifacts_download` |
| Architecture | Controlled via `SFM_ARCH` env var — swap models without code changes |

---

### 2. Decoder Serving (`decoder/`)

Deploys task-specific decoder heads on top of encoder embeddings.

| Feature | Detail |
|---|---|
| Decoupled architecture | Encoder and decoder run as separate services — scale independently |
| Fused architecture | Encoder + decoder in one pod — lower latency for single requests |
| Unified inference | `inference_service.py` — single `/infer` endpoint fans out to all decoders in parallel |
| Task heads | Classification, detection, segmentation — controlled via `HEAD_TYPE` env var |
| Encoder URL | Configured via `ENCODER_URL` env var — swap encoders without redeploying decoders |
| No GPU needed | Decoder services run on CPU only |

---

### 3. Fine-Tuning (`encoders/sfm/finetune.py`)

Long-running fine-tuning jobs with automatic checkpoint save and resume.

| Feature | Detail |
|---|---|
| Checkpoint save | Saved to ML Repo every epoch — full model + optimizer state |
| Auto-resume | On retry, scans ML Repo for latest checkpoint and resumes from that epoch |
| Spot instance safe | `retries: 3` in config — preemptions never lose training progress |
| Dataset preprocessing | `preprocess_dataset.py` — pre-cache dataset as ML Repo artifact for 21x faster training |
| HuggingFace support | Set `HF_DATASET` env var — dataset downloaded directly into job |
| Metrics logging | Loss + accuracy logged per epoch to ML Repo run dashboard |
| Full lineage | Final model traces back to base model, dataset, and training run |
| Configurable | All hyperparameters controlled via env vars — no code changes needed |

---

### 4. Load Testing (`locust_service/`)

Locust load testing service deployed on the same cluster for clean GPU-only measurement.

| Feature | Detail |
|---|---|
| Server types | Supports both FastAPI (`SERVER_TYPE=fastapi`) and PyTriton (`SERVER_TYPE=triton`) |
| Batch size | Configurable via `BATCH_SIZE` env var |
| Pod-to-pod | Deployed on cluster — uses internal URL, eliminates network overhead |
| Autoscaling test | Ramp users to trigger autoscaling and observe RPS scaling |

---

## Prerequisites

- TrueFoundry account with a connected GPU cluster
- ML Repo created (e.g. `slb-pov`)
- Base model weights uploaded to ML Repo (see `UI_README.md` Section 2)
- GitHub repo access (public — no auth needed)

---

## Getting Started

For full step-by-step UI instructions see **[UI_README.md](./UI_README.md)**.

Quick overview:
1. Upload your model weights to ML Repo → get FQN
2. Deploy encoder service → get internal cluster URL
3. Run preprocessing job → cache dataset as ML Repo artifact
4. Trigger fine-tuning job → monitor metrics in ML Repo
5. Deploy Locust service → run load test → observe autoscaling

---

## Key Concepts

**Internal cluster URL** — services communicate pod-to-pod using:
```
http://SERVICE-NAME.WORKSPACE-NAME.svc.cluster.local:8000
```
This is 80-100x faster than the external URL. Always use this for service-to-service calls.

**ML Repo FQN** — every model, artifact, and checkpoint has a unique versioned identifier:
```
model:tenant/ml-repo/model-name:version
artifact:tenant/ml-repo/artifact-name:version
```

**Env var driven** — all scripts are configured entirely via environment variables. No code changes needed to swap models, datasets, or hyperparameters.
