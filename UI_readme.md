# TrueFoundry — SLB Workflow Guide

This document provides step-by-step instructions for SLB to independently run and validate all workflows demonstrated during the PoC — model serving, fine-tuning, batch inference, and load testing — using the TrueFoundry platform UI.

---

## Important: Deploy vs Trigger (for Jobs)

- **Deploy** — done once. Sets up the job config, builds the Docker image, registers the job on TrueFoundry.
- **Trigger** — done every time you want to run. Just click the ▶ play button on an already deployed job.

---

## Repository Structure

The GitHub repository (`reyashv/slb-pov`) is organized as follows. SLB should place their own models and datasets in the indicated locations:

```
slb-pov/
├── encoders/
│   └── sfm/
│       ├── sfm_server.py          # FastAPI inference server
│       ├── sfm_server_trt.py      # PyTriton TRT inference server
│       ├── finetune.py            # Fine-tuning script
│       ├── batch_infer.py         # Batch inference script
│       ├── preprocess_dataset.py  # Dataset preprocessing script
│       ├── convert_trt.py         # TRT conversion script
│       └── requirements.txt
│   └── vjepa/
│       ├── vjepa_encoder_server.py
│       ├── probe_train.py
│       └── requirements.txt
├── decoder/
│   ├── server.py                  # Decoupled decoder server
│   ├── fused_server.py            # Fused encoder+decoder server
│   ├── inference_service.py       # Unified inference entry point
│   ├── heads/
│   │   ├── classify.py
│   │   ├── detect.py
│   │   └── segment.py
│   └── requirements.txt
├── locust_service/
│   ├── locust_benchmark.py        # Load test script
│   └── requirements.txt
└── yaml/                          # Service/job configs (optional)
```

**Where to put your own files:**
- Your model weights → upload to TrueFoundry ML Repo (see Section 2)
- Your dataset → either upload to ML Repo as an artifact, or reference a HuggingFace dataset name in the job env vars
- Your decoder heads → replace `decoder/heads/classify.py`, `detect.py`, `segment.py` with your own

---

## Section 1 — Initial Setup

### 1.1 Create a Workspace

| Step | Where | What to do |
|---|---|---|
| 1 | TrueFoundry UI → Workspaces | Click **New Workspace** |
| 2 | Name | Enter your workspace name (e.g. `slb-ws`) |
| 3 | Cluster | Select your cluster |
| 4 | Permission | Add permission for ML Repo |
| 5 | Save | Click **Create** |

### 1.2 Create an ML Repo

| Step | Where | What to do |
|---|---|---|
| 1 | TrueFoundry UI → Repositories | Click **+New Repository** |
| 2 | Name | Enter repo name (e.g. `slb-pov`) |
| 3 | Storage Integration | Select your Storage Integration |
| 4 | Save | Click **Create** |

---

## Section 2 — Storing Models and Artifacts in ML Repo

Everything — model weights, checkpoints, preprocessed datasets, output embeddings — is stored in ML Repo as versioned artifacts. Every artifact gets a unique FQN (Fully Qualified Name) used to reference it in code and service configs.

### 2.1 Upload a Model

**Option A — Upload via UI:**

| Step | Where | What to fill |
|---|---|---|
| 1 | ML Repo → Models → **New Model** | Click **New Model** |
| 2 | Name | Enter model name (e.g. `sfm-large`) |
| 3 | Description | Optional — add a short description |
| 4 | Model Source | Select **TrueFoundry Managed Source** |
| 5 | Upload | Click **Click to upload files** → select your `.pth` file (max 2048MB) OR **Click to upload folders** → select your model folder |
| 6 | Environment | Optional — skip for now |
| 7 | Framework | Optional — toggle on → select `PyTorch` |
| 8 | Metadata | Leave as `{}` or add key-value info e.g. `{"arch": "vit_large", "img_size": 224}` |
| 9 | Version Alias | Optional — e.g. `v1.0.0` |
| 10 | Create | Click **Create** |
| 11 | Get FQN | After creation → click on the model version → copy the FQN shown (e.g. `model:slb-pilot/slb-pov/sfm-large:1`) |

> **Note:** If your model file is larger than 2048MB, use Option B (Python upload) instead — the UI has a 2048MB file size limit.

**Option B — Upload via Python:**

```python
from truefoundry.ml import get_client, PyTorchFramework

client = get_client()
run = client.create_run(ml_repo="slb-pov", run_name="upload-model")
mv = run.log_model(
    name="your-model-name",
    model_file_or_folder="/path/to/model/folder",
    framework=PyTorchFramework()
)
print(f"Model FQN: {mv.fqn}")
run.end()
```

### 2.2 Upload a Dataset as Artifact

**Option A — Upload via UI:**

| Step | Where | What to fill |
|---|---|---|
| 1 | ML Repo → Artifacts → **New Artifact** | Click **New Artifact** |
| 2 | Name | Enter artifact name (e.g. `geobody-preprocessed-train`) |
| 3 | Description | Optional — add a short description |
| 4 | Artifact Source | Select **TrueFoundry Managed Source** |
| 5 | Upload | Click **Click to upload files** → select your dataset file (max 2048MB) OR **Click to upload folders** → select your dataset folder |
| 6 | Metadata | Leave as `{}` or add info e.g. `{"split": "train", "num_samples": 3500}` |
| 7 | Version Alias | Optional — e.g. `v1.0.0` |
| 8 | Create | Click **Create** |
| 9 | Get FQN | After creation → click on the artifact version → copy FQN (e.g. `artifact:slb-pilot/slb-pov/geobody-preprocessed-train:1`) |

> **Note:** If your dataset file is larger than 2048MB, use the Python upload method instead:

```python
from truefoundry.ml import get_client, ArtifactPath

client = get_client()
run = client.create_run(ml_repo="slb-pov", run_name="upload-dataset")
run.log_artifact(
    name="your-dataset-name",
    artifact_paths=[ArtifactPath(src="/path/to/your/data", dest="data")]
)
run.end()
```

### 2.3 Understanding FQNs

FQNs follow this pattern:
```
model:tenant/ml-repo-name/model-name:version
artifact:tenant/ml-repo-name/artifact-name:version
```

Examples:
```
model:slb-pilot/slb-pov/sfm-large:1
artifact:slb-pilot/slb-pov/geobody-preprocessed-train:1
artifact:slb-pilot/slb-pov/sfm-geobody-finetune-checkpoint:10
```

The version number increments automatically every time you log a new version. Always use the latest version number when referencing in configs.

---

## Section 3 — Deploying an Inference Service

### 3.1 Deploy the Encoder Service (e.g. sfm-large)

**Step 1 — Select Workspace and Source**

| Step | Where | What to do |
|---|---|---|
| 1 | Deployments → Services → **Deploy new Service** | Click Deploy new Service |
| 2 | Select workspace | Select your workspace (e.g. `slb-ws`) |
| 3 | Source type | Select **Code from Git repo** |
| 4 | Git repo type | Click **Deploy from a public Git repo** (or your own if forked) |
| 5 | Click | **Next** |

**Step 2 — Configure Service**

| Step | Field | What to fill |
|---|---|---|
| 1 | Name | `sfm-large` |
| 2 | Repo URL | `https://github.com/reyashv/slb-pov` |
| 3 | Branch Name | `restructure/slb-pov` |
| 4 | Commit SHA | Enter latest commit hash |
| 5 | Build type | **Python Code (I don't have Dockerfile)** |
| 6 | Python version | `3.11` |
| 7 | Requirements | pip → `requirements.txt` |
| 8 | Command | `gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 sfm_server:app` |
| 9 | Port | `8000` |
| 10 | Expose | ✅ Yes |

**Step 3 — Resources**

| Field | Value |
|---|---|
| Device Type | Select your GPU (e.g. **L40S (48GB)** for sfm-large) |
| CPU Request / Limit | `4` / `8` |
| Memory Request / Limit | `16000` MB / `32000` MB |
| Storage Request / Limit | `20000` MB / `50000` MB |
| Capacity Type | **Prefer Spot** for cost saving, **On Demand** for demo stability |

**Step 4 — Environment Variables**

Toggle **Environment Variables** ON and add:

| Variable | Value | Description |
|---|---|---|
| `MODEL_DIR` | `/mnt/model` | Path where model weights are downloaded |
| `SFM_ARCH` | `vit_large_patch16` or `vit_base_patch16` | Model architecture |
| `SFM_IMG_SIZE` | `224` or `512` | Input image size |

**Step 5 — Artifacts Download**

Toggle **Artifacts Download** ON and add:

| Field | Value |
|---|---|
| FQN | Your model FQN e.g. `model:slb-pilot/slb-pov/sfm-large:1` |
| Download path | `/mnt/model` |

**Step 6 — Replicas and Autoscaling**

| Field | Value |
|---|---|
| Desired replicas | `1` |
| Enable Autoscaling | Toggle ON |
| Min replicas | `1` |
| Max replicas | `3` |
| Scale trigger | RPS per replica = `20` |
| Auto Shutdown | Toggle ON → `300` seconds |

**Step 7 — Submit**

| Step | What to do |
|---|---|
| 1 | Review all settings |
| 2 | Click **Deploy** |
| 3 | Wait for pod to show **Running** in Services → Pods tab |
| 4 | Copy internal URL: `http://sfm-large.YOUR-WS.svc.cluster.local:8000` |

### 3.2 Get the Internal Cluster URL

Once deployed, the internal cluster URL follows this pattern:
```
http://SERVICE-NAME.WORKSPACE-NAME.svc.cluster.local:8000
```

Example:
```
http://sfm-large.slb-ws.svc.cluster.local:8000
```

Use this URL when configuring decoder services or batch inference jobs — it is 80-100x faster than the external URL.

### 3.3 Deploy the Decoder Services

Deploy `decoder-classify`, `decoder-detect`, `decoder-segment` — same steps as 3.1 but with these differences:

| Field | Value |
|---|---|
| Build context | `./decoder/` |
| Command | `gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 server:app` |
| GPU | Chosen based on what is required |
| `ENCODER_URL` | `http://sfm-large.slb-ws.svc.cluster.local:8000` |
| `HEAD_TYPE` | `classify` / `detect` / `segment` |

### 3.4 Configure Autoscaling

| Step | Where | What to fill |
|---|---|---|
| 1 | Service → Edit → Autoscaling | Enable autoscaling |
| 2 | Min replicas | `1` |
| 3 | Max replicas | `3` (or more based on expected load) |
| 4 | Scale trigger | RPS per replica → `20` |
| 5 | Auto-shutdown | Enable → `300` seconds (scales to 0 when idle, saves GPU cost) |
| 6 | Save | Click **Update** |

### 3.5 Monitor a Service

| What to see | Where |
|---|---|
| Pod status, replica count | Service → **Pods** tab |
| GPU utilization, memory, temperature | Service → **Monitoring** tab |
| Request logs | Service → Pods → click pod → **Logs** |
| Cost | Service → **Cost** tab |
| Endpoint URL | Service → top of page → **Endpoint/Credentials** |

---

## Section 4 — Running a Fine-Tuning Job

### 4.1 One-time Dataset Preprocessing (Recommended)

Before fine-tuning, preprocess your dataset into a cached artifact to significantly reduce training time.

> We saw **7 hours → 20 minutes** with this optimization (21x speedup).

**Deploy (one time):**

Follow Deploy new Job → Code from Git repo → Python Code buildpack with these settings:

| Field | Value |
|---|---|
| Name | `preprocess-dataset` |
| Repo URL | `https://github.com/reyashv/slb-pov` |
| Branch | `restructure/slb-pov` |
| Build context | `./encoders/sfm/` |
| Command | `python preprocess_dataset.py` |
| GPU | None (CPU only) |
| CPU Request / Limit | `4` / `8` |
| Memory Request / Limit | `16000` MB / `32000` MB |
| Storage Request / Limit | `50000` MB / `100000` MB |
| Retries | `2` |
| Timeout | `7200` (2 hours) |

**Environment variables:**

| Variable | Value |
|---|---|
| `HF_DATASET` | Your HuggingFace dataset name e.g. `porestar/seismicfoundationmodel-geobody` |

Click **Submit** to deploy.

**Trigger (every time you want to preprocess):**
1. Go to Workspace → Jobs → `preprocess-dataset`
2. Click **▶ Trigger**
3. Watch logs — you should see `Processed 500/3500`, `Processed 1000/3500` etc.
4. Wait ~30-60 mins depending on dataset size
5. Verify in ML Repo → Artifacts → `your-dataset-preprocessed-train:1` and `your-dataset-preprocessed-validation:1` exist

> **Note:** Only needs to be triggered once per dataset. After artifacts are created, all future fine-tuning runs reuse them.

> **If using your own dataset instead of HuggingFace:** Update `preprocess_dataset.py` to load from your own source. The output format must remain the same — `torch.save({"tensors": ..., "labels": ...}, output_file)`.

### 4.2 Fine-Tuning Job

**Deploy (one time):**

Follow the same Deploy new Job flow with these settings:

| Field | Value |
|---|---|
| Name | `sfm-finetune` |
| Repo URL | `https://github.com/reyashv/slb-pov` |
| Branch | `restructure/slb-pov` |
| Build context | `./encoders/sfm/` |
| Command | `python finetune.py` |
| GPU | Select your GPU type (e.g. L40S), count = 1 |
| CPU Request / Limit | `4` / `8` |
| Memory Request / Limit | `27200` MB / `32000` MB |
| Storage Request / Limit | `20000` MB / `100000` MB |
| Retries | `3` |
| Timeout | `36000` (10 hours) |

**Environment variables:**

| Variable | Value | Description |
|---|---|---|
| `MODEL_DIR` | FQN of your base model | e.g. `model:tenant/repo/sfm-large:1` |
| `SFM_ARCH` | `vit_large_patch16` | Model architecture |
| `SFM_IMG_SIZE` | `224` | Image size |
| `EPOCHS` | `10` | Number of training epochs |
| `BATCH_SIZE` | `8` | Training batch size |
| `LR` | `0.0001` | Learning rate |
| `NUM_CLASSES` | `2` | Number of output classes |
| `CHECKPOINT_EVERY` | `1` | Save checkpoint every N epochs |
| `CHECKPOINT_NAME` | `your-finetune-checkpoint` | Unique name — change for each new run |
| `OUTPUT_MODEL_NAME` | `your-model-finetuned` | Name for output model in ML Repo |
| `USE_PREPROCESSED` | `true` | Use preprocessed dataset artifact |
| `PREPROCESSED_TRAIN_FQN` | `artifact:tenant/repo/your-dataset-train:1` | FQN of preprocessed train artifact |
| `PREPROCESSED_VAL_FQN` | `artifact:tenant/repo/your-dataset-val:1` | FQN of preprocessed val artifact |

Click **Submit** to deploy.

**Trigger (every time you want to train):**
1. Go to Workspace → Jobs → `sfm-finetune`
2. Click **▶ Trigger**
3. Watch logs — epoch progress e.g. `Epoch 1/10 — Train Loss: 0.56, Train Acc: 0.74`
4. Watch metrics — ML Repo → your run → **Metrics** tab → loss and accuracy curves
5. Watch GPU — Job → your run → **Monitoring** tab → GPU utilization
6. Job completes automatically — final model saved to ML Repo as `your-model-finetuned:1`

> **Important — Checkpoint Resume:**
> - If the job crashes, TrueFoundry auto-retries up to 3 times
> - On retry, code automatically finds the latest checkpoint and resumes from that epoch
> - Use a **unique `CHECKPOINT_NAME`** for each new training run to avoid picking up old checkpoints

### 4.3 Monitor the Fine-Tuning Job

| What to see | Where |
|---|---|
| Training logs (epoch progress) | Jobs → Monitoring → **Logs** tab |
| Loss and accuracy curves | Jobs → Monitoring → **Metrics** tab |
| GPU utilization during training | Jobs → Monitoring → **Metrics** tab |
| Checkpoints saved | ML Repo → Artifacts → your checkpoint name |
| Final model | ML Repo → Models → your output model name |
| Full lineage | ML Repo → Models → click model → **Lineage** tab |

---

## Section 5 — Running Batch Inference

**Deploy (one time):**

| Field | Value |
|---|---|
| Name | `batch-infer` |
| Repo URL | `https://github.com/reyashv/slb-pov` |
| Branch | `restructure/slb-pov` |
| Build context | `./encoders/sfm/` |
| Command | `python batch_infer.py` |
| GPU | None — uses the running encoder service |
| CPU Request / Limit | `2` / `4` |
| Memory Request / Limit | `8000` MB / `16000` MB |
| Retries | `3` |
| Timeout | `10800` (3 hours) |

**Environment variables:**

| Variable | Value | Description |
|---|---|---|
| `ENCODER_URL` | `http://sfm-large.slb-ws.svc.cluster.local:8000` | Internal URL of running encoder service |
| `TARGET_HOURS` | `1.5` | How long to run (set to `0` for single pass) |
| `SAVE_EVERY` | `500` | Save progress every N slices |

Click **Submit** to deploy.

**Trigger:**
1. Make sure encoder service is running first
2. Go to Jobs → `batch-infer` → click **▶ Trigger**
3. Watch logs for progress every 500 slices
4. Output embeddings saved to ML Repo → Artifacts automatically

**Check outputs:**

| What to see | Where |
|---|---|
| Progress logs | Job → click run → **Logs** tab |
| Output embeddings | ML Repo → Artifacts → `sfm-interpolation-embeddings:1` |
| Progress checkpoints | ML Repo → Artifacts → `sfm-sustained-infer-progress` |

---

## Section 6 — Quick Reference

### Key Internal URLs Pattern
```
http://SERVICE-NAME.WORKSPACE-NAME.svc.cluster.local:8000
```

### Key FQN Patterns
```
model:TENANT/ML-REPO/MODEL-NAME:VERSION
artifact:TENANT/ML-REPO/ARTIFACT-NAME:VERSION
```

### Recommended GPU Types

| Use case | Recommended GPU |
|---|---|
| sfm-base (224x224) | L4 (24GB) |
| sfm-large (224x224) | L40S (48GB) |
| sfm-large (512x512) | A100 (80GB) |
| Fine-tuning sfm-large | L40S (48GB) |
| Decoder services | CPU only |
| Batch inference job | CPU only |

### Cost Saving Tips
- Enable **auto-shutdown** on all GPU services (300 seconds idle → scales to 0)
- Use **spot** instances for jobs (fine-tuning, batch inference) — retries handle preemptions
- Use **on-demand** instances for services during demos — avoids mid-demo preemptions
- Decoder services need **no GPU** — run on CPU only

### Using tfy apply (Advanced)
If you prefer deploying via CLI instead of UI, all service and job configs can be exported as YAML from the TrueFoundry UI (Service → Spec tab → **Apply using YAML**) and deployed with:
```bash
tfy apply -f your-service.yaml
```
