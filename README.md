# SFM PoV on TrueFoundry

Deploy SFM and V-JEPA 2 encoders + decoder heads on TrueFoundry. Fine-tune jobs swap in seamlessly.

## Step 0: Setup


```bash
git clone git@github.com:reyashv/slb-pov.git
cd slb-pov
```



Edit `env.sh`: set `WORKSPACE_FQN` to your workspace. Then:


```bash
source env.sh
```



Artifacts (`model:slb-pilot/slb-pov/...`) must already exist in the `slb-pilot` ML Repo. If not, run `upload-scripts/upload_vjepa.py` (auto from HuggingFace) and `upload-scripts/upload-sfm.py` (needs Baidu checkpoints, see script).

## Step 1: Deploy

Apply all services and jobs:


```bash
for f in yaml/*.yaml; do envsubst < "$f" > /tmp/x.yaml && tfy apply -f /tmp/x.yaml; done
```



Or one at a time:


```bash
envsubst < yaml/sfm-base.yaml > /tmp/x.yaml && tfy apply -f /tmp/x.yaml
```



What's in `yaml/`:

- `sfm-*`, `vjepa2-*`: encoder services
- `decoder-{classify,detect,segment}`: head only, calls encoder over internal DNS
- `fused-{classify,detect,segment}`: encoder + head in one pod
- `sfm-finetune`, `vjepa-{probe,joint}-*`: fine-tune jobs

Validate everything:


```bash
python testing/validate_slb.py    # health + inference across all services
python testing/test_sfm.py        # SFM encoders only, prints feature dim + sample
```



> Hostnames in the test scripts are hardcoded to `dipo-ws`. For a different workspace, sed the file or set the hostnames via env var.

## Step 2: Fine-tune

Already deployed in Step 1. Trigger from TF UI → Jobs → `sfm-finetune` → Run, or via CLI. When done, the job logs the new artifact FQN, e.g.:


```
Done — logged as: model:slb-pilot/slb-pov/sfm-base-finetuned:3
```



Also visible in TF UI → Models.

## Step 3: Redeploy with fine-tuned weights

Override the artifact var for one apply:


```bash
ARTIFACT_SFM_BASE=model:slb-pilot/slb-pov/sfm-base-finetuned:3 \
  envsubst < yaml/sfm-base.yaml > /tmp/x.yaml && tfy apply -f /tmp/x.yaml
```


Re-run the validation script to confirm new output. The encoder pod rolls a new version; one-click rollback in the Versions tab.

## Step 4: Metrics + monitoring

TF UI → Service → Monitoring tab. GPU / memory / latency live, exportable to Grafana. Training curves under Jobs → Run → Metrics.
