# SFM PoV on TrueFoundry

Deploy SFM and V-JEPA 2 models on TrueFoundry


## Step 0: Setup

- Clone:

    ```shell
    git clone git@github.com:reyashv/slb-pov.git
    cd slb-pov
    ```

- Download SFM checkpoints, e.g. `SFM-base`: https://rec.ustc.edu.cn/share/5264ec70-e839-11ee-bbda-13c1c8639a68, into `./Data/` (gitignored):

    ```shell
    mkdir -p Data
    ```

- Export your tenant-specific values:

    ```shell
    export WORKSPACE_FQN=tfy-slb-demo:<your-workspace>
    export MODEL_FQN=<your-model-fqn>
    ```

## Step 1: Deploy

- Deploy yaml using TrueFoundry CLI (example: `sfm` model)

    ```shell
    tfy apply -f sfm-deploy.yaml
    ```

## Step 2: Inference

- Use inference script from this repo 
    
    ```shell
    python testing-files/test-sfm.py
    ```

## Step 3: Fine-tune

- Submit the fine-tune Job (Facies, joint mode):

    ```shell
    tfy apply -f sfm-finetune.yaml
    ```

- Watch progress in TF UI → Jobs. Note the output FQNs from the Models tab.

## Step 4: Redeploy with fine-tuned weights

- Edit `sfm-deploy.yaml`, swap `artifact_version_fqn` to the new fine-tuned FQN, then:

    ```shell
    tfy apply -f sfm-deploy.yaml
    ```

- Re-run Step 2 to confirm new output.

## Step 5: Metrics + monitoring

- TF UI → Service → Monitoring tab. GPU/memory/latency live, exportable to Grafana. Training curves under Jobs → Run → Metrics.