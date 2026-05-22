import os
import glob
import json
import time
import numpy as np
import requests
from PIL import Image
from truefoundry.ml import get_client, ArtifactPath

# ── Config ────────────────────────────────────────────────────────────────────
ENCODER_URL       = os.environ.get("ENCODER_URL", "https://sfm-large-slb-ws-8000.slb-pilot.truefoundry.cloud")
DATA_ARTIFACT_FQN = os.environ.get("DATA_ARTIFACT_FQN", "artifact:slb-pilot/slb-pov/sfm-facies-data:1")
OUTPUT_NAME       = os.environ.get("OUTPUT_ARTIFACT_NAME", "sfm-batch-embeddings")
IMG_SIZE          = int(os.environ.get("SFM_IMG_SIZE", "224"))
TIMEOUT           = int(os.environ.get("REQUEST_TIMEOUT", "60"))
BATCH_SIZE        = int(os.environ.get("BATCH_SIZE", "1"))  # slices per request

# Local paths inside job container
DATA_DIR          = "/tmp/sfm-data"
OUTPUT_DIR        = "/tmp/sfm-output"
PROGRESS_FILE     = os.path.join(OUTPUT_DIR, "progress.json")
EMBEDDINGS_FILE   = os.path.join(OUTPUT_DIR, "embeddings.json")

# Progress artifact name in ML Repo
PROGRESS_ARTIFACT = "sfm-batch-infer-progress"
MAX_VERSION_SEARCH = 20


# ── Step 1: Load progress from ML Repo ────────────────────────────────────────

def load_progress(client):
    """
    Check ML Repo for latest progress artifact.
    Returns set of already-processed slice filenames.
    """
    print("Checking ML Repo for existing progress...")

    for version in range(MAX_VERSION_SEARCH, 0, -1):
        try:
            fqn = f"artifact:slb-pilot/slb-pov/{PROGRESS_ARTIFACT}:{version}"
            av = client.get_artifact_version_by_fqn(fqn=fqn)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            av.download(path=OUTPUT_DIR)

            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, "r") as f:
                    progress = json.load(f)
                processed = set(progress.get("processed", []))
                print(f"Found progress: {len(processed)} slices already done (from {fqn})")
                return processed
        except Exception:
            continue

    print("No existing progress found — starting from scratch")
    return set()


# ── Step 2: Save progress to ML Repo ─────────────────────────────────────────

def save_progress(client, run, processed_set, embeddings):
    """Save progress and partial embeddings to ML Repo."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save progress tracker
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"processed": list(processed_set)}, f)

    # Save embeddings so far
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings, f)

    av = run.log_artifact(
        name=PROGRESS_ARTIFACT,
        artifact_paths=[
            ArtifactPath(src=PROGRESS_FILE, dest="progress.json"),
            ArtifactPath(src=EMBEDDINGS_FILE, dest="embeddings.json"),
        ],
        metadata={"processed_count": len(processed_set)}
    )
    print(f"Progress saved — {len(processed_set)} slices done → {av.fqn}")


# ── Step 3: Load seismic slice ─────────────────────────────────────────────────

def load_slice(dat_file, img_size):
    """Load a .dat file and prepare as flat list for the API."""
    raw = np.fromfile(dat_file, dtype=np.float32).reshape(768, 768)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    img = Image.fromarray(raw).resize((img_size, img_size))
    return np.array(img, dtype=np.float32).flatten().tolist()


# ── Step 4: Send to encoder ────────────────────────────────────────────────────

def infer_slice(data, img_size, timeout):
    """Send one slice to the SFM encoder and return feature vector."""
    payload = {"data": data, "height": img_size, "width": img_size}
    t0 = time.time()
    resp = requests.post(f"{ENCODER_URL}/infer", json=payload, timeout=timeout)
    resp.raise_for_status()
    latency = time.time() - t0
    return resp.json()["features"], latency


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    client = get_client()
    run = client.create_run(ml_repo="slb-pov", run_name="sfm-batch-infer")
    print(f"Started run: {run.run_id}")

    try:
        run.log_params({
            "encoder_url": ENCODER_URL,
            "data_artifact_fqn": DATA_ARTIFACT_FQN,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
        })

        # ── a. Download dataset ──
        print(f"Downloading dataset from {DATA_ARTIFACT_FQN}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        result = client.get_artifact_version_by_fqn(DATA_ARTIFACT_FQN).download(path=DATA_DIR)
        data_dir = result if isinstance(result, str) else result.download_dir
        seismic_dir = os.path.join(data_dir, "seismic")

        dat_files = sorted(glob.glob(os.path.join(seismic_dir, "*.dat")))
        print(f"Found {len(dat_files)} seismic slices")

        # ── c. Load progress — resume from where we left off ──
        processed = load_progress(client)

        # Load existing embeddings if resuming
        embeddings = {}
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "r") as f:
                embeddings = json.load(f)

        # ── Run inference ──
        total = len(dat_files)
        remaining = [f for f in dat_files if os.path.basename(f) not in processed]
        print(f"Processing {len(remaining)} remaining slices ({len(processed)} already done)...")

        latencies = []
        errors = 0
        save_every = 10  # save progress every 10 slices

        for i, dat_file in enumerate(remaining):
            slice_name = os.path.basename(dat_file)

            try:
                # Load and send slice
                data = load_slice(dat_file, IMG_SIZE)
                features, latency = infer_slice(data, IMG_SIZE, TIMEOUT)

                # Store embedding
                embeddings[slice_name] = {
                    "features": features,
                    "latency_s": round(latency, 3),
                    "dim": len(features),
                }
                processed.add(slice_name)
                latencies.append(latency)

                print(f"[{len(processed)}/{total}] {slice_name} — {latency:.2f}s, dim={len(features)}")

                # ── b. Save outputs + progress every N slices ──
                if (i + 1) % save_every == 0:
                    save_progress(client, run, processed, embeddings)
                    avg_lat = sum(latencies) / len(latencies)
                    run.log_metrics({
                        "slices_processed": len(processed),
                        "avg_latency_s": round(avg_lat, 3),
                        "errors": errors,
                    }, step=len(processed))

            except Exception as e:
                errors += 1
                print(f"ERROR on {slice_name}: {e} — skipping")
                run.log_metrics({"errors": errors}, step=len(processed))
                continue

        # ── Final save ──
        save_progress(client, run, processed, embeddings)

        # ── Log final embeddings as clean output artifact ──
        print(f"Logging final embeddings to ML Repo as {OUTPUT_NAME}...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        final_output = os.path.join(OUTPUT_DIR, "final_embeddings.json")
        with open(final_output, "w") as f:
            json.dump({
                "total_slices": total,
                "processed": len(processed),
                "errors": errors,
                "img_size": IMG_SIZE,
                "encoder_url": ENCODER_URL,
                "embeddings": embeddings,
            }, f)

        av = run.log_artifact(
            name=OUTPUT_NAME,
            artifact_paths=[ArtifactPath(src=final_output, dest="final_embeddings.json")],
            metadata={
                "total_slices": total,
                "processed": len(processed),
                "errors": errors,
                "encoder": ENCODER_URL,
            }
        )

        # ── Summary ──
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        print(f"\n{'='*50}")
        print(f"Batch inference complete")
        print(f"  Total slices:    {total}")
        print(f"  Processed:       {len(processed)}")
        print(f"  Errors:          {errors}")
        print(f"  Avg latency:     {avg_lat:.2f}s")
        print(f"  Output artifact: {av.fqn}")
        print(f"{'='*50}")

        run.log_metrics({
            "final_slices_processed": len(processed),
            "final_errors": errors,
            "final_avg_latency_s": round(avg_lat, 3),
        })

    finally:
        run.end()


if __name__ == "__main__":
    main()
