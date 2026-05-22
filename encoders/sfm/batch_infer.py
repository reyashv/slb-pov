import os
import json
import time
import numpy as np
import requests
from truefoundry.ml import get_client, ArtifactPath

# ── Config ────────────────────────────────────────────────────────────────────
ENCODER_URL      = os.environ.get("ENCODER_URL", "http://sfm-large.slb-ws.svc.cluster.local:8000")
HF_DATASET       = os.environ.get("HF_DATASET", "porestar/seismicfoundationmodel-interpolation")
HF_SPLIT         = os.environ.get("HF_SPLIT", "train")
OUTPUT_NAME      = os.environ.get("OUTPUT_ARTIFACT_NAME", "sfm-interpolation-embeddings")
IMG_SIZE         = int(os.environ.get("SFM_IMG_SIZE", "224"))
TIMEOUT          = int(os.environ.get("REQUEST_TIMEOUT", "60"))
MAX_SLICES       = int(os.environ.get("MAX_SLICES", "8000"))  # process all by default

OUTPUT_DIR       = "/tmp/sfm-output"
PROGRESS_FILE    = os.path.join(OUTPUT_DIR, "progress.json")
EMBEDDINGS_FILE  = os.path.join(OUTPUT_DIR, "embeddings.json")
PROGRESS_ARTIFACT = "sfm-interpolation-infer-progress"
MAX_VERSION_SEARCH = 20
SAVE_EVERY       = 50  # save progress every 50 slices


# ── Progress helpers ──────────────────────────────────────────────────────────

def load_progress(client):
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
    print("No existing progress — starting from scratch")
    return set()


def save_progress(client, run, processed_set, embeddings):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"processed": list(processed_set)}, f)
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


# ── Inference ─────────────────────────────────────────────────────────────────

def infer_slice(data, img_size, timeout):
    payload = {"data": data, "height": img_size, "width": img_size}
    t0 = time.time()
    resp = requests.post(f"{ENCODER_URL}/infer", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["features"], time.time() - t0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from datasets import load_dataset

    client = get_client()
    run = client.create_run(ml_repo="slb-pov", run_name="sfm-interpolation-batch-infer")
    print(f"Started run: {run.run_id}")

    try:
        run.log_params({
            "encoder_url": ENCODER_URL,
            "hf_dataset": HF_DATASET,
            "hf_split": HF_SPLIT,
            "img_size": IMG_SIZE,
            "max_slices": MAX_SLICES,
        })

        # ── a. Load dataset from HuggingFace ──
        print(f"Loading {HF_DATASET} (split={HF_SPLIT}) from HuggingFace...")
        dataset = load_dataset(HF_DATASET, split=HF_SPLIT).with_format(type="numpy")
        total = min(len(dataset), MAX_SLICES)
        print(f"Dataset size: {len(dataset)}, processing: {total}")

        # ── c. Load progress ──
        processed = load_progress(client)

        embeddings = {}
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "r") as f:
                embeddings = json.load(f)

        # ── Run inference ──
        remaining_indices = [i for i in range(total) if str(i) not in processed]
        print(f"Processing {len(remaining_indices)} remaining slices ({len(processed)} already done)...")

        latencies = []
        errors = 0

        for i, idx in enumerate(remaining_indices):
            try:
                item = dataset[idx]

                # Get seismic array
                seismic = item["seismic"].astype(np.float32)
                if seismic.ndim == 3:
                    seismic = seismic[:, :, 0]

                # Normalize
                seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)

                # Flatten for API
                data = seismic.flatten().tolist()
                h, w = seismic.shape

                features, latency = infer_slice(data, h, TIMEOUT)

                embeddings[str(idx)] = {
                    "features": features,
                    "latency_s": round(latency, 3),
                    "dim": len(features),
                }
                processed.add(str(idx))
                latencies.append(latency)

                print(f"[{len(processed)}/{total}] slice_{idx} — {latency:.3f}s, dim={len(features)}")

                # ── b. Save progress every N slices ──
                if (i + 1) % SAVE_EVERY == 0:
                    save_progress(client, run, processed, embeddings)
                    avg_lat = sum(latencies) / len(latencies)
                    run.log_metrics({
                        "slices_processed": len(processed),
                        "avg_latency_s": round(avg_lat, 3),
                        "errors": errors,
                    }, step=len(processed))

            except Exception as e:
                errors += 1
                print(f"ERROR on slice {idx}: {e} — skipping")
                continue

        # ── Final save ──
        save_progress(client, run, processed, embeddings)

        # ── Log final output artifact ──
        final_output = os.path.join(OUTPUT_DIR, "final_embeddings.json")
        with open(final_output, "w") as f:
            json.dump({
                "total_slices": total,
                "processed": len(processed),
                "errors": errors,
                "img_size": IMG_SIZE,
                "encoder_url": ENCODER_URL,
                "dataset": HF_DATASET,
                "embeddings": embeddings,
            }, f)

        av = run.log_artifact(
            name=OUTPUT_NAME,
            artifact_paths=[ArtifactPath(src=final_output, dest="final_embeddings.json")],
            metadata={
                "total_slices": total,
                "processed": len(processed),
                "errors": errors,
                "dataset": HF_DATASET,
            }
        )

        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        print(f"\n{'='*55}")
        print(f"Batch inference complete")
        print(f"  Dataset:         {HF_DATASET}")
        print(f"  Total slices:    {total}")
        print(f"  Processed:       {len(processed)}")
        print(f"  Errors:          {errors}")
        print(f"  Avg latency:     {avg_lat:.3f}s")
        print(f"  Output artifact: {av.fqn}")
        print(f"{'='*55}")

        run.log_metrics({
            "final_slices_processed": len(processed),
            "final_errors": errors,
            "final_avg_latency_s": round(avg_lat, 3),
        })

    finally:
        run.end()


if __name__ == "__main__":
    main()
