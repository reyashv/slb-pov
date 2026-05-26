import os
import json
import time
import numpy as np
import requests
from truefoundry.ml import get_client, ArtifactPath
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
ENCODER_URL      = os.environ.get("ENCODER_URL", "http://sfm-large.slb-ws.svc.cluster.local:8000")
HF_DATASET       = os.environ.get("HF_DATASET", "porestar/seismicfoundationmodel-interpolation")
HF_SPLIT         = os.environ.get("HF_SPLIT", "train")
IMG_SIZE         = int(os.environ.get("SFM_IMG_SIZE", "224"))
TIMEOUT          = int(os.environ.get("REQUEST_TIMEOUT", "60"))
TARGET_HOURS     = float(os.environ.get("TARGET_HOURS", "1.5"))  # run for 1.5 hours
SAVE_EVERY       = int(os.environ.get("SAVE_EVERY", "500"))      # save progress every 500 slices

OUTPUT_DIR        = "/tmp/sfm-sustained"
PROGRESS_FILE     = os.path.join(OUTPUT_DIR, "progress.json")
PROGRESS_ARTIFACT = "sfm-sustained-infer-progress"
MAX_VERSION_SEARCH = 20


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
                total_processed = progress.get("total_processed", 0)
                elapsed = progress.get("elapsed_seconds", 0)
                print(f"Resuming — {total_processed} slices done, {elapsed/3600:.2f}h elapsed")
                return total_processed, elapsed
        except Exception:
            continue
    print("No existing progress — starting from scratch")
    return 0, 0


def save_progress(client, run, total_processed, elapsed_seconds, latencies):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "total_processed": total_processed,
            "elapsed_seconds": elapsed_seconds,
            "avg_latency_s": round(avg_lat, 4),
        }, f)

    av = run.log_artifact(
        name=PROGRESS_ARTIFACT,
        artifact_paths=[ArtifactPath(src=PROGRESS_FILE, dest="progress.json")],
        metadata={
            "total_processed": total_processed,
            "elapsed_hours": round(elapsed_seconds / 3600, 2),
        }
    )
    print(f"Progress saved — {total_processed} slices, {elapsed_seconds/3600:.2f}h → {av.fqn}")


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
    run = client.create_run(ml_repo="slb-pov", run_name="sfm-sustained-inference")
    print(f"Started run: {run.run_id}")
    print(f"Target duration: {TARGET_HOURS} hours")

    try:
        run.log_params({
            "encoder_url": ENCODER_URL,
            "dataset": HF_DATASET,
            "target_hours": TARGET_HOURS,
            "img_size": IMG_SIZE,
        })

        # Load dataset
        print(f"Loading {HF_DATASET} from HuggingFace...")
        dataset = load_dataset(HF_DATASET, split=HF_SPLIT).with_format(type="numpy")
        dataset_size = len(dataset)
        print(f"Dataset: {dataset_size} slices (will loop until {TARGET_HOURS}h reached)")

        # Load progress — resume if retrying
        total_processed, prev_elapsed = load_progress(client)

        target_seconds = TARGET_HOURS * 3600
        job_start = time.time()
        latencies = []
        errors = 0
        loop = 0

        print(f"\nStarting sustained inference...")
        print(f"Will run until {TARGET_HOURS} hours of processing time is reached\n")

        while True:
            elapsed_total = prev_elapsed + (time.time() - job_start)

            # Check if we've hit the target duration
            if elapsed_total >= target_seconds:
                print(f"\nTarget duration reached: {elapsed_total/3600:.2f}h ✅")
                break

            loop += 1
            print(f"Loop {loop} — elapsed: {elapsed_total/3600:.2f}h / {TARGET_HOURS}h")

            for i in range(dataset_size):
                # Check time every 100 slices
                if i % 100 == 0:
                    elapsed_total = prev_elapsed + (time.time() - job_start)
                    if elapsed_total >= target_seconds:
                        break

                try:
                    item = dataset[i]
                    seismic = item["seismic"].astype(np.float32)
                    if seismic.ndim == 3:
                        seismic = seismic[:, :, 0]
                    seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
                    data = seismic.flatten().tolist()
                    h, w = seismic.shape

                    _, latency = infer_slice(data, h, TIMEOUT)
                    latencies.append(latency)
                    total_processed += 1

                except Exception as e:
                    errors += 1
                    continue

                # Save progress every N slices
                if total_processed % SAVE_EVERY == 0:
                    elapsed_total = prev_elapsed + (time.time() - job_start)
                    avg_lat = sum(latencies[-SAVE_EVERY:]) / len(latencies[-SAVE_EVERY:])
                    throughput = len(latencies) / (time.time() - job_start)

                    print(f"[{total_processed} slices | {elapsed_total/3600:.2f}h] "
                          f"avg: {avg_lat:.3f}s | throughput: {throughput:.1f}/s | errors: {errors}")

                    save_progress(client, run, total_processed, elapsed_total, latencies)

                    run.log_metrics({
                        "total_processed": total_processed,
                        "elapsed_hours": round(elapsed_total / 3600, 3),
                        "avg_latency_s": round(avg_lat, 4),
                        "throughput_per_sec": round(throughput, 2),
                        "errors": errors,
                    }, step=total_processed)

        # Final summary
        elapsed_total = prev_elapsed + (time.time() - job_start)
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        throughput = len(latencies) / (time.time() - job_start) if time.time() > job_start else 0

        print(f"\n{'='*60}")
        print(f"Sustained inference complete")
        print(f"  Total slices processed: {total_processed}")
        print(f"  Total time:             {elapsed_total/3600:.2f} hours")
        print(f"  Avg latency:            {avg_lat:.4f}s")
        print(f"  Throughput:             {throughput:.2f} slices/sec")
        print(f"  Errors:                 {errors}")
        print(f"  Dataset loops:          {loop}")
        print(f"{'='*60}")

        run.log_metrics({
            "final_total_processed": total_processed,
            "final_elapsed_hours": round(elapsed_total / 3600, 3),
            "final_avg_latency_s": round(avg_lat, 4),
            "final_throughput": round(throughput, 2),
            "final_errors": errors,
            "dataset_loops": loop,
        })

    finally:
        run.end()


if __name__ == "__main__":
    main()
