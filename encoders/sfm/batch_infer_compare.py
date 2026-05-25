"""
batch_infer_compare.py

Runs batch inference on interpolation dataset (3500 slices) 
against both PyTorch and TensorRT endpoints and compares:
- Total time
- Average latency per slice
- Throughput (slices/sec)
- Speedup factor

Outputs comparison artifact to ML Repo.
"""

import os
import json
import time
import numpy as np
import requests
from truefoundry.ml import get_client, ArtifactPath
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
PYTORCH_URL  = os.environ.get("PYTORCH_URL", "http://sfm-base.slb-ws.svc.cluster.local:8000")
TRT_URL      = os.environ.get("TRT_URL", "http://sfm-base-trt.slb-ws.svc.cluster.local:8000")
HF_DATASET   = os.environ.get("HF_DATASET", "porestar/seismicfoundationmodel-interpolation")
HF_SPLIT     = os.environ.get("HF_SPLIT", "train")
IMG_SIZE     = int(os.environ.get("SFM_IMG_SIZE", "224"))
TIMEOUT      = int(os.environ.get("REQUEST_TIMEOUT", "60"))
MAX_SLICES   = int(os.environ.get("MAX_SLICES", "3500"))
OUTPUT_DIR   = "/tmp/sfm-compare"


def infer_slice(url, data, img_size, timeout):
    payload = {"data": data, "height": img_size, "width": img_size}
    t0 = time.time()
    resp = requests.post(f"{url}/infer", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["features"], time.time() - t0


def run_inference(url, dataset, img_size, max_slices, label):
    print(f"\nRunning inference on {label}...")
    print(f"  Endpoint: {url}")
    print(f"  Slices: {max_slices}")

    latencies = []
    errors = 0
    total = min(len(dataset), max_slices)

    t_start = time.time()

    for i in range(total):
        try:
            item = dataset[i]
            seismic = item["seismic"].astype(np.float32)
            if seismic.ndim == 3:
                seismic = seismic[:, :, 0]
            seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
            data = seismic.flatten().tolist()
            h, w = seismic.shape

            _, latency = infer_slice(url, data, h, TIMEOUT)
            latencies.append(latency)

            if (i + 1) % 500 == 0:
                avg = sum(latencies) / len(latencies)
                print(f"  [{i+1}/{total}] avg latency: {avg:.3f}s")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR on slice {i}: {e}")

    total_time = time.time() - t_start
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput = len(latencies) / total_time if total_time > 0 else 0

    return {
        "label": label,
        "url": url,
        "total_slices": total,
        "processed": len(latencies),
        "errors": errors,
        "total_time_s": round(total_time, 2),
        "avg_latency_s": round(avg_latency, 4),
        "p50_latency_s": round(sorted(latencies)[len(latencies)//2], 4) if latencies else 0,
        "p95_latency_s": round(sorted(latencies)[int(len(latencies)*0.95)], 4) if latencies else 0,
        "throughput_slices_per_sec": round(throughput, 2),
    }


def main():
    client = get_client()
    run = client.create_run(ml_repo="slb-pov", run_name="sfm-pytorch-vs-trt")
    print(f"Started run: {run.run_id}")

    try:
        run.log_params({
            "pytorch_url": PYTORCH_URL,
            "trt_url": TRT_URL,
            "dataset": HF_DATASET,
            "max_slices": MAX_SLICES,
            "img_size": IMG_SIZE,
        })

        # Load dataset
        print(f"Loading {HF_DATASET} from HuggingFace...")
        dataset = load_dataset(HF_DATASET, split=HF_SPLIT).with_format(type="numpy")
        print(f"Dataset size: {len(dataset)} slices, using {MAX_SLICES}")

        # Warmup both endpoints
        print("\nWarming up endpoints...")
        item = dataset[0]
        seismic = item["seismic"].astype(np.float32)
        if seismic.ndim == 3:
            seismic = seismic[:, :, 0]
        seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
        data = seismic.flatten().tolist()
        h, w = seismic.shape

        for url, label in [(PYTORCH_URL, "PyTorch"), (TRT_URL, "TensorRT")]:
            try:
                _, lat = infer_slice(url, data, h, TIMEOUT)
                print(f"  {label} warmup: {lat:.3f}s ✅")
            except Exception as e:
                print(f"  {label} warmup failed: {e} ❌")

        # Run inference on both
        pytorch_results = run_inference(PYTORCH_URL, dataset, IMG_SIZE, MAX_SLICES, "PyTorch FP32")
        trt_results     = run_inference(TRT_URL, dataset, IMG_SIZE, MAX_SLICES, "TensorRT FP16")

        # Compute speedup
        speedup_latency    = pytorch_results["avg_latency_s"] / trt_results["avg_latency_s"] if trt_results["avg_latency_s"] > 0 else 0
        speedup_throughput = trt_results["throughput_slices_per_sec"] / pytorch_results["throughput_slices_per_sec"] if pytorch_results["throughput_slices_per_sec"] > 0 else 0
        time_saved_s       = pytorch_results["total_time_s"] - trt_results["total_time_s"]

        comparison = {
            "pytorch": pytorch_results,
            "tensorrt": trt_results,
            "speedup": {
                "latency_speedup": round(speedup_latency, 2),
                "throughput_speedup": round(speedup_throughput, 2),
                "total_time_saved_s": round(time_saved_s, 2),
                "total_time_saved_min": round(time_saved_s / 60, 2),
            }
        }

        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS — PyTorch FP32 vs TensorRT FP16")
        print(f"{'='*60}")
        print(f"{'Metric':<30} {'PyTorch':>12} {'TensorRT':>12} {'Speedup':>10}")
        print("-" * 68)
        print(f"{'Avg latency (s)':<30} {pytorch_results['avg_latency_s']:>12.4f} {trt_results['avg_latency_s']:>12.4f} {speedup_latency:>9.1f}x")
        print(f"{'P95 latency (s)':<30} {pytorch_results['p95_latency_s']:>12.4f} {trt_results['p95_latency_s']:>12.4f}")
        print(f"{'Throughput (slices/s)':<30} {pytorch_results['throughput_slices_per_sec']:>12.2f} {trt_results['throughput_slices_per_sec']:>12.2f} {speedup_throughput:>9.1f}x")
        print(f"{'Total time (s)':<30} {pytorch_results['total_time_s']:>12.2f} {trt_results['total_time_s']:>12.2f}")
        print(f"{'Total time saved':<30} {'':>12} {time_saved_s:>11.1f}s")
        print(f"{'Slices processed':<30} {pytorch_results['processed']:>12} {trt_results['processed']:>12}")
        print(f"{'Errors':<30} {pytorch_results['errors']:>12} {trt_results['errors']:>12}")
        print(f"{'='*60}")

        # Log metrics
        run.log_metrics({
            "pytorch_avg_latency_s": pytorch_results["avg_latency_s"],
            "trt_avg_latency_s": trt_results["avg_latency_s"],
            "latency_speedup": speedup_latency,
            "throughput_speedup": speedup_throughput,
            "time_saved_s": time_saved_s,
        })

        # Save and log comparison artifact
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, "comparison.json")
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)

        av = run.log_artifact(
            name="sfm-pytorch-vs-trt-comparison",
            artifact_paths=[ArtifactPath(src=output_file, dest="comparison.json")],
            metadata={
                "latency_speedup": speedup_latency,
                "throughput_speedup": speedup_throughput,
                "slices_tested": MAX_SLICES,
            }
        )
        print(f"\nComparison saved → {av.fqn}")

    finally:
        run.end()


if __name__ == "__main__":
    main()
