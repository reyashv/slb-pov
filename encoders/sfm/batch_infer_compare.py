"""
batch_infer_compare.py

Compares PyTorch vs TensorRT FP16 on 3500 interpolation slices.
Tests both sequential (batch_size=1) and concurrent (batch_size=8) modes.
"""

import os
import json
import time
import numpy as np
import requests
import concurrent.futures
from truefoundry.ml import get_client, ArtifactPath
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
PYTORCH_URL  = os.environ.get("PYTORCH_URL", "http://sfm-base.slb-ws.svc.cluster.local:8000")
TRT_URL      = os.environ.get("TRT_URL", "http://sfm-base-trt.slb-ws.svc.cluster.local:8000")
HF_DATASET   = os.environ.get("HF_DATASET", "porestar/seismicfoundationmodel-interpolation")
HF_SPLIT     = os.environ.get("HF_SPLIT", "train")
IMG_SIZE     = int(os.environ.get("SFM_IMG_SIZE", "224"))
TIMEOUT      = int(os.environ.get("REQUEST_TIMEOUT", "60"))
MAX_SLICES   = int(os.environ.get("MAX_SLICES", "500"))  # 500 slices per test mode
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "8"))    # concurrent requests
OUTPUT_DIR   = "/tmp/sfm-compare"


def infer_slice(url, data, img_size, timeout):
    payload = {"data": data, "height": img_size, "width": img_size}
    t0 = time.time()
    resp = requests.post(f"{url}/infer", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["features"], time.time() - t0


def prepare_slices(dataset, img_size, max_slices):
    slices = []
    for i in range(min(len(dataset), max_slices)):
        item = dataset[i]
        seismic = item["seismic"].astype(np.float32)
        if seismic.ndim == 3:
            seismic = seismic[:, :, 0]
        seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
        h, w = seismic.shape
        slices.append((seismic.flatten().tolist(), h, w))
    return slices


def run_sequential(url, slices, label):
    """Send slices one at a time."""
    print(f"\n  Sequential inference on {label} ({len(slices)} slices)...")
    latencies = []
    errors = 0
    t_start = time.time()

    for i, (data, h, w) in enumerate(slices):
        try:
            _, latency = infer_slice(url, data, h, TIMEOUT)
            latencies.append(latency)
        except Exception as e:
            errors += 1

    total_time = time.time() - t_start
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    throughput = len(latencies) / total_time if total_time > 0 else 0

    print(f"  Done — avg: {avg_lat:.4f}s, throughput: {throughput:.2f}/s, errors: {errors}")
    return {
        "mode": "sequential",
        "label": label,
        "slices": len(slices),
        "processed": len(latencies),
        "errors": errors,
        "total_time_s": round(total_time, 2),
        "avg_latency_s": round(avg_lat, 4),
        "p95_latency_s": round(sorted(latencies)[int(len(latencies)*0.95)], 4) if latencies else 0,
        "throughput_slices_per_sec": round(throughput, 2),
    }


def run_concurrent(url, slices, batch_size, label):
    """Send batch_size slices concurrently using ThreadPoolExecutor."""
    print(f"\n  Concurrent inference on {label} (batch_size={batch_size}, {len(slices)} slices)...")
    latencies = []
    errors = 0
    t_start = time.time()

    def infer_one(args):
        data, h, w = args
        try:
            _, latency = infer_slice(url, data, h, TIMEOUT)
            return latency, True
        except Exception:
            return 0, False

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(infer_one, s) for s in slices]
        for f in concurrent.futures.as_completed(futures):
            latency, success = f.result()
            if success:
                latencies.append(latency)
            else:
                errors += 1

    total_time = time.time() - t_start
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    throughput = len(latencies) / total_time if total_time > 0 else 0

    print(f"  Done — avg: {avg_lat:.4f}s, throughput: {throughput:.2f}/s, errors: {errors}")
    return {
        "mode": f"concurrent_{batch_size}",
        "label": label,
        "slices": len(slices),
        "processed": len(latencies),
        "errors": errors,
        "total_time_s": round(total_time, 2),
        "avg_latency_s": round(avg_lat, 4),
        "p95_latency_s": round(sorted(latencies)[int(len(latencies)*0.95)], 4) if latencies else 0,
        "throughput_slices_per_sec": round(throughput, 2),
    }


def print_comparison(mode_label, pytorch, trt):
    speedup_lat = pytorch["avg_latency_s"] / trt["avg_latency_s"] if trt["avg_latency_s"] > 0 else 0
    speedup_thr = trt["throughput_slices_per_sec"] / pytorch["throughput_slices_per_sec"] if pytorch["throughput_slices_per_sec"] > 0 else 0
    print(f"\n  {'Metric':<30} {'PyTorch':>12} {'TensorRT':>12} {'Speedup':>10}")
    print(f"  {'-'*68}")
    print(f"  {'Avg latency (s)':<30} {pytorch['avg_latency_s']:>12.4f} {trt['avg_latency_s']:>12.4f} {speedup_lat:>9.1f}x")
    print(f"  {'P95 latency (s)':<30} {pytorch['p95_latency_s']:>12.4f} {trt['p95_latency_s']:>12.4f}")
    print(f"  {'Throughput (slices/s)':<30} {pytorch['throughput_slices_per_sec']:>12.2f} {trt['throughput_slices_per_sec']:>12.2f} {speedup_thr:>9.1f}x")
    print(f"  {'Total time (s)':<30} {pytorch['total_time_s']:>12.2f} {trt['total_time_s']:>12.2f}")
    print(f"  {'Errors':<30} {pytorch['errors']:>12} {trt['errors']:>12}")
    return speedup_lat, speedup_thr


def main():
    client = get_client()
    run = client.create_run(ml_repo="slb-pov", run_name="sfm-pytorch-vs-trt-batch")
    print(f"Started run: {run.run_id}")

    try:
        run.log_params({
            "pytorch_url": PYTORCH_URL,
            "trt_url": TRT_URL,
            "dataset": HF_DATASET,
            "max_slices": MAX_SLICES,
            "batch_size": BATCH_SIZE,
        })

        # Load dataset
        print(f"Loading {HF_DATASET} from HuggingFace...")
        dataset = load_dataset(HF_DATASET, split=HF_SPLIT).with_format(type="numpy")
        slices = prepare_slices(dataset, IMG_SIZE, MAX_SLICES)
        print(f"Prepared {len(slices)} slices")

        # Warmup
        print("\nWarming up both endpoints...")
        data, h, w = slices[0]
        for url, label in [(PYTORCH_URL, "PyTorch"), (TRT_URL, "TensorRT")]:
            try:
                _, lat = infer_slice(url, data, h, TIMEOUT)
                print(f"  {label}: {lat:.3f}s ✅")
            except Exception as e:
                print(f"  {label}: FAILED — {e} ❌")

        results = {}

        # ── Test 1: Sequential (batch_size=1) ──
        print(f"\n{'='*65}")
        print(f"TEST 1 — Sequential (1 slice at a time)")
        print(f"{'='*65}")
        pt_seq  = run_sequential(PYTORCH_URL, slices, "PyTorch FP32")
        trt_seq = run_sequential(TRT_URL, slices, "TensorRT FP16")
        sp_lat1, sp_thr1 = print_comparison("Sequential", pt_seq, trt_seq)
        results["sequential"] = {"pytorch": pt_seq, "tensorrt": trt_seq,
                                  "speedup_latency": round(sp_lat1, 2),
                                  "speedup_throughput": round(sp_thr1, 2)}

        run.log_metrics({
            "seq_pytorch_avg_lat": pt_seq["avg_latency_s"],
            "seq_trt_avg_lat": trt_seq["avg_latency_s"],
            "seq_latency_speedup": sp_lat1,
            "seq_throughput_speedup": sp_thr1,
        })

        # ── Test 2: Concurrent (batch_size=8) ──
        print(f"\n{'='*65}")
        print(f"TEST 2 — Concurrent (batch_size={BATCH_SIZE} simultaneous requests)")
        print(f"{'='*65}")
        pt_con  = run_concurrent(PYTORCH_URL, slices, BATCH_SIZE, "PyTorch FP32")
        trt_con = run_concurrent(TRT_URL, slices, BATCH_SIZE, "TensorRT FP16")
        sp_lat2, sp_thr2 = print_comparison("Concurrent", pt_con, trt_con)
        results["concurrent"] = {"pytorch": pt_con, "tensorrt": trt_con,
                                  "speedup_latency": round(sp_lat2, 2),
                                  "speedup_throughput": round(sp_thr2, 2)}

        run.log_metrics({
            "con_pytorch_avg_lat": pt_con["avg_latency_s"],
            "con_trt_avg_lat": trt_con["avg_latency_s"],
            "con_latency_speedup": sp_lat2,
            "con_throughput_speedup": sp_thr2,
        })

        # ── Final summary ──
        print(f"\n{'='*65}")
        print(f"FINAL SUMMARY")
        print(f"{'='*65}")
        print(f"  Sequential   — Latency speedup: {sp_lat1:.1f}x | Throughput speedup: {sp_thr1:.1f}x")
        print(f"  Concurrent 8 — Latency speedup: {sp_lat2:.1f}x | Throughput speedup: {sp_thr2:.1f}x")
        print(f"{'='*65}")

        # Save artifact
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, "comparison.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        av = run.log_artifact(
            name="sfm-pytorch-vs-trt-comparison",
            artifact_paths=[ArtifactPath(src=output_file, dest="comparison.json")],
            metadata={
                "seq_latency_speedup": round(sp_lat1, 2),
                "con_latency_speedup": round(sp_lat2, 2),
                "slices_tested": MAX_SLICES,
                "batch_size": BATCH_SIZE,
            }
        )
        print(f"\nComparison saved → {av.fqn}")

    finally:
        run.end()


if __name__ == "__main__":
    main()
