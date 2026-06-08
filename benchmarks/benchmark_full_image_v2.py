"""
benchmark_full_image_v2.py — Server-side tiling benchmark (base64 encoding)
Tests SLB target: 4K x 2K image p50 ≤ 3000ms, p95 ≤ 6000ms

Sends ONE request with the full image (base64 encoded) to /infer_full_image.
Server tiles internally and runs GPU inference in batches.
"""

import os
import time
import json
import base64
import argparse
import numpy as np
import requests
import statistics

FULL_IMAGE_H = 4096
FULL_IMAGE_W = 2048

TARGETS = {"p50_ms": 3000, "p95_ms": 6000}

URL = os.environ.get(
    "SFM_LARGE_TRT_URL",
    "http://sfm-large-trt.slb-ws.svc.cluster.local:8080"
)
ENDPOINT = "/infer_full_image"


def make_payload(h, w):
    """Generate a random full image as base64 encoded float32."""
    data = np.random.randn(h, w).astype(np.float32)
    data_b64 = base64.b64encode(data.tobytes()).decode("ascii")
    return {"data_b64": data_b64, "height": h, "width": w}


def run_request(session, url, endpoint, payload):
    t0 = time.perf_counter()
    resp = session.post(url + endpoint, json=payload, timeout=120)
    t1 = time.perf_counter()
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    return (t1 - t0) * 1000.0, body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default="full_image_v2_results.json")
    args = parser.parse_args()

    raw_size_mb = FULL_IMAGE_H * FULL_IMAGE_W * 4 / (1024 * 1024)
    b64_size_mb = raw_size_mb * 4 / 3

    print(f"\nSLB Full Image Benchmark (server-side tiling, base64)")
    print(f"Image: {FULL_IMAGE_H}x{FULL_IMAGE_W}")
    print(f"Payload: {raw_size_mb:.0f}MB raw → {b64_size_mb:.0f}MB base64")
    print(f"Target: p50 ≤ {TARGETS['p50_ms']}ms, p95 ≤ {TARGETS['p95_ms']}ms")
    print(f"URL: {URL}{ENDPOINT}")
    print(f"Warmup: {args.warmup} | Timed: {args.n}")

    session = requests.Session()

    # Health check
    try:
        r = session.get(URL + "/health", timeout=5)
        print(f"Health: {r.status_code}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # Build payload once
    print("Building base64 payload...", end="", flush=True)
    payload = make_payload(FULL_IMAGE_H, FULL_IMAGE_W)
    print(f" done ({len(payload['data_b64'])} chars)")

    # Single test
    print("Testing single request...", end="", flush=True)
    try:
        ms, body = run_request(session, URL, ENDPOINT, payload)
        print(f" OK — {ms:.0f}ms, {body['n_tiles']} tiles, GPU: {body['gpu_time_ms']}ms")
    except Exception as e:
        print(f" FAILED: {e}")
        return

    # Warmup
    print(f"Warming up ({args.warmup} requests)...", end="", flush=True)
    for _ in range(args.warmup):
        try:
            run_request(session, URL, ENDPOINT, payload)
        except Exception:
            pass
    print(" done")

    # Timed runs
    latencies = []
    gpu_times = []
    errors = 0
    print(f"Running {args.n} timed requests...", end="", flush=True)
    for i in range(args.n):
        try:
            ms, body = run_request(session, URL, ENDPOINT, payload)
            latencies.append(ms)
            gpu_times.append(body.get("gpu_time_ms", 0))
        except Exception as e:
            errors += 1
            print(f"\n  Error on run {i}: {e}")
        if (i + 1) % 5 == 0:
            print(f" {i+1}", end="", flush=True)
    print(" done")

    if not latencies:
        print("ERROR: All runs failed")
        return

    latencies.sort()
    gpu_times.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = statistics.mean(latencies)
    gpu_p50 = statistics.median(gpu_times)

    p50_pass = p50 <= TARGETS["p50_ms"]
    p95_pass = p95 <= TARGETS["p95_ms"]

    print(f"\n{'='*50}")
    print(f"  RESULTS — Full Image ({FULL_IMAGE_H}x{FULL_IMAGE_W})")
    print(f"  Server-side tiling, base64 payload")
    print(f"{'='*50}")
    print(f"  Total (end-to-end):")
    print(f"    p50:  {p50:.0f} ms  {'✅ PASS' if p50_pass else '❌ FAIL'} (target ≤ {TARGETS['p50_ms']}ms)")
    print(f"    p95:  {p95:.0f} ms  {'✅ PASS' if p95_pass else '❌ FAIL'} (target ≤ {TARGETS['p95_ms']}ms)")
    print(f"    p99:  {p99:.0f} ms")
    print(f"    mean: {mean:.0f} ms")
    print(f"  GPU only:")
    print(f"    p50:  {gpu_p50:.0f} ms")
    print(f"  Errors: {errors}/{args.n}")

    results = {
        "benchmark": "full_image_server_side_tiling_b64",
        "image_size": f"{FULL_IMAGE_H}x{FULL_IMAGE_W}",
        "payload_mb": round(b64_size_mb, 1),
        "targets": TARGETS,
        "p50_ms": round(p50, 0),
        "p95_ms": round(p95, 0),
        "p99_ms": round(p99, 0),
        "mean_ms": round(mean, 0),
        "gpu_p50_ms": round(gpu_p50, 0),
        "p50_pass": p50_pass,
        "p95_pass": p95_pass,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
