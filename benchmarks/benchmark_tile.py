"""
benchmark_tile.py — Single tile latency benchmark
Tests SLB success metric: 256x256 tile p50 ≤ 400ms, p95 ≤ 1000ms

Tests three models:
  - sfm-large       (FastAPI, FP32, 224x224 input)
  - sfm-large-trt   (TRT FP16 + PyTriton, 224x224 input)
  - sfm-large-512   (FastAPI, FP32, 512x512 input — closest to SLB's 256x256 tile spec)

Run from inside the cluster (pod-to-pod) for clean GPU-only measurement.
Or set USE_EXTERNAL=1 and set *_URL env vars to use external endpoints.

Usage:
  python benchmark_tile.py                        # all models, 100 warmup + 500 timed requests
  python benchmark_tile.py --model sfm-large-trt  # single model
  python benchmark_tile.py --n 1000               # more samples for tighter p95

Environment variables:
  SFM_LARGE_URL      default: http://sfm-large.slb-ws.svc.cluster.local:8000
  SFM_LARGE_TRT_URL  default: http://sfm-large-trt.slb-ws.svc.cluster.local:8000
  SFM_LARGE_512_URL  default: http://sfm-large-512.slb-ws.svc.cluster.local:8000
  WARMUP_N           default: 20
  N                  default: 500
"""

import os
import time
import argparse
import json
import numpy as np
import requests
import statistics

# ── URLs ──────────────────────────────────────────────────────────────────────

MODELS = {
    "sfm-large": {
        "url": os.environ.get("SFM_LARGE_URL", "http://sfm-large.slb-ws.svc.cluster.local:8000"),
        "endpoint": "/infer",
        "img_size": 224,
        "type": "fastapi",          # payload format
        "description": "FastAPI FP32 baseline",
    },
    "sfm-large-trt": {
        "url": os.environ.get("SFM_LARGE_TRT_URL", "http://sfm-large-trt.slb-ws.svc.cluster.local:8000"),
        "endpoint": "/v2/models/sfm_large/infer",
        "img_size": 224,
        "type": "triton",           # payload format
        "description": "TRT FP16 + PyTriton (recommended)",
    },
    "sfm-large-512": {
        "url": os.environ.get("SFM_LARGE_512_URL", "http://sfm-large-512.slb-ws.svc.cluster.local:8000"),
        "endpoint": "/infer",
        "img_size": 512,
        "type": "fastapi",
        "description": "FastAPI FP32 512x512 (closest to SLB 256x256 tile spec)",
    },
}

# SLB success metric targets
TARGETS = {
    "p50_ms": 400,
    "p95_ms": 1000,
}


# ── Payload builders ──────────────────────────────────────────────────────────

def make_fastapi_payload(img_size: int) -> dict:
    """
    Standard FastAPI /infer payload.
    Tile is 256x256 from SLB spec — we send it as-is for 512 model,
    pad/resize to 224 for the 224 model (server handles this or we pad here).
    For benchmarking we just send the correct model input size.
    """
    data = np.random.randn(img_size * img_size).astype(np.float32).tolist()
    return {"data": data, "height": img_size, "width": img_size}


def make_triton_payload(img_size: int) -> dict:
    """Triton HTTP REST /v2/models/{model}/infer payload."""
    data = np.random.randn(1, 1, img_size, img_size).astype(np.float32)
    return {
        "inputs": [
            {
                "name": "INPUT",
                "shape": [1, 1, img_size, img_size],
                "datatype": "FP32",
                "data": data.flatten().tolist(),
            }
        ],
        "outputs": [{"name": "OUTPUT"}],
    }


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_single_request(session: requests.Session, url: str, endpoint: str, payload: dict) -> float:
    """Returns latency in milliseconds."""
    t0 = time.perf_counter()
    resp = session.post(url + endpoint, json=payload, timeout=30)
    t1 = time.perf_counter()
    resp.raise_for_status()
    return (t1 - t0) * 1000.0


def benchmark_model(name: str, config: dict, n: int, warmup_n: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  {name} — {config['description']}")
    print(f"  URL: {config['url']}")
    print(f"  Input size: {config['img_size']}x{config['img_size']}")
    print(f"  Warmup: {warmup_n} requests | Timed: {n} requests")
    print(f"{'='*60}")

    img_size = config["img_size"]
    if config["type"] == "fastapi":
        payload = make_fastapi_payload(img_size)
    else:
        payload = make_triton_payload(img_size)

    session = requests.Session()

    # Health check
    try:
        if config["type"] == "triton":
            health_url = config["url"] + "/v2/health/ready"
        else:
            health_url = config["url"] + "/health"
        r = session.get(health_url, timeout=5)
        print(f"  Health: {r.status_code}")
    except Exception as e:
        print(f"  Health check failed: {e} — skipping model")
        return None

    # Warmup
    print(f"  Warming up ({warmup_n} requests)...", end="", flush=True)
    for _ in range(warmup_n):
        try:
            run_single_request(session, config["url"], config["endpoint"], payload)
        except Exception:
            pass
    print(" done")

    # Timed run
    latencies = []
    errors = 0
    print(f"  Running {n} timed requests...", end="", flush=True)
    for i in range(n):
        try:
            ms = run_single_request(session, config["url"], config["endpoint"], payload)
            latencies.append(ms)
        except Exception as e:
            errors += 1
        if (i + 1) % 100 == 0:
            print(f" {i+1}", end="", flush=True)
    print(" done")

    if not latencies:
        print("  ERROR: All requests failed")
        return None

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = statistics.mean(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)

    result = {
        "model": name,
        "description": config["description"],
        "img_size": img_size,
        "n_requests": n,
        "errors": errors,
        "error_rate_pct": round(errors / n * 100, 2),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "mean_ms": round(mean, 1),
        "min_ms": round(min_lat, 1),
        "max_ms": round(max_lat, 1),
        "p50_pass": p50 <= TARGETS["p50_ms"],
        "p95_pass": p95 <= TARGETS["p95_ms"],
    }

    # Print results
    print(f"\n  Results:")
    print(f"    p50:  {p50:.1f} ms  {'✅ PASS' if result['p50_pass'] else '❌ FAIL'} (target ≤ {TARGETS['p50_ms']}ms)")
    print(f"    p95:  {p95:.1f} ms  {'✅ PASS' if result['p95_pass'] else '❌ FAIL'} (target ≤ {TARGETS['p95_ms']}ms)")
    print(f"    p99:  {p99:.1f} ms")
    print(f"    mean: {mean:.1f} ms")
    print(f"    min:  {min_lat:.1f} ms | max: {max_lat:.1f} ms")
    print(f"    errors: {errors}/{n} ({result['error_rate_pct']}%)")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SLB SFM tile latency benchmark")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                        help="Which model to benchmark (default: all)")
    parser.add_argument("--n", type=int, default=int(os.environ.get("N", "500")),
                        help="Number of timed requests (default: 500)")
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP_N", "20")),
                        help="Number of warmup requests (default: 20)")
    parser.add_argument("--output", type=str, default="tile_benchmark_results.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    models_to_test = MODELS if args.model == "all" else {args.model: MODELS[args.model]}

    print(f"\nSLB SFM Tile Benchmark")
    print(f"Target: p50 ≤ {TARGETS['p50_ms']}ms, p95 ≤ {TARGETS['p95_ms']}ms (single 256x256 tile, 1 user)")
    print(f"Models to test: {list(models_to_test.keys())}")

    results = []
    for name, config in models_to_test.items():
        result = benchmark_model(name, config, args.n, args.warmup)
        if result:
            results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY — Single Tile Latency (1 user, sequential)")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'p50':>8} {'p95':>8} {'p99':>8} {'p50':>8} {'p95':>8}")
    print(f"  {'':20} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'target':>8} {'target':>8}")
    print(f"  {'-'*64}")
    for r in results:
        p50_status = "✅" if r["p50_pass"] else "❌"
        p95_status = "✅" if r["p95_pass"] else "❌"
        print(f"  {r['model']:<20} {r['p50_ms']:>8} {r['p95_ms']:>8} {r['p99_ms']:>8} {p50_status:>8} {p95_status:>8}")

    # Save results
    with open(args.output, "w") as f:
        json.dump({
            "benchmark": "single_tile_latency",
            "targets": TARGETS,
            "results": results
        }, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
