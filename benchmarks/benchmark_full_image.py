"""
benchmark_full_image.py — 4K x 2K full image benchmark
Tests SLB success metric: 4096x2048 image p50 ≤ 3000ms, p95 ≤ 6000ms

Strategy:
  - Tile 4096x2048 image into non-overlapping patches of model input size
  - sfm-large / sfm-large-trt: 224x224 patches → ~18x9 = 162 tiles
  - sfm-large-512:              512x512 patches → ~8x4  = 32 tiles
  - Two modes:
      sequential: tiles sent one by one (simulates single-threaded client)
      parallel:   tiles sent concurrently (simulates production batch client)
  - Total latency = time from first tile sent to last result received

Run from inside cluster for clean measurement.

Usage:
  python benchmark_full_image.py                       # all models, sequential + parallel
  python benchmark_full_image.py --model sfm-large-512 --mode parallel
  python benchmark_full_image.py --n 50               # repeat 50 times for stable p50/p95

Environment variables:
  SFM_LARGE_URL      default: http://sfm-large.slb-ws.svc.cluster.local:8000
  SFM_LARGE_TRT_URL  default: http://sfm-large-trt.slb-ws.svc.cluster.local:8000
  SFM_LARGE_512_URL  default: http://sfm-large-512.slb-ws.svc.cluster.local:8000
  MAX_WORKERS        default: 16 (thread pool size for parallel mode)
"""

import os
import time
import argparse
import json
import math
import numpy as np
import requests
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────

FULL_IMAGE_H = 4096   # SLB spec: 4K x 2K
FULL_IMAGE_W = 2048

MODELS = {
    "sfm-large": {
        "url": os.environ.get("SFM_LARGE_URL", "http://sfm-large.slb-ws.svc.cluster.local:8000"),
        "endpoint": "/infer",
        "img_size": 224,
        "type": "fastapi",
        "description": "FastAPI FP32 224x224",
    },
    "sfm-large-trt": {
        "url": os.environ.get("SFM_LARGE_TRT_URL", "http://sfm-large-trt.slb-ws.svc.cluster.local:8000"),
        "endpoint": "/v2/models/sfm_large/infer",
        "img_size": 224,
        "type": "triton",
        "description": "TRT FP16 + PyTriton 224x224",
    },
    "sfm-large-512": {
        "url": os.environ.get("SFM_LARGE_512_URL", "http://sfm-large-512.slb-ws.svc.cluster.local:8000"),
        "endpoint": "/infer",
        "img_size": 512,
        "type": "fastapi",
        "description": "FastAPI FP32 512x512",
    },
}

TARGETS = {
    "p50_ms": 3000,
    "p95_ms": 6000,
}

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "16"))


# ── Tiling ────────────────────────────────────────────────────────────────────

def compute_tiles(image_h: int, image_w: int, tile_size: int) -> list:
    """
    Non-overlapping tile grid covering the full image.
    Last tile in each row/col may be a partial tile — we pad it to tile_size.
    Returns list of (row, col, h_start, h_end, w_start, w_end).
    """
    tiles = []
    n_rows = math.ceil(image_h / tile_size)
    n_cols = math.ceil(image_w / tile_size)
    for r in range(n_rows):
        for c in range(n_cols):
            h_start = r * tile_size
            h_end = min(h_start + tile_size, image_h)
            w_start = c * tile_size
            w_end = min(w_start + tile_size, image_w)
            tiles.append((r, c, h_start, h_end, w_start, w_end))
    return tiles


# ── Payload builders ──────────────────────────────────────────────────────────

def make_fastapi_payload(tile_size: int) -> dict:
    data = np.random.randn(tile_size * tile_size).astype(np.float32).tolist()
    return {"data": data, "height": tile_size, "width": tile_size}


def make_triton_payload(tile_size: int) -> dict:
    data = np.random.randn(1, 1, tile_size, tile_size).astype(np.float32)
    return {
        "inputs": [
            {
                "name": "INPUT",
                "shape": [1, 1, tile_size, tile_size],
                "datatype": "FP32",
                "data": data.flatten().tolist(),
            }
        ],
        "outputs": [{"name": "OUTPUT"}],
    }


# ── Request ───────────────────────────────────────────────────────────────────

def send_tile(session: requests.Session, url: str, endpoint: str, payload: dict) -> float:
    """Send one tile, return latency ms."""
    t0 = time.perf_counter()
    resp = session.post(url + endpoint, json=payload, timeout=30)
    t1 = time.perf_counter()
    resp.raise_for_status()
    return (t1 - t0) * 1000.0


# ── Sequential mode ───────────────────────────────────────────────────────────

def run_sequential(session, url, endpoint, payload, n_tiles) -> float:
    """Send all tiles one by one. Returns total wall time ms."""
    t_start = time.perf_counter()
    for _ in range(n_tiles):
        send_tile(session, url, endpoint, payload)
    t_end = time.perf_counter()
    return (t_end - t_start) * 1000.0


# ── Parallel mode ─────────────────────────────────────────────────────────────

def run_parallel(url, endpoint, payload, n_tiles, max_workers) -> float:
    """Send all tiles concurrently. Returns total wall time ms."""
    def _send(_):
        s = requests.Session()
        return send_tile(s, url, endpoint, payload)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_send, i) for i in range(n_tiles)]
        for f in as_completed(futures):
            f.result()  # raise if any tile failed
    t_end = time.perf_counter()
    return (t_end - t_start) * 1000.0


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_model(name: str, config: dict, n: int, modes: list) -> dict:
    img_size = config["img_size"]
    tiles = compute_tiles(FULL_IMAGE_H, FULL_IMAGE_W, img_size)
    n_tiles = len(tiles)

    print(f"\n{'='*60}")
    print(f"  {name} — {config['description']}")
    print(f"  Image: {FULL_IMAGE_H}x{FULL_IMAGE_W}  Tile: {img_size}x{img_size}  Tiles: {n_tiles}")
    print(f"  Modes: {modes}  Repeats: {n}  Workers (parallel): {MAX_WORKERS}")
    print(f"{'='*60}")

    if config["type"] == "fastapi":
        payload = make_fastapi_payload(img_size)
    else:
        payload = make_triton_payload(img_size)

    session = requests.Session()

    # Health check
    try:
        health_ep = "/v2/health/ready" if config["type"] == "triton" else "/health"
        r = session.get(config["url"] + health_ep, timeout=5)
        print(f"  Health: {r.status_code}")
    except Exception as e:
        print(f"  Health check failed: {e} — skipping")
        return None

    result = {
        "model": name,
        "description": config["description"],
        "img_size": img_size,
        "image_size": f"{FULL_IMAGE_H}x{FULL_IMAGE_W}",
        "n_tiles": n_tiles,
        "n_repeats": n,
    }

    for mode in modes:
        print(f"\n  Mode: {mode}")

        # Warmup — 3 full image runs
        print(f"  Warming up (3 full images)...", end="", flush=True)
        for _ in range(3):
            try:
                if mode == "sequential":
                    run_sequential(session, config["url"], config["endpoint"], payload, n_tiles)
                else:
                    run_parallel(config["url"], config["endpoint"], payload, n_tiles, MAX_WORKERS)
            except Exception:
                pass
        print(" done")

        # Timed runs
        times_ms = []
        errors = 0
        print(f"  Running {n} full image requests...", end="", flush=True)
        for i in range(n):
            try:
                if mode == "sequential":
                    ms = run_sequential(session, config["url"], config["endpoint"], payload, n_tiles)
                else:
                    ms = run_parallel(config["url"], config["endpoint"], payload, n_tiles, MAX_WORKERS)
                times_ms.append(ms)
            except Exception as e:
                errors += 1
            if (i + 1) % 10 == 0:
                print(f" {i+1}", end="", flush=True)
        print(" done")

        if not times_ms:
            print(f"  ERROR: All runs failed")
            continue

        times_ms.sort()
        p50 = statistics.median(times_ms)
        p95 = times_ms[int(len(times_ms) * 0.95)]
        p99 = times_ms[int(len(times_ms) * 0.99)]
        mean = statistics.mean(times_ms)

        p50_pass = p50 <= TARGETS["p50_ms"]
        p95_pass = p95 <= TARGETS["p95_ms"]

        result[mode] = {
            "p50_ms": round(p50, 0),
            "p95_ms": round(p95, 0),
            "p99_ms": round(p99, 0),
            "mean_ms": round(mean, 0),
            "min_ms": round(min(times_ms), 0),
            "max_ms": round(max(times_ms), 0),
            "errors": errors,
            "p50_pass": p50_pass,
            "p95_pass": p95_pass,
            # Derived: per-tile avg
            "avg_tile_ms": round(mean / n_tiles, 1),
        }

        print(f"\n  Results ({mode}):")
        print(f"    p50:  {p50:.0f} ms  {'✅ PASS' if p50_pass else '❌ FAIL'} (target ≤ {TARGETS['p50_ms']}ms)")
        print(f"    p95:  {p95:.0f} ms  {'✅ PASS' if p95_pass else '❌ FAIL'} (target ≤ {TARGETS['p95_ms']}ms)")
        print(f"    p99:  {p99:.0f} ms")
        print(f"    mean: {mean:.0f} ms  (avg {mean/n_tiles:.1f} ms/tile × {n_tiles} tiles)")
        print(f"    errors: {errors}/{n}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SLB SFM 4K x 2K full image benchmark")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    parser.add_argument("--mode", choices=["sequential", "parallel", "both"], default="both")
    parser.add_argument("--n", type=int, default=30,
                        help="Number of full-image timed runs (default: 30)")
    parser.add_argument("--output", default="full_image_benchmark_results.json")
    args = parser.parse_args()

    modes = ["sequential", "parallel"] if args.mode == "both" else [args.mode]
    models_to_test = MODELS if args.model == "all" else {args.model: MODELS[args.model]}

    print(f"\nSLB SFM Full Image Benchmark — {FULL_IMAGE_H}x{FULL_IMAGE_W}")
    print(f"Target: p50 ≤ {TARGETS['p50_ms']}ms, p95 ≤ {TARGETS['p95_ms']}ms")
    print(f"Models: {list(models_to_test.keys())}")

    # Print tile counts upfront so it's clear
    print(f"\nTile breakdown:")
    for name, cfg in models_to_test.items():
        tiles = compute_tiles(FULL_IMAGE_H, FULL_IMAGE_W, cfg["img_size"])
        rows = math.ceil(FULL_IMAGE_H / cfg["img_size"])
        cols = math.ceil(FULL_IMAGE_W / cfg["img_size"])
        print(f"  {name}: {cfg['img_size']}x{cfg['img_size']} tiles → {rows}x{cols} = {len(tiles)} tiles")

    all_results = []
    for name, config in models_to_test.items():
        r = benchmark_model(name, config, args.n, modes)
        if r:
            all_results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — Full Image ({FULL_IMAGE_H}x{FULL_IMAGE_W})")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n  {r['model']} ({r['n_tiles']} tiles):")
        for mode in modes:
            if mode in r:
                m = r[mode]
                p50_s = "✅" if m["p50_pass"] else "❌"
                p95_s = "✅" if m["p95_pass"] else "❌"
                print(f"    {mode:<12} p50={m['p50_ms']:.0f}ms {p50_s}  p95={m['p95_ms']:.0f}ms {p95_s}  (avg {m['avg_tile_ms']}ms/tile)")

    with open(args.output, "w") as f:
        json.dump({
            "benchmark": "full_image_latency",
            "image_size": f"{FULL_IMAGE_H}x{FULL_IMAGE_W}",
            "targets": TARGETS,
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
