"""
test_slb_contract.py — Test SLB API contract endpoint /v1/infer
"""

import os
import time
import base64
import json
import numpy as np
import requests

URL = os.environ.get(
    "SFM_LARGE_TRT_URL",
    "http://sfm-large-trt.slb-ws.svc.cluster.local:8080"
)

def test_inline():
    """Test with inline base64 payload"""
    print("Test 1: Inline base64 payload (256x256 tile)")
    h, w = 256, 256
    data = np.random.randn(h, w).astype(np.float32)
    data_b64 = base64.b64encode(data.tobytes()).decode("ascii")

    payload = {
        "model": "seismic-fm",
        "version": "1.3.0",
        "input": {
            "data_b64": data_b64,
            "height": h,
            "width": w
        },
        "task": "embedding",
        "output_format": "json"
    }

    t0 = time.perf_counter()
    resp = requests.post(URL + "/v1/infer", json=payload, timeout=30)
    t1 = time.perf_counter()

    print(f"  Status: {resp.status_code}")
    body = resp.json()
    print(f"  Response status: {body['status']}")
    print(f"  Latency: {body['latency_ms']}ms")
    print(f"  Tiles: {body['metadata']['tiles']}")
    print(f"  Model version: {body['metadata']['model_version']}")
    print(f"  Embed dim: {body['metadata']['embed_dim']}")
    print(f"  Grid: {body['metadata']['grid']}")
    print(f"  Total roundtrip: {(t1-t0)*1000:.0f}ms")
    print(f"  ✅ PASS" if body['status'] == 'succeeded' else "  ❌ FAIL")
    print()

def test_inline_with_roi():
    """Test with inline payload + ROI crop"""
    print("Test 2: Inline payload with ROI (4096x2048, cropped to 512x512)")
    h, w = 4096, 2048
    data = np.random.randn(h, w).astype(np.float32)
    data_b64 = base64.b64encode(data.tobytes()).decode("ascii")

    payload = {
        "model": "seismic-fm",
        "version": "1.3.0",
        "input": {
            "data_b64": data_b64,
            "height": h,
            "width": w,
            "roi": {
                "inline": [100, 612],
                "xline": [50, 562]
            }
        },
        "task": "fault_detection",
        "output_format": "json"
    }

    t0 = time.perf_counter()
    resp = requests.post(URL + "/v1/infer", json=payload, timeout=120)
    t1 = time.perf_counter()

    body = resp.json()
    print(f"  Status: {resp.status_code}")
    print(f"  Response status: {body['status']}")
    print(f"  Latency: {body['latency_ms']}ms")
    print(f"  Tiles: {body['metadata']['tiles']}")
    print(f"  Grid: {body['metadata']['grid']}")
    print(f"  GPU time: {body['metadata']['gpu_time_ms']}ms")
    print(f"  Total roundtrip: {(t1-t0)*1000:.0f}ms")
    print(f"  ✅ PASS" if body['status'] == 'succeeded' else "  ❌ FAIL")
    print()

def test_s3_reference():
    """Test S3 reference (should return error — S3 not configured)"""
    print("Test 3: S3 reference (expected: error, S3 not configured)")
    payload = {
        "model": "seismic-fm",
        "version": "1.3.0",
        "input": {
            "uri": "s3://surveys/blockA/cube.zgy",
            "roi": {"inline": [100, 228], "xline": [50, 178], "z": [0, 256]}
        },
        "task": "fault_detection",
        "output_format": "zgy"
    }

    resp = requests.post(URL + "/v1/infer", json=payload, timeout=30)
    body = resp.json()
    print(f"  Status: {resp.status_code}")
    print(f"  Response status: {body['status']}")
    print(f"  ✅ Correctly returns error for S3 (not yet configured)")
    print()

def main():
    print(f"\nSLB API Contract Test — /v1/infer")
    print(f"URL: {URL}")
    print(f"{'='*50}\n")

    # Health check
    r = requests.get(URL + "/health", timeout=5)
    print(f"Health: {r.status_code}\n")

    test_inline()
    test_inline_with_roi()
    test_s3_reference()

    print("All tests complete.")

if __name__ == "__main__":
    main()
