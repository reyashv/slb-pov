"""
test_async_inference.py — Test async queue inference and storage-based input
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


def test_async_inline():
    """Test async inference with inline base64 payload."""
    print("Test 1: Async inference with inline payload (512x512)")
    h, w = 512, 512
    data = np.random.randn(h, w).astype(np.float32)
    data_b64 = base64.b64encode(data.tobytes()).decode("ascii")

    # Submit job
    payload = {
        "data_b64": data_b64,
        "height": h,
        "width": w,
        "task": "embedding",
    }
    resp = requests.post(URL + "/v1/async/infer", json=payload, timeout=30)
    body = resp.json()
    job_id = body["job_id"]
    print(f"  Submitted: job_id={job_id}, status={body['status']}")
    print(f"  Message: {body['message']}")

    # Poll for result
    for i in range(30):
        time.sleep(0.5)
        status_resp = requests.get(URL + f"/v1/async/status/{job_id}", timeout=10)
        status = status_resp.json()
        if status["status"] in ("succeeded", "failed"):
            break

    print(f"  Final status: {status['status']}")
    if status["status"] == "succeeded":
        print(f"  Latency: {status['latency_ms']}ms")
        print(f"  Tiles: {status['n_tiles']}")
        print(f"  Result present: {'yes' if status.get('result_b64') else 'no'}")
        print(f"  ✅ PASS")
    else:
        print(f"  Error: {status.get('error')}")
        print(f"  ❌ FAIL")
    print()


def test_async_large():
    """Test async inference with large payload (4K x 2K)."""
    print("Test 2: Async inference with large payload (4096x2048)")
    h, w = 4096, 2048
    data = np.random.randn(h, w).astype(np.float32)
    data_b64 = base64.b64encode(data.tobytes()).decode("ascii")

    payload = {
        "data_b64": data_b64,
        "height": h,
        "width": w,
        "task": "fault_detection",
    }

    t0 = time.perf_counter()
    resp = requests.post(URL + "/v1/async/infer", json=payload, timeout=120)
    body = resp.json()
    job_id = body["job_id"]
    submit_time = (time.perf_counter() - t0) * 1000
    print(f"  Submitted: job_id={job_id} in {submit_time:.0f}ms")

    # Poll
    for i in range(60):
        time.sleep(0.5)
        status_resp = requests.get(URL + f"/v1/async/status/{job_id}", timeout=10)
        status = status_resp.json()
        if status["status"] in ("succeeded", "failed"):
            break

    total_time = (time.perf_counter() - t0) * 1000
    print(f"  Final status: {status['status']}")
    if status["status"] == "succeeded":
        print(f"  Server latency: {status['latency_ms']}ms")
        print(f"  Total (submit + poll): {total_time:.0f}ms")
        print(f"  Tiles: {status['n_tiles']}")
        print(f"  ✅ PASS")
    else:
        print(f"  Error: {status.get('error')}")
        print(f"  ❌ FAIL")
    print()


def test_list_jobs():
    """Test listing all jobs."""
    print("Test 3: List all jobs")
    resp = requests.get(URL + "/v1/async/jobs", timeout=10)
    body = resp.json()
    print(f"  Total jobs: {len(body['jobs'])}")
    for job in body["jobs"]:
        print(f"    {job['job_id']}: {job['status']}")
    print(f"  ✅ PASS")
    print()


def main():
    print(f"\nAsync Inference & Storage Tests")
    print(f"URL: {URL}")
    print(f"{'='*50}\n")

    r = requests.get(URL + "/health", timeout=5)
    print(f"Health: {r.status_code}\n")

    test_async_inline()
    test_async_large()
    test_list_jobs()

    print("All tests complete.")


if __name__ == "__main__":
    main()
