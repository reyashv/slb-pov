"""
locust_tile_benchmark.py — Concurrent tile load test for all 3 SFM models
Tests SLB success metric: p50 ≤ 400ms, p95 ≤ 1000ms under concurrent load

Deploy as a Locust service on the cluster (pod-to-pod, same as existing locust-sfm).
Control which model and batch size via env vars in TrueFoundry UI.

Environment variables:
  MODEL          sfm-large | sfm-large-trt | sfm-large-512  (default: sfm-large-trt)
  IMG_SIZE       224 | 512                                   (default: auto from MODEL)
  HOST           override model URL                          (default: auto from MODEL)

Run config:
  Users:    ramp from 1 → 10 → 50 → 100 → 200 → 500
  Spawn:    10 users/sec
  Duration: 60s per step minimum (watch until RPS plateaus)

The Locust UI shows p50 and p95 per endpoint — screenshot each step.
"""

import os
import json
import numpy as np
from locust import HttpUser, task, constant, events

# ── Model config ──────────────────────────────────────────────────────────────

MODEL = os.environ.get("MODEL", "sfm-large-trt")

MODEL_CONFIGS = {
    "sfm-large": {
        "host": "http://sfm-large.slb-ws.svc.cluster.local:8000",
        "endpoint": "/infer",
        "img_size": 224,
        "type": "fastapi",
    },
    "sfm-large-trt": {
        "host": "http://sfm-large-trt.slb-ws.svc.cluster.local:8000",
        "endpoint": "/v2/models/sfm_large/infer",
        "img_size": 224,
        "type": "triton",
    },
    "sfm-large-512": {
        "host": "http://sfm-large-512.slb-ws.svc.cluster.local:8000",
        "endpoint": "/infer",
        "img_size": 512,
        "type": "fastapi",
    },
}

if MODEL not in MODEL_CONFIGS:
    raise ValueError(f"Unknown MODEL={MODEL}. Choose from: {list(MODEL_CONFIGS.keys())}")

cfg = MODEL_CONFIGS[MODEL]
HOST = os.environ.get("HOST", cfg["host"])
IMG_SIZE = int(os.environ.get("IMG_SIZE", cfg["img_size"]))
ENDPOINT = cfg["endpoint"]
MODEL_TYPE = cfg["type"]

print(f"Locust tile benchmark: MODEL={MODEL}, HOST={HOST}, IMG_SIZE={IMG_SIZE}")

# ── Payload ───────────────────────────────────────────────────────────────────

if MODEL_TYPE == "triton":
    data = np.zeros((1, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    PAYLOAD = {
        "inputs": [
            {
                "name": "INPUT",
                "shape": [1, 1, IMG_SIZE, IMG_SIZE],
                "datatype": "FP32",
                "data": data.flatten().tolist(),
            }
        ],
        "outputs": [{"name": "OUTPUT"}],
    }
else:
    data = np.zeros(IMG_SIZE * IMG_SIZE, dtype=np.float32)
    PAYLOAD = {"data": data.tolist(), "height": IMG_SIZE, "width": IMG_SIZE}


# ── SLB target thresholds printed at start ────────────────────────────────────

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"\n{'='*50}")
    print(f"SLB Target: p50 ≤ 400ms, p95 ≤ 1000ms")
    print(f"Model:      {MODEL}")
    print(f"Endpoint:   {HOST}{ENDPOINT}")
    print(f"Input:      {IMG_SIZE}x{IMG_SIZE} single tile")
    print(f"{'='*50}\n")


# ── User class ────────────────────────────────────────────────────────────────

class SFMTileUser(HttpUser):
    """
    Sends single tile requests as fast as possible (wait_time=0).
    Locust UI shows p50/p95 per endpoint — matches SLB success metrics directly.
    """
    host = HOST
    wait_time = constant(0)

    @task
    def infer_tile(self):
        with self.client.post(
            ENDPOINT,
            json=PAYLOAD,
            catch_response=True,
            name=f"/infer [{MODEL}]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:100]}")
