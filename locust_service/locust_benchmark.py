"""
locust_benchmark.py — SFM PyTriton batch size benchmark
Uses Locust's built-in HTTP client to call Triton's HTTP REST API directly.
Control batch size via BATCH_SIZE env var.
"""

import os
import json
import numpy as np
from locust import HttpUser, task, constant

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
MODEL_NAME = "sfm_large"
IMG_SIZE = 224

print(f"Running with BATCH_SIZE={BATCH_SIZE}")

# Build Triton HTTP REST payload
# POST /v2/models/{model}/infer
def make_payload(batch_size):
    data = np.zeros((batch_size, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    return {
        "inputs": [
            {
                "name": "INPUT",
                "shape": [batch_size, 1, IMG_SIZE, IMG_SIZE],
                "datatype": "FP32",
                "data": data.flatten().tolist()
            }
        ],
        "outputs": [{"name": "OUTPUT"}]
    }

PAYLOAD = make_payload(BATCH_SIZE)
ENDPOINT = f"/v2/models/{MODEL_NAME}/infer"


class SFMTritonUser(HttpUser):
    wait_time = constant(0)

    @task
    def infer(self):
        self.client.post(
            ENDPOINT,
            json=PAYLOAD,
            name=f"/infer [bs={BATCH_SIZE}]"
        )
