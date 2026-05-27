"""
locust_benchmark.py — SFM PyTriton batch size benchmark

Uses tritonclient to send requests to PyTriton server.
Control batch size via BATCH_SIZE env var in TFY UI.

Default: BATCH_SIZE=1
"""

import os
import numpy as np
from locust import HttpUser, task, constant
import tritonclient.http as httpclient

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
HOST = os.environ.get("TRITON_HOST", "sfm-large-trt.slb-ws.svc.cluster.local")
PORT = int(os.environ.get("TRITON_PORT", "8000"))
MODEL_NAME = "sfm_large"
IMG_SIZE = 224

print(f"Running with BATCH_SIZE={BATCH_SIZE}, host={HOST}:{PORT}")

# Prepare input data
SINGLE_SLICE = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
BATCH_INPUT = np.zeros((BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)


class SFMTritonUser(HttpUser):
    wait_time = constant(0)

    def on_start(self):
        self.triton_client = httpclient.InferenceServerClient(
            url=f"{HOST}:{PORT}",
            verbose=False
        )

    @task
    def infer(self):
        inputs = [
            httpclient.InferInput("INPUT", BATCH_INPUT.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(BATCH_INPUT)

        outputs = [
            httpclient.InferRequestedOutput("OUTPUT")
        ]

        result = self.triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs
        )
        _ = result.as_numpy("OUTPUT")
