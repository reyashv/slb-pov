import os
import numpy as np
from locust import HttpUser, task, constant

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
MODEL_NAME = os.environ.get("MODEL_NAME", "sfm_large")
SERVER_TYPE = os.environ.get("SERVER_TYPE", "triton")  # "triton" or "fastapi"
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))

# Pre-build payload once
data = np.zeros((BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)

if SERVER_TYPE == "fastapi":
    ENDPOINT = "/infer"
    PAYLOAD = {
        "data": data.flatten().tolist(),
        "height": IMG_SIZE,
        "width": IMG_SIZE
    }
else:
    ENDPOINT = f"/v2/models/{MODEL_NAME}/infer"
    PAYLOAD = {
        "inputs": [{
            "name": "INPUT",
            "shape": [BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE],
            "datatype": "FP32",
            "data": data.flatten().tolist()
        }]
    }

class SFMUser(HttpUser):
    wait_time = constant(0)

    @task
    def infer(self):
        self.client.post(
            ENDPOINT,
            json=PAYLOAD,
            name=f"/infer [bs={BATCH_SIZE}]"
        )
