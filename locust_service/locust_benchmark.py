"""
locust_benchmark.py — SFM batch size benchmark
Control which batch size runs via BATCH_SIZE env var in yaml.
Default is 1. Change env var and redeploy to test different sizes.
"""

import os
from locust import HttpUser, task, constant

SINGLE_SLICE = [0.5] * (224 * 224)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))

print(f"Running with BATCH_SIZE={BATCH_SIZE}")

if BATCH_SIZE == 1:
    PAYLOAD = {"data": SINGLE_SLICE, "height": 224, "width": 224}
    ENDPOINT = "/infer"
    NAME = "/infer [bs=1]"
else:
    PAYLOAD = {"slices": [SINGLE_SLICE] * BATCH_SIZE, "height": 224, "width": 224}
    ENDPOINT = "/batch_infer"
    NAME = f"/batch_infer [bs={BATCH_SIZE}]"


class SFMUser(HttpUser):
    wait_time = constant(0)

    @task
    def infer(self):
        self.client.post(ENDPOINT, json=PAYLOAD, name=NAME)
