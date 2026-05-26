"""
locust_benchmark.py

Load tests sfm-base service with:
1. Single slice requests (one by one)
2. Batch requests (8 slices at a time)

Deploy this as a service on TFY cluster for accurate latency measurements.
Run with: locust -f locust_benchmark.py
"""

import numpy as np
from locust import FastHttpUser, task, between

# Single 224x224 seismic slice payload (synthetic)
SINGLE_PAYLOAD = {
    "data": [0.5] * (224 * 224),
    "height": 224,
    "width": 224
}

class SFMSingleUser(FastHttpUser):
    """
    Simulates a user sending one seismic slice at a time.
    Measures time per single request.
    """
    wait_time = between(0.1, 0.5)  # wait 0.1-0.5s between requests

    @task
    def infer_single(self):
        with self.client.post(
            "/infer",
            json=SINGLE_PAYLOAD,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


class SFMBatchUser(FastHttpUser):
    """
    Simulates a user sending 8 seismic slices back to back.
    Measures total batch time and per-slice time.
    """
    wait_time = between(0.5, 1.0)

    @task
    def infer_batch(self):
        for i in range(8):
            with self.client.post(
                "/infer",
                json=SINGLE_PAYLOAD,
                catch_response=True,
                name="/infer [batch-8]"  # group all 8 under same name in UI
            ) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Status {response.status_code}")
