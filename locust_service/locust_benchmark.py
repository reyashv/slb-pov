"""
locust_benchmark.py — SFM load test
Single slice test: POST /infer with one 224x224 seismic slice
Batch test: POST /infer [batch-8] sends 8 slices back to back
"""

from locust import HttpUser, task, between

# Single 224x224 seismic slice payload
SINGLE_PAYLOAD = {
    "data": [0.5] * (224 * 224),
    "height": 224,
    "width": 224
}

class SFMSingleUser(HttpUser):
    """
    Single slice test — 1 request at a time, waits for full response.
    Use this first with 1 user to get baseline time per request.
    """
    wait_time = between(0.01, 0.05)  # tiny wait — lets inference time dominate

    @task(3)  # weight 3 — runs more often
    def infer_single(self):
        self.client.post(
            "/infer",
            json=SINGLE_PAYLOAD,
            name="/infer [single]"
        )

    @task(1)  # weight 1 — runs less often
    def infer_batch_8(self):
        for i in range(8):
            self.client.post(
                "/infer",
                json=SINGLE_PAYLOAD,
                name="/infer [batch-8]"
            )
