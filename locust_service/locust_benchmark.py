"""
locust_benchmark.py — SFM batch size benchmark
Use Profile field in Locust UI to run one batch size at a time:
  - Profile: bs1  → only batch size 1
  - Profile: bs4  → only batch size 4
  - Profile: bs8  → only batch size 8
  - Profile: bs16 → only batch size 16
  - Profile: bs32 → only batch size 32
"""

from locust import HttpUser, task, constant, tag

SINGLE_SLICE = [0.5] * (224 * 224)

SINGLE_PAYLOAD = {"data": SINGLE_SLICE, "height": 224, "width": 224}
BATCH_4        = {"slices": [SINGLE_SLICE] * 4,  "height": 224, "width": 224}
BATCH_8        = {"slices": [SINGLE_SLICE] * 8,  "height": 224, "width": 224}
BATCH_16       = {"slices": [SINGLE_SLICE] * 16, "height": 224, "width": 224}
BATCH_32       = {"slices": [SINGLE_SLICE] * 32, "height": 224, "width": 224}


class SFMBatchUser(HttpUser):
    wait_time = constant(0)

    @tag("bs1")
    @task
    def infer_bs1(self):
        self.client.post("/infer", json=SINGLE_PAYLOAD, name="/infer [bs=1]")

    @tag("bs4")
    @task
    def infer_bs4(self):
        self.client.post("/batch_infer", json=BATCH_4, name="/batch_infer [bs=4]")

    @tag("bs8")
    @task
    def infer_bs8(self):
        self.client.post("/batch_infer", json=BATCH_8, name="/batch_infer [bs=8]")

    @tag("bs16")
    @task
    def infer_bs16(self):
        self.client.post("/batch_infer", json=BATCH_16, name="/batch_infer [bs=16]")

    @tag("bs32")
    @task
    def infer_bs32(self):
        self.client.post("/batch_infer", json=BATCH_32, name="/batch_infer [bs=32]")
