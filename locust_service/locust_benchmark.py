"""
locust_benchmark.py — SFM batch size benchmark
Tests /infer (single slice) vs /batch_infer (N slices in one GPU forward pass)
Run one user class at a time to get clean per-batch-size numbers.
"""
from locust import HttpUser, task, constant

SINGLE_SLICE = [0.5] * (224 * 224)

# Single slice payload
SINGLE_PAYLOAD = {"data": SINGLE_SLICE, "height": 224, "width": 224}

# Batch payloads
BATCH_4  = {"slices": [SINGLE_SLICE] * 4,  "height": 224, "width": 224}
BATCH_8  = {"slices": [SINGLE_SLICE] * 8,  "height": 224, "width": 224}
BATCH_16 = {"slices": [SINGLE_SLICE] * 16, "height": 224, "width": 224}
BATCH_32 = {"slices": [SINGLE_SLICE] * 32, "height": 224, "width": 224}


class Batch1(HttpUser):
    """1 slice per request — baseline"""
    wait_time = constant(0)
    @task
    def infer(self):
        self.client.post("/infer", json=SINGLE_PAYLOAD, name="/infer [bs=1]")


class Batch4(HttpUser):
    """4 slices per request"""
    wait_time = constant(0)
    @task
    def infer(self):
        self.client.post("/batch_infer", json=BATCH_4, name="/batch_infer [bs=4]")


class Batch8(HttpUser):
    """8 slices per request"""
    wait_time = constant(0)
    @task
    def infer(self):
        self.client.post("/batch_infer", json=BATCH_8, name="/batch_infer [bs=8]")


class Batch16(HttpUser):
    """16 slices per request"""
    wait_time = constant(0)
    @task
    def infer(self):
        self.client.post("/batch_infer", json=BATCH_16, name="/batch_infer [bs=16]")


class Batch32(HttpUser):
    """32 slices per request"""
    wait_time = constant(0)
    @task
    def infer(self):
        self.client.post("/batch_infer", json=BATCH_32, name="/batch_infer [bs=32]")
