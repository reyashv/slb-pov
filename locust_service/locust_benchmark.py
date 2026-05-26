from locust import HttpUser, task, between

SINGLE_PAYLOAD = {
    "data": [0.5] * (224 * 224),
    "height": 224,
    "width": 224
}

class SFMUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task
    def infer_single(self):
        self.client.post(
            "/infer",
            json=SINGLE_PAYLOAD,
            name="/infer [single]"
        )
