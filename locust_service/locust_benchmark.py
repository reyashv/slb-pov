from locust import HttpUser, task, constant

SINGLE_PAYLOAD = {
    "data": [0.5] * (224 * 224),
    "height": 224,
    "width": 224
}

class SFMUser(HttpUser):
    wait_time = constant(0)  # no wait between requests

    @task
    def infer_single(self):
        self.client.post(
            "/infer",
            json=SINGLE_PAYLOAD,
            name="/infer [single]"
        )
