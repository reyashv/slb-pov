"""
test_grpc.py — Test gRPC endpoint on sfm-large-trt (port 8001)
Sends a single tile via gRPC and prints the response.
"""

import os
import time
import numpy as np
import tritonclient.grpc as grpcclient

HOST = os.environ.get("GRPC_HOST", "sfm-large-trt.slb-ws.svc.cluster.local")
PORT = os.environ.get("GRPC_PORT", "8001")
MODEL_NAME = "sfm_large"
IMG_SIZE = int(os.environ.get("SFM_IMG_SIZE", "224"))

def main():
    url = f"{HOST}:{PORT}"
    print(f"Testing gRPC endpoint: {url}")
    print(f"Model: {MODEL_NAME}, Input: 1x1x{IMG_SIZE}x{IMG_SIZE}")

    # Connect
    client = grpcclient.InferenceServerClient(url=url)

    # Health check
    if client.is_server_ready():
        print("Server ready: ✅")
    else:
        print("Server NOT ready ❌")
        return

    if client.is_model_ready(MODEL_NAME):
        print(f"Model '{MODEL_NAME}' ready: ✅")
    else:
        print(f"Model '{MODEL_NAME}' NOT ready ❌")
        return

    # Build input
    data = np.random.randn(1, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)
    input_tensor = grpcclient.InferInput("INPUT", data.shape, "FP32")
    input_tensor.set_data_from_numpy(data)

    output = grpcclient.InferRequestedOutput("OUTPUT")

    # Warmup
    print("Warming up (3 requests)...", end="", flush=True)
    for _ in range(3):
        client.infer(model_name=MODEL_NAME, inputs=[input_tensor], outputs=[output])
    print(" done")

    # Timed runs
    n = 50
    latencies = []
    errors = 0
    print(f"Running {n} gRPC requests...", end="", flush=True)
    for i in range(n):
        try:
            t0 = time.perf_counter()
            result = client.infer(model_name=MODEL_NAME, inputs=[input_tensor], outputs=[output])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
            features = result.as_numpy("OUTPUT")
        except Exception as e:
            errors += 1
            print(f"\n  Error: {e}")
        if (i + 1) % 10 == 0:
            print(f" {i+1}", end="", flush=True)
    print(" done")

    if not latencies:
        print("ERROR: All requests failed")
        return

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print(f"\n{'='*50}")
    print(f"  gRPC Results — {MODEL_NAME}")
    print(f"{'='*50}")
    print(f"  Requests: {n}")
    print(f"  Errors:   {errors}")
    print(f"  p50:      {p50:.1f} ms")
    print(f"  p95:      {p95:.1f} ms")
    print(f"  min:      {min(latencies):.1f} ms")
    print(f"  max:      {max(latencies):.1f} ms")
    print(f"  Output shape: {features.shape}")
    print(f"  gRPC working: ✅")


if __name__ == "__main__":
    main()
