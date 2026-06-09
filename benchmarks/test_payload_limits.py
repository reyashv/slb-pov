"""
test_payload_limits.py — Test payload size limits
SLB requirement: REST ≤6MB, gRPC ≤32MB

Tests:
  1. REST 6MB payload (FastAPI port 8080)
  2. REST at various sizes to find actual limit
  3. gRPC 32MB payload (PyTriton port 8001)
"""

import os
import time
import base64
import json
import numpy as np
import requests

REST_URL = os.environ.get(
    "REST_URL",
    "http://sfm-large-trt.slb-ws.svc.cluster.local:8080"
)
GRPC_HOST = os.environ.get(
    "GRPC_HOST",
    "sfm-large-trt.slb-ws.svc.cluster.local"
)
GRPC_PORT = os.environ.get("GRPC_PORT", "8001")
IMG_SIZE = int(os.environ.get("SFM_IMG_SIZE", "224"))


def test_rest_payload(h, w, label):
    """Send a base64 payload of h x w float32 via REST."""
    raw_mb = h * w * 4 / (1024 * 1024)
    b64_mb = raw_mb * 4 / 3
    print(f"\n  {label}: {h}x{w} = {raw_mb:.1f}MB raw, {b64_mb:.1f}MB base64")

    data = np.random.randn(h, w).astype(np.float32)
    data_b64 = base64.b64encode(data.tobytes()).decode("ascii")

    payload = {
        "data_b64": data_b64,
        "height": h,
        "width": w,
    }

    try:
        t0 = time.perf_counter()
        resp = requests.post(
            REST_URL + "/infer_full_image",
            json=payload,
            timeout=120,
        )
        t1 = time.perf_counter()

        if resp.status_code == 200:
            body = resp.json()
            print(f"  Status: 200 ✅  Tiles: {body['n_tiles']}  Latency: {body['latency_ms']}ms  Roundtrip: {(t1-t0)*1000:.0f}ms")
            return True
        else:
            print(f"  Status: {resp.status_code} ❌  {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_grpc_payload(h, w, label):
    """Send a payload of h x w float32 via gRPC."""
    import tritonclient.grpc as grpcclient

    raw_mb = h * w * 4 / (1024 * 1024)
    print(f"\n  {label}: {h}x{w} = {raw_mb:.1f}MB raw")

    # Tile the image into IMG_SIZE x IMG_SIZE patches and send as batch
    n_rows = int(np.ceil(h / IMG_SIZE))
    n_cols = int(np.ceil(w / IMG_SIZE))
    n_tiles = n_rows * n_cols

    # For gRPC, send one tile at the target payload size
    # A single tile is small, so to test 32MB we send a large batch
    # Actually, let's send raw data as a single large input
    # PyTriton accepts batch, so send n_tiles at once
    print(f"  Tiles: {n_tiles} ({n_rows}x{n_cols})")

    # Build batch of tiles
    tiles = np.random.randn(n_tiles, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)
    payload_mb = tiles.nbytes / (1024 * 1024)
    print(f"  gRPC payload: {payload_mb:.1f}MB ({n_tiles} tiles)")

    try:
        url = f"{GRPC_HOST}:{GRPC_PORT}"
        client = grpcclient.InferenceServerClient(url=url)

        input_tensor = grpcclient.InferInput("INPUT", tiles.shape, "FP32")
        input_tensor.set_data_from_numpy(tiles)
        output = grpcclient.InferRequestedOutput("OUTPUT")

        t0 = time.perf_counter()
        result = client.infer(
            model_name="sfm_large",
            inputs=[input_tensor],
            outputs=[output],
        )
        t1 = time.perf_counter()

        features = result.as_numpy("OUTPUT")
        print(f"  Status: ✅  Output: {features.shape}  Roundtrip: {(t1-t0)*1000:.0f}ms")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    print(f"SLB Payload Limit Tests")
    print(f"REST URL: {REST_URL}")
    print(f"gRPC: {GRPC_HOST}:{GRPC_PORT}")
    print(f"{'='*50}")

    # ── REST Tests ──
    print(f"\n--- REST Payload Tests (target: ≤6MB) ---")

    # 1MB: 512x512
    test_rest_payload(512, 512, "~1MB")

    # 3MB: 880x880
    test_rest_payload(880, 880, "~3MB")

    # 6MB: 1254x1254
    test_rest_payload(1254, 1254, "~6MB (SLB REST limit)")

    # 8MB: 1450x1450 (over limit — should still work via base64, just documenting)
    test_rest_payload(1450, 1450, "~8MB (above REST limit)")

    # ── gRPC Tests ──
    print(f"\n--- gRPC Payload Tests (target: ≤32MB) ---")

    # 8MB: batch of 16 tiles
    test_grpc_payload(896, 896, "~3MB (16 tiles)")

    # 32MB: batch of tiles for a 4Kx2K image = 190 tiles
    # 190 * 1 * 224 * 224 * 4 bytes = ~38MB
    test_grpc_payload(4096, 2048, "~38MB (190 tiles, 4Kx2K equivalent)")

    print(f"\n{'='*50}")
    print(f"All tests complete.")


if __name__ == "__main__":
    main()
