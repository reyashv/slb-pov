import requests
import numpy as np

frames_payload = {
    "frames": [np.random.rand(64 * 64 * 3).tolist() for _ in range(8)],
    "height": 64, "width": 64, "channels": 3
}

print("\n=== Fused Services — Single Inference ===")

# Fused classify
resp = requests.post("https://fused-classify-dipo-ws-8000.slb-pilot.truefoundry.cloud/classify", json=frames_payload, timeout=120)
result = resp.json()
print(f"fused-classify (encoder+decoder in one pod):")
print(f"  class_id: {result['class_id']}, confidence: {result['confidence']:.4f}")

# Fused segment
resp = requests.post("https://fused-segment-dipo-ws-8000.slb-pilot.truefoundry.cloud/segment", json=frames_payload, timeout=120)
result = resp.json()
print(f"\nfused-segment (encoder+decoder in one pod):")
print(f"  mask: {result['out_height']}x{result['out_width']}, first 3: {[round(x,3) for x in result['mask'][:3]]}")

# Fused detect
resp = requests.post("https://fused-detect-dipo-ws-8000.slb-pilot.truefoundry.cloud/detect", json=frames_payload, timeout=120)
result = resp.json()
print(f"\nfused-detect (encoder+decoder in one pod):")
print(f"  {len(result['boxes'])} boxes, first: {result['boxes'][0]}")