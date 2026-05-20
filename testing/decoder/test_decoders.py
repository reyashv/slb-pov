import requests
import numpy as np

frames_payload = {
    "frames": [np.random.rand(64 * 64 * 3).tolist() for _ in range(8)],
    "height": 64, "width": 64, "channels": 3
}

print("\n=== Decoupled Decoders (Composed Pipeline) ===")

# Classification
resp = requests.post("https://decoder-classify-dipo-ws-8000.slb-pilot.truefoundry.cloud/classify", json=frames_payload, timeout=120)
result = resp.json()
print(f"decoder-classify:")
print(f"  Input: 8 frames 64x64 RGB → encoder called internally → linear head")
print(f"  class_id: {result['class_id']}")
print(f"  confidence: {result['confidence']:.4f}")
print(f"  logits: {[round(x,3) for x in result['logits']]}")

# Segmentation
resp = requests.post("https://decoder-segment-dipo-ws-8000.slb-pilot.truefoundry.cloud/segment", json=frames_payload, timeout=120)
result = resp.json()
print(f"\ndecoder-segment:")
print(f"  Input: 8 frames 64x64 RGB → encoder called internally → MLP head")
print(f"  mask size: {result['out_height']}x{result['out_width']} = {len(result['mask'])} pixels")
print(f"  first 5 mask values: {[round(x,3) for x in result['mask'][:5]]}")

# Detection
resp = requests.post("https://decoder-detect-dipo-ws-8000.slb-pilot.truefoundry.cloud/detect", json=frames_payload, timeout=120)
result = resp.json()
print(f"\ndecoder-detect:")
print(f"  Input: 8 frames 64x64 RGB → encoder called internally → linear head")
print(f"  num boxes: {len(result['boxes'])}")
best_box = max(result['boxes'], key=lambda x: x['confidence'])
print(f"  highest confidence box: x={best_box['x']:.3f} y={best_box['y']:.3f} w={best_box['w']:.3f} h={best_box['h']:.3f} conf={best_box['confidence']:.3f}")