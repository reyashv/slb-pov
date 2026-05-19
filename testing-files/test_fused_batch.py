import requests
import numpy as np

batch_payload = {
    "videos": [
        {"frames": [np.random.rand(64*64*3).tolist() for _ in range(8)], "height": 64, "width": 64, "channels": 3},
        {"frames": [np.random.rand(64*64*3).tolist() for _ in range(8)], "height": 64, "width": 64, "channels": 3},
        {"frames": [np.random.rand(64*64*3).tolist() for _ in range(8)], "height": 64, "width": 64, "channels": 3},
    ]
}

print("\n=== Fused Services — Batch Inference ===")

# Batch classify
resp = requests.post("https://fused-classify-dipo-ws-8000.slb-pilot.truefoundry.cloud/batch_classify", json=batch_payload, timeout=120)
result = resp.json()
print(f"batch_classify — {len(result['results'])} results:")
for i, r in enumerate(result['results']):
    print(f"  Video {i+1}: class_id={r['class_id']}, confidence={r['confidence']:.4f}")

# Batch segment
resp = requests.post("https://fused-segment-dipo-ws-8000.slb-pilot.truefoundry.cloud/batch_segment", json=batch_payload, timeout=120)
result = resp.json()
print(f"\nbatch_segment — {len(result['results'])} masks:")
for i, r in enumerate(result['results']):
    print(f"  Video {i+1}: mask {r['out_height']}x{r['out_width']}, first value: {round(r['mask'][0],3)}")

# Batch detect
resp = requests.post("https://fused-detect-dipo-ws-8000.slb-pilot.truefoundry.cloud/batch_detect", json=batch_payload, timeout=120)
result = resp.json()
print(f"\nbatch_detect — {len(result['results'])} detection results:")
for i, r in enumerate(result['results']):
    best = max(r['boxes'], key=lambda x: x['confidence'])
    print(f"  Video {i+1}: {len(r['boxes'])} boxes, best confidence={best['confidence']:.3f}")