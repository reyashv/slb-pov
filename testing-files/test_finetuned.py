import requests
import numpy as np

print("\n" + "="*60)
print("Fine-tuned Model Services Test")
print("="*60)

# Shared payload
frames_payload = {
    "frames": [np.random.rand(64 * 64 * 3).tolist() for _ in range(8)],
    "height": 64,
    "width": 64,
    "channels": 3
}

seismic_224 = {
    "data": [0.5] * (224 * 224),
    "height": 224,
    "width": 224
}

# ── SFM-Base Fine-tuned ───────────────────────────────────────────────────────
print("\n--- SFM-Base Fine-tuned ---")
print("Input: 224x224 seismic slice")
resp = requests.post(
    "https://sfm-base-finetuned-dipo-ws-8000.slb-pilot.truefoundry.cloud/infer",
    json=seismic_224,
    timeout=60
)
if resp.status_code == 200:
    result = resp.json()
    print(f"✅ Status: {resp.status_code}")
    print(f"Output: {len(result['features'])} dimensional feature vector")
    print(f"First 5 values: {result['features'][:5]}")
else:
    print(f"❌ Status: {resp.status_code} — {resp.text[:200]}")

# ── Decoder-Classify Fine-tuned (probe trained) ───────────────────────────────
print("\n--- Decoder-Classify Fine-tuned (probe trained on frozen encoder) ---")
print("Input: 8 frames 64x64 RGB")
resp = requests.post(
    "https://decoder-classify-finetuned-dipo-ws-8000.slb-pilot.truefoundry.cloud/classify",
    json=frames_payload,
    timeout=120
)
if resp.status_code == 200:
    result = resp.json()
    print(f"✅ Status: {resp.status_code}")
    print(f"class_id: {result['class_id']}")
    print(f"confidence: {result['confidence']:.4f}")
    print(f"logits: {[round(x,3) for x in result['logits']]}")
else:
    print(f"❌ Status: {resp.status_code} — {resp.text[:200]}")

# ── Decoder-Classify Joint (encoder+decoder jointly trained) ──────────────────
print("\n--- Decoder-Classify Joint (encoder+decoder jointly fine-tuned) ---")
print("Input: 8 frames 64x64 RGB")
resp = requests.post(
    "https://decoder-classify-joint-finetuned-dipo-ws-8000.slb-pilot.truefoundry.cloud/classify",
    json=frames_payload,
    timeout=120
)
if resp.status_code == 200:
    result = resp.json()
    print(f"✅ Status: {resp.status_code}")
    print(f"class_id: {result['class_id']}")
    print(f"confidence: {result['confidence']:.4f}")
    print(f"logits: {[round(x,3) for x in result['logits']]}")
else:
    print(f"❌ Status: {resp.status_code} — {resp.text[:200]}")

print("\n" + "="*60)
print("Fine-tuned model tests complete")
print("="*60)