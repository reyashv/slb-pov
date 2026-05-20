import requests
import numpy as np

print("\n" + "="*60)
print("Base vs Fine-tuned Model Comparison")
print("="*60)

# Fixed seed so same input goes to both models
np.random.seed(42)
frames_payload = {
    "frames": [np.random.rand(64 * 64 * 3).tolist() for _ in range(8)],
    "height": 64, "width": 64, "channels": 3
}

seismic_224 = {
    "data": [0.5] * (224 * 224),
    "height": 224, "width": 224
}

# ── SFM Base vs Fine-tuned ────────────────────────────────────────────────────
print("\n--- SFM: Base vs Fine-tuned ---")
print("Same 224x224 seismic slice sent to both")

resp_base = requests.post(
    "https://sfm-base-dipo-ws-8000.slb-pilot.truefoundry.cloud/infer",
    json=seismic_224, timeout=60
)
resp_ft = requests.post(
    "https://sfm-base-finetuned-dipo-ws-8000.slb-pilot.truefoundry.cloud/infer",
    json=seismic_224, timeout=60
)

if resp_base.status_code == 200 and resp_ft.status_code == 200:
    base_features = np.array(resp_base.json()["features"])
    ft_features = np.array(resp_ft.json()["features"])

    # Cosine similarity between base and finetuned embeddings
    cosine_sim = np.dot(base_features, ft_features) / (
        np.linalg.norm(base_features) * np.linalg.norm(ft_features)
    )
    # L2 distance
    l2_dist = np.linalg.norm(base_features - ft_features)

    print(f"\nBase model     first 5: {[round(x,4) for x in base_features[:5].tolist()]}")
    print(f"Finetuned model first 5: {[round(x,4) for x in ft_features[:5].tolist()]}")
    print(f"\nCosine similarity: {cosine_sim:.4f} (1.0=identical, 0.0=orthogonal)")
    print(f"L2 distance:       {l2_dist:.4f} (0.0=identical)")
    print(f"Feature dim:       {len(base_features)} (both should be 768)")

# ── Decoder-Classify: Base vs Probe-trained vs Joint ─────────────────────────
print("\n--- Decoder-Classify: Base vs Probe-trained vs Joint ---")
print("Same 8 frames sent to all three")

resp_base = requests.post(
    "https://decoder-classify-dipo-ws-8000.slb-pilot.truefoundry.cloud/classify",
    json=frames_payload, timeout=120
)
resp_probe = requests.post(
    "https://decoder-classify-finetuned-dipo-ws-8000.slb-pilot.truefoundry.cloud/classify",
    json=frames_payload, timeout=120
)
resp_joint = requests.post(
    "https://decoder-classify-joint-finetuned-dipo-ws-8000.slb-pilot.truefoundry.cloud/classify",
    json=frames_payload, timeout=120
)

if all(r.status_code == 200 for r in [resp_base, resp_probe, resp_joint]):
    base   = resp_base.json()
    probe  = resp_probe.json()
    joint  = resp_joint.json()

    print(f"\n{'Model':<30} {'class_id':<12} {'confidence':<12} {'top logit'}")
    print("-" * 65)
    print(f"{'Base (random weights)':<30} {base['class_id']:<12} {base['confidence']:<12.4f} {max(base['logits']):.4f}")
    print(f"{'Probe-trained (frozen enc)':<30} {probe['class_id']:<12} {probe['confidence']:<12.4f} {max(probe['logits']):.4f}")
    print(f"{'Joint (enc+dec trained)':<30} {joint['class_id']:<12} {joint['confidence']:<12.4f} {max(joint['logits']):.4f}")

    print(f"\nFull logits comparison:")
    print(f"Base:   {[round(x,3) for x in base['logits']]}")
    print(f"Probe:  {[round(x,3) for x in probe['logits']]}")
    print(f"Joint:  {[round(x,3) for x in joint['logits']]}")

    # Check if predictions differ
    if base['class_id'] != probe['class_id']:
        print(f"\n⚡ Probe training changed prediction: {base['class_id']} → {probe['class_id']}")
    if base['class_id'] != joint['class_id']:
        print(f"⚡ Joint training changed prediction: {base['class_id']} → {joint['class_id']}")
    if base['class_id'] == probe['class_id'] == joint['class_id']:
        print(f"\nℹ️  All models predict same class (expected with placeholder data)")

print("\n" + "="*60)
print("Comparison complete")
print("="*60)