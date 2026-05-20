import requests

print("\n=== SFM Encoder Services ===")

# SFM-Base — 224x224
payload = {"data": [0.5] * (224 * 224), "height": 224, "width": 224}
resp = requests.post("https://sfm-base-dipo-ws-8000.slb-pilot.truefoundry.cloud/infer", json=payload, timeout=60)
result = resp.json()
print(f"SFM-Base: {len(result['features'])} features, first 3: {result['features'][:3]}")

# SFM-Base-512 — 512x512
payload = {"data": [0.5] * (512 * 512), "height": 512, "width": 512}
resp = requests.post("https://sfm-base-512-dipo-ws-8000.slb-pilot.truefoundry.cloud/infer", json=payload, timeout=60)
result = resp.json()
print(f"SFM-Base-512: {len(result['features'])} features, first 3: {result['features'][:3]}")

# SFM-Large — 224x224
payload = {"data": [0.5] * (224 * 224), "height": 224, "width": 224}
resp = requests.post("https://sfm-large-dipo-ws-8000.slb-pilot.truefoundry.cloud/infer", json=payload, timeout=60)
result = resp.json()
print(f"SFM-Large: {len(result['features'])} features, first 3: {result['features'][:3]}")

# SFM-Large-512 — 512x512
payload = {"data": [0.5] * (512 * 512), "height": 512, "width": 512}
resp = requests.post("https://sfm-large-512-dipo-ws-8000.slb-pilot.truefoundry.cloud/infer", json=payload, timeout=60)
result = resp.json()
print(f"SFM-Large-512: {len(result['features'])} features, first 3: {result['features'][:3]}")
