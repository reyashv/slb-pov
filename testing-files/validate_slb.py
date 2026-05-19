import requests
import numpy as np

def test(name, url, payload, expected_key):
    try:
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code == 200 and expected_key in resp.json():
            print(f"✅ {name}")
            return True
        else:
            print(f"❌ {name} — Status: {resp.status_code} — {resp.text[:150]}")
            return False
    except Exception as e:
        print(f"❌ {name} — {e}")
        return False

def health_check(name, url):
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            print(f"✅ {name} — healthy")
            return True
        else:
            print(f"❌ {name} — Status: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name} — {e}")
        return False

seismic_224 = {"data": [0.5] * (224 * 224), "height": 224, "width": 224}
seismic_512 = {"data": [0.5] * (512 * 512), "height": 512, "width": 512}
frames = {"frames": [np.random.rand(64*64*3).tolist() for _ in range(8)], "height": 64, "width": 64, "channels": 3}
batch = {"videos": [
    {"frames": [np.random.rand(64*64*3).tolist() for _ in range(8)], "height": 64, "width": 64, "channels": 3},
    {"frames": [np.random.rand(64*64*3).tolist() for _ in range(8)], "height": 64, "width": 64, "channels": 3},
]}

print("\n" + "="*60)
print("SLB PoV — TrueFoundry Endpoint Validation")
print("="*60)

print("\n--- SFM Encoder Health Checks ---")
health_check("SFM-Base",      "https://sfm-base-dipo-ws-8000.slb-pilot.truefoundry.cloud/health")
health_check("SFM-Base-512",  "https://sfm-base-512-dipo-ws-8000.slb-pilot.truefoundry.cloud/health")
health_check("SFM-Large",     "https://sfm-large-dipo-ws-8000.slb-pilot.truefoundry.cloud/health")
health_check("SFM-Large-512", "https://sfm-large-512-dipo-ws-8000.slb-pilot.truefoundry.cloud/health")

print("\n--- SFM Encoder Inference ---")
test("SFM-Base /infer",      "https://sfm-base-dipo-ws-8000.slb-pilot.truefoundry.cloud/infer",      seismic_224, "features")
test("SFM-Base-512 /infer",  "https://sfm-base-512-dipo-ws-8000.slb-pilot.truefoundry.cloud/infer",  seismic_512, "features")
test("SFM-Large /infer",     "https://sfm-large-dipo-ws-8000.slb-pilot.truefoundry.cloud/infer",     seismic_224, "features")
test("SFM-Large-512 /infer", "https://sfm-large-512-dipo-ws-8000.slb-pilot.truefoundry.cloud/infer", seismic_512, "features")

print("\n--- V-JEPA Encoder Health Checks ---")
health_check("V-JEPA ViT-L",     "https://vjepa2-vitl-dipo-ws-8000.slb-pilot.truefoundry.cloud/health")
health_check("V-JEPA ViT-H",     "https://vjepa2-vith-dipo-ws-8000.slb-pilot.truefoundry.cloud/health")
health_check("V-JEPA ViT-g",     "https://vjepa2-vitg-1-dipo-ws-8000.slb-pilot.truefoundry.cloud/health")
health_check("V-JEPA ViT-g-384", "https://vjepa2-vitg-384-1-dipo-ws-8000.slb-pilot.truefoundry.cloud/health")

print("\n--- V-JEPA Encoder Inference ---")
test("V-JEPA ViT-L /embeddings",     "https://vjepa2-vitl-dipo-ws-8000.slb-pilot.truefoundry.cloud/embeddings",     frames, "embeddings")
test("V-JEPA ViT-H /embeddings",     "https://vjepa2-vith-dipo-ws-8000.slb-pilot.truefoundry.cloud/embeddings",     frames, "embeddings")
test("V-JEPA ViT-g /embeddings",     "https://vjepa2-vitg-1-dipo-ws-8000.slb-pilot.truefoundry.cloud/embeddings",     frames, "embeddings")
test("V-JEPA ViT-g-384 /embeddings", "https://vjepa2-vitg-384-1-dipo-ws-8000.slb-pilot.truefoundry.cloud/embeddings", frames, "embeddings")

print("\n--- Decoupled Decoders ---")
test("decoder-classify /classify", "https://decoder-classify-dipo-ws-8000.slb-pilot.truefoundry.cloud/classify", frames, "class_id")
test("decoder-segment /segment",   "https://decoder-segment-dipo-ws-8000.slb-pilot.truefoundry.cloud/segment",   frames, "mask")
test("decoder-detect /detect",     "https://decoder-detect-dipo-ws-8000.slb-pilot.truefoundry.cloud/detect",     frames, "boxes")

print("\n--- Fused Services Single ---")
test("fused-classify /classify", "https://fused-classify-dipo-ws-8000.slb-pilot.truefoundry.cloud/classify", frames, "class_id")
test("fused-segment /segment",   "https://fused-segment-dipo-ws-8000.slb-pilot.truefoundry.cloud/segment",   frames, "mask")
test("fused-detect /detect",     "https://fused-detect-dipo-ws-8000.slb-pilot.truefoundry.cloud/detect",     frames, "boxes")

print("\n--- Fused Services Batch ---")
test("fused-classify /batch_classify", "https://fused-classify-dipo-ws-8000.slb-pilot.truefoundry.cloud/batch_classify", batch, "results")
test("fused-segment /batch_segment",   "https://fused-segment-dipo-ws-8000.slb-pilot.truefoundry.cloud/batch_segment",   batch, "results")
test("fused-detect /batch_detect",     "https://fused-detect-dipo-ws-8000.slb-pilot.truefoundry.cloud/batch_detect",     batch, "results")

print("\n" + "="*60)
print("Validation complete")
print("="*60)