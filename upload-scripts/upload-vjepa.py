import os
from transformers import AutoModel, AutoVideoProcessor
from truefoundry.ml import get_client, TransformersFramework

client = get_client()

variants = [
    ("vjepa2-vitl",    "facebook/vjepa2-vitl-fpc64-256"),
    ("vjepa2-vith",    "facebook/vjepa2-vith-fpc64-256"),
    ("vjepa2-vitg",    "facebook/vjepa2-vitg-fpc64-256"),
    ("vjepa2-vitg-384","facebook/vjepa2-vitg-fpc64-384"),
]

for name, hf_repo in variants:
    print(f"Downloading {name} from HuggingFace...")
    local_path = f"./checkpoints/{name}"
    os.makedirs(local_path, exist_ok=True)

    model = AutoModel.from_pretrained(hf_repo)
    processor = AutoVideoProcessor.from_pretrained(hf_repo)
    model.save_pretrained(local_path)
    processor.save_pretrained(local_path)

    print(f"Uploading {name} to ML Repo...")
    mv = client.log_model(
        ml_repo="slb-pov",
        name=name,
        model_file_or_folder=local_path,
        framework=TransformersFramework(pipeline_tag="feature-extraction"),
        metadata={
            "hf_repo": hf_repo,
            "params": "1B" if "vitg" in name else "600M" if "vith" in name else "300M"
        }
    )
    print(f"{name}: {mv.fqn}")
    # Save these FQNs — you'll paste them into the UI later