import os
import torch
import numpy as np
from truefoundry.ml import get_client, ArtifactPath

OUTPUT_DIR = "/tmp/geobody-preprocessed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "geobody_preprocessed.pt")
HF_DATASET = os.environ.get("HF_DATASET", "porestar/seismicfoundationmodel-geobody")

def preprocess():
    from datasets import load_dataset
    client = get_client()
    run = client.create_run(ml_repo="slb-pov", run_name="geobody-preprocess")

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for split in ["train","validation"]:
            print(f"Loading {split} split from HuggingFace...")
            dataset = load_dataset(HF_DATASET, split=split).with_format(type="numpy")
            print(f"Loaded {len(dataset)} samples — preprocessing...")

            tensors = []
            labels = []

            for idx in range(len(dataset)):
                item = dataset[idx]

                # seismic
                seismic = item["seismic"].astype(np.float32)
                if seismic.ndim == 3:
                    seismic = seismic[:, :, 0]
                seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
                tensor = torch.from_numpy(seismic).unsqueeze(0)  # [1, H, W]
                tensors.append(tensor)

                # label
                label = item["label"].astype(np.float32)
                if label.ndim == 3:
                    label = label[:, :, 0]
                label_class = int(np.bincount(label.flatten().astype(int)).argmax())
                label_class = min(label_class, 1)
                labels.append(label_class)

                if (idx + 1) % 500 == 0:
                    print(f"  Processed {idx + 1}/{len(dataset)}")

            # Stack into tensors
            tensors_stacked = torch.stack(tensors)   # [N, 1, H, W]
            labels_tensor = torch.tensor(labels, dtype=torch.long)  # [N]

            output_file = os.path.join(OUTPUT_DIR, f"geobody_{split}.pt")
            torch.save({
                "tensors": tensors_stacked,
                "labels": labels_tensor,
                "split": split,
                "num_samples": len(dataset),
            }, output_file)
            print(f"Saved {split} → {output_file} ({tensors_stacked.shape})")

            # Log to ML Repo
            run.log_artifact(
                name=f"geobody-preprocessed-{split}",
                artifact_paths=[ArtifactPath(src=output_file, dest=f"geobody_{split}.pt")],
                metadata={"split": split, "num_samples": len(dataset)}
            )
            print(f"Logged to ML Repo: geobody-preprocessed-{split}")

    finally:
        run.end()

if __name__ == "__main__":
    preprocess()
