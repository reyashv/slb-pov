import os
import glob
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
import timm.models.vision_transformer


# ── Inline VisionTransformer from facebookresearch/mae ──────────────────────

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome


def vit_base_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_large_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


MODEL_REGISTRY = {
    "vit_base_patch16": vit_base_patch16,
    "vit_large_patch16": vit_large_patch16,
}


# ── Real Seismic Dataset ──────────────────────────────────────────────────────

class SeismicDatDataset(Dataset):
    """
    Loads real seismic data from SFM facies dataset.
    Each .dat file is a raw float32 binary array of shape 768x768.
    Labels are 1-6 (6 facies classes), converted to 0-5.
    """
    def __init__(self, data_dir, label_dir, img_size=224):
        self.img_size = img_size
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "*.dat")))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*.dat")))
        assert len(self.data_files) == len(self.label_files), \
            f"Mismatch: {len(self.data_files)} seismic vs {len(self.label_files)} labels"
        print(f"Found {len(self.data_files)} seismic slices")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        from PIL import Image

        # Load 768x768 seismic slice — raw float32 binary
        seismic = np.fromfile(self.data_files[idx], dtype=np.float32)
        seismic = seismic.reshape(768, 768)

        # Resize to model input size
        img = Image.fromarray(seismic)
        img = img.resize((self.img_size, self.img_size))
        seismic = np.array(img, dtype=np.float32)

        # Normalize to 0-1
        seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
        tensor = torch.from_numpy(seismic).unsqueeze(0)  # [1, H, W]

        # Load label — also 768x768, values 1-6
        label = np.fromfile(self.label_files[idx], dtype=np.float32)
        label = label.reshape(768, 768)
        label_img = Image.fromarray(label)
        label_img = label_img.resize(
            (self.img_size, self.img_size), resample=Image.NEAREST
        )
        label_arr = np.array(label_img, dtype=np.int64)

        # Most common class in slice — shift from 1-6 to 0-5
        label_class = int(np.bincount(label_arr.flatten()).argmax())
        label_class = max(0, label_class - 1)

        return tensor, torch.tensor(label_class, dtype=torch.long)


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, classifier, dataloader, optimizer_enc, optimizer_cls,
          criterion, epochs, run):
    model.train()
    classifier.train()
    final_loss = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer_enc.zero_grad()
            optimizer_cls.zero_grad()

            features = model.forward_features(inputs)
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)

            optimizer_enc.step()
            optimizer_cls.step()

            loss_val = float(loss.item())
            if loss_val == loss_val:  # NaN check
                total_loss += loss_val

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(len(dataloader), 1)
        accuracy = correct / max(total, 1)
        final_loss = avg_loss

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        if avg_loss == avg_loss and avg_loss != float('inf'):
            run.log_metrics({
                "train_loss": float(avg_loss),
                "train_accuracy": float(accuracy)
            }, step=epoch)

    return model, classifier, final_loss


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from truefoundry.ml import get_client, PyTorchFramework

    model_fqn = os.environ["MODEL_DIR"]
    arch = os.environ.get("SFM_ARCH", "vit_base_patch16")
    img_size = int(os.environ.get("SFM_IMG_SIZE", "224"))
    epochs = int(os.environ.get("EPOCHS", "10"))
    lr = float(os.environ.get("LR", "1e-4"))
    num_classes = int(os.environ.get("NUM_CLASSES", "6"))
    output_name = os.environ.get("OUTPUT_MODEL_NAME", "sfm-base-finetuned")
    data_artifact_fqn = os.environ.get("DATA_ARTIFACT_FQN", "")

    client = get_client()

    run = client.create_run(
        ml_repo="slb-pov",
        run_name="sfm-base-finetune"
    )
    print(f"Started run: {run.run_id}")

    try:
        run.log_params({
            "base_model_fqn": str(model_fqn),
            "arch": str(arch),
            "img_size": int(img_size),
            "epochs": int(epochs),
            "lr": float(lr),
            "num_classes": int(num_classes),
            "data_artifact_fqn": str(data_artifact_fqn) if data_artifact_fqn else "placeholder",
            "training_type": "encoder_finetune_real_data" if data_artifact_fqn else "encoder_finetune_placeholder"
        })

        # Step 1 — Download model from ML Repo
        print(f"Downloading model from {model_fqn}...")
        os.makedirs("/tmp/sfm-model", exist_ok=True)
        download_info = client.get_model_version_by_fqn(model_fqn).download(
            path="/tmp/sfm-model"
        )
        model_dir_path = download_info.download_dir
        print(f"Model downloaded to {model_dir_path}")

        # Step 2 — Download data from ML Repo (if provided)
        data_dir = ""
        label_dir = ""
        if data_artifact_fqn:
            print(f"Downloading data from {data_artifact_fqn}...")
            os.makedirs("/tmp/sfm-data", exist_ok=True)
            result = client.get_artifact_version_by_fqn(data_artifact_fqn).download(
                path="/tmp/sfm-data"
            )
            # Handle both string and object return types
            if isinstance(result, str):
                data_artifact_dir = result
            else:
                data_artifact_dir = result.download_dir
            data_dir = os.path.join(data_artifact_dir, "seismic")
            label_dir = os.path.join(data_artifact_dir, "label")
            print(f"Data downloaded to {data_artifact_dir}")
            print(f"  Seismic: {data_dir}")
            print(f"  Labels:  {label_dir}")

        # Step 3 — Build model architecture
        if arch not in MODEL_REGISTRY:
            raise ValueError(f"Unknown arch: {arch}")

        m = MODEL_REGISTRY[arch](
            num_classes=0,
            global_pool=False,
            in_chans=1,
            img_size=img_size
        )

        # Step 4 — Load pretrained checkpoint
        pth_files = glob.glob(os.path.join(model_dir_path, "*.pth"))
        if not pth_files:
            raise FileNotFoundError(
                f"No .pth file in {model_dir_path}. Contents: {os.listdir(model_dir_path)}"
            )

        checkpoint_path = pth_files[0]
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        state_dict = checkpoint.get("model", checkpoint)
        m.load_state_dict(state_dict, strict=False)
        m.cuda()
        print(f"Model loaded — arch={arch}, img_size={img_size}")

        # Step 5 — Build classifier head on top of encoder
        embed_dim = 768 if "base" in arch else 1024
        classifier = nn.Linear(embed_dim, num_classes).cuda()

        # Step 6 — Optimizers
        optimizer_enc = torch.optim.AdamW(m.parameters(), lr=lr)
        optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=lr * 10)
        criterion = nn.CrossEntropyLoss()

        # Step 7 — Dataset
        if data_dir and label_dir and os.path.exists(data_dir):
            print(f"Loading real seismic facies data...")
            dataset_obj = SeismicDatDataset(data_dir, label_dir, img_size=img_size)
            dataloader = DataLoader(
                dataset_obj, batch_size=4, shuffle=True, num_workers=2
            )
        else:
            print("No data artifact provided — using placeholder random data")
            placeholder = [
                (
                    torch.randn(1, img_size, img_size),
                    torch.randint(0, num_classes, (1,)).item()
                )
                for _ in range(100)
            ]
            dataloader = DataLoader(placeholder, batch_size=4, shuffle=True)

        # Step 8 — Fine-tune
        print(f"Starting fine-tuning for {epochs} epochs...")
        m, classifier, final_loss = train(
            m, classifier, dataloader,
            optimizer_enc, optimizer_cls,
            criterion, epochs, run
        )

        if final_loss == final_loss and final_loss != float('inf'):
            run.log_metrics({"final_loss": float(final_loss)})

        # Step 9 — Save checkpoint
        output_dir = "/output"
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            "model": m.state_dict(),
            "classifier": classifier.state_dict(),
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            "arch": arch,
            "img_size": img_size
        }, os.path.join(output_dir, "finetuned.pth"))
        print(f"Saved checkpoint to {output_dir}/finetuned.pth")

        # Step 10 — Log to ML Repo with run ID for lineage
        print(f"Logging fine-tuned model as {output_name}...")
        mv = run.log_model(
            name=output_name,
            model_file_or_folder=output_dir,
            framework=PyTorchFramework(),
            metadata={
                "base_model_fqn": str(model_fqn),
                "arch": str(arch),
                "img_size": int(img_size),
                "epochs": int(epochs),
                "lr": float(lr),
                "num_classes": int(num_classes),
                "training_type": "encoder_finetune_real_data" if data_artifact_fqn else "placeholder"
            }
        )
        print(f"Done — logged as: {mv.fqn}")

    finally:
        run.end()


if __name__ == "__main__":
    main()
