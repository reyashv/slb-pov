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
    def __init__(self, data_dir, label_dir, img_size=224):
        from PIL import Image
        self.img_size = img_size
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "*.dat")))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*.dat")))
        print(f"Found {len(self.data_files)} seismic slices")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        from PIL import Image

        # Load 768x768 seismic slice
        seismic = np.fromfile(self.data_files[idx], dtype=np.float32)
        seismic = seismic.reshape(768, 768)

        # Resize to model input size
        img = Image.fromarray(seismic)
        img = img.resize((self.img_size, self.img_size))
        seismic = np.array(img, dtype=np.float32)

        # Normalize to 0-1
        seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
        tensor = torch.from_numpy(seismic).unsqueeze(0)  # [1, H, W]

        # Load label — also 768x768
        label = np.fromfile(self.label_files[idx], dtype=np.float32)
        label = label.reshape(768, 768)
        label_img = Image.fromarray(label)
        label_img = label_img.resize(
            (self.img_size, self.img_size), resample=Image.NEAREST
        )
        label_arr = np.array(label_img, dtype=np.int64)

        # Most common class in slice — labels are 1-6, convert to 0-5
        label_class = int(np.bincount(label_arr.flatten()).argmax())
        label_class = max(0, label_class - 1)  # shift 1-6 → 0-5

        return tensor, torch.tensor(label_class, dtype=torch.long)


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, classifier, dataloader, optimizer_enc, optimizer_cls,
          criterion, epochs, run):
    model.train()
    classifier.train()
    final_loss = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer_enc.zero_grad()
            optimizer_cls.zero_grad()

            features = model.forward_features(inputs)
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)

            optimizer_enc.step()
            optimizer_cls.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / len(dataloader)
        final_loss = avg_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if avg_loss == avg_loss and avg_loss != float('inf'):
            run.log_metrics({"train_loss": float(avg_loss)}, step=epoch)

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

    # Data dirs — set by Artifacts Download
    data_dir = os.environ.get("DATA_DIR", "")
    label_dir = os.environ.get("LABEL_DIR", "")

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
            "training_type": "encoder_finetune_real_data" if data_dir else "encoder_finetune_placeholder"
        })

        # Step 1 — Download model
        print(f"Downloading model from {model_fqn}...")
        os.makedirs("/tmp/sfm-model", exist_ok=True)
        download_info = client.get_model_version_by_fqn(model_fqn).download(
            path="/tmp/sfm-model"
        )
        model_dir = download_info.download_dir

        # Step 2 — Build model
        if arch not in MODEL_REGISTRY:
            raise ValueError(f"Unknown arch: {arch}")

        m = MODEL_REGISTRY[arch](
            num_classes=0,
            global_pool=False,
            in_chans=1,
            img_size=img_size
        )

        # Step 3 — Load checkpoint
        pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth file in {model_dir}")

        checkpoint = torch.load(pth_files[0], map_location="cuda")
        state_dict = checkpoint.get("model", checkpoint)
        m.load_state_dict(state_dict, strict=False)
        m.cuda()
        print(f"Model loaded — arch={arch}, img_size={img_size}")

        # Step 4 — Build classifier head
        embed_dim = 768 if "base" in arch else 1024
        classifier = nn.Linear(embed_dim, num_classes).cuda()

        # Step 5 — Optimizers
        optimizer_enc = torch.optim.AdamW(m.parameters(), lr=lr)
        optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=lr * 10)
        criterion = nn.CrossEntropyLoss()

        # Step 6 — Dataset
        if data_dir and label_dir:
            print(f"Loading real seismic data...")
            print(f"  Seismic: {data_dir}")
            print(f"  Labels:  {label_dir}")
            dataset_obj = SeismicDatDataset(data_dir, label_dir, img_size=img_size)
            dataloader = DataLoader(dataset_obj, batch_size=4, shuffle=True)
        else:
            print("No DATA_DIR/LABEL_DIR set — using placeholder data")
            placeholder = [
                (torch.randn(1, img_size, img_size), torch.randint(0, num_classes, (1,)).item())
                for _ in range(100)
            ]
            dataloader = DataLoader(placeholder, batch_size=4, shuffle=True)

        # Step 7 — Train
        print(f"Starting fine-tuning for {epochs} epochs...")
        m, classifier, final_loss = train(
            m, classifier, dataloader,
            optimizer_enc, optimizer_cls,
            criterion, epochs, run
        )

        if final_loss == final_loss:
            run.log_metrics({"final_loss": float(final_loss)})

        # Step 8 — Save
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

        # Step 9 — Log to ML Repo
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
                "training_type": "encoder_finetune_real_data" if data_dir else "placeholder"
            }
        )
        print(f"Done — logged as: {mv.fqn}")

    finally:
        run.end()


if __name__ == "__main__":
    main()
