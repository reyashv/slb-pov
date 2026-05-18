import os
import glob
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import timm.models.vision_transformer


# ── Inline VisionTransformer from facebookresearch/mae ──────────────────────
# Same architecture as sfm_server.py

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
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


MODEL_REGISTRY = {
    "vit_base_patch16": vit_base_patch16,
    "vit_large_patch16": vit_large_patch16,
}


# ── Training ─────────────────────────────────────────────────────────────────

def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            features = model.forward_features(inputs)
            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from truefoundry.ml import get_client, PyTorchFramework

    # Read env vars
    model_fqn = os.environ["MODEL_DIR"]
    arch = os.environ.get("SFM_ARCH", "vit_base_patch16")
    img_size = int(os.environ.get("SFM_IMG_SIZE", "224"))
    epochs = int(os.environ.get("EPOCHS", "10"))
    lr = float(os.environ.get("LR", "1e-4"))
    output_name = os.environ.get("OUTPUT_MODEL_NAME", "sfm-base-finetuned")

    # Step 1 — Download model from ML Repo
    print(f"Downloading model from {model_fqn}...")
    client = get_client()
    download_info = client.get_model_version_by_fqn(model_fqn).download(
        path="/tmp/sfm-model"
    )
    model_dir = download_info.download_dir
    print(f"Model downloaded to {model_dir}")

    # Step 2 — Build model architecture
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arch: {arch}. Choose from {list(MODEL_REGISTRY.keys())}")

    m = MODEL_REGISTRY[arch](
        num_classes=0,
        global_pool=False,
        in_chans=1,
        img_size=img_size
    )

    # Step 3 — Load pretrained checkpoint
    pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(
            f"No .pth file in {model_dir}. Contents: {os.listdir(model_dir)}"
        )

    checkpoint_path = pth_files[0]
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    state_dict = checkpoint.get("model", checkpoint)
    m.load_state_dict(state_dict, strict=False)
    m.cuda()
    print(f"Model loaded — arch={arch}, img_size={img_size}")

    # Step 4 — Set up optimizer and loss
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

    # embed_dim depends on arch
    embed_dim = 768 if "base" in arch else 1024
    criterion = nn.MSELoss()

    # Step 5 — Create placeholder dataset
    print("Creating placeholder dataset (random seismic-like data)...")
    dataset = [
        (
            torch.randn(1, img_size, img_size),  # single channel seismic slice
            torch.randn(embed_dim)                # target feature vector
        )
        for _ in range(100)
    ]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    # Step 6 — Fine-tune
    print(f"Starting fine-tuning for {epochs} epochs...")
    m = train(m, dataloader, optimizer, criterion, epochs)

    # Step 7 — Save checkpoint
    output_dir = "/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "finetuned.pth")
    torch.save({"model": m.state_dict()}, output_path)
    print(f"Saved fine-tuned checkpoint to {output_path}")

    # Step 8 — Log new model version to ML Repo
    # This creates sfm-base-finetuned:1 in ML Repo
    # with a run ID linking back to this job
    print(f"Logging fine-tuned model to ML Repo as {output_name}...")
    mv = client.log_model(
        ml_repo="slb-pov",
        name=output_name,
        model_file_or_folder=output_dir,
        framework=PyTorchFramework(),
        metadata={
            "base_model_fqn": model_fqn,
            "arch": arch,
            "img_size": img_size,
            "epochs": epochs,
            "lr": lr,
            "training_type": "encoder_finetune"
        }
    )
    print(f"Done — logged as: {mv.fqn}")


if __name__ == "__main__":
    main()
