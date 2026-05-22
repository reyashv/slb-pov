import os
import glob
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
import timm.models.vision_transformer

# ── Inline VisionTransformer ─────────────────────────────────────────────────

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

# ── Checkpoint config ─────────────────────────────────────────────────────────

CHECKPOINT_NAME        = "sfm-geobody-finetune-checkpoint"
CHECKPOINT_DIR         = "/tmp/sfm-checkpoint"
CHECKPOINT_FILE        = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
MAX_CHECKPOINT_SEARCH  = 50


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def try_load_checkpoint(client, run):
    print("Searching ML Repo for latest checkpoint...")
    latest_av = None

    for version in range(MAX_CHECKPOINT_SEARCH, 0, -1):
        try:
            fqn = f"artifact:slb-pilot/slb-pov/{CHECKPOINT_NAME}:{version}"
            av = client.get_artifact_version_by_fqn(fqn=fqn)
            latest_av = av
            print(f"Found latest checkpoint: {fqn}")
            break
        except Exception:
            continue

    if latest_av is None:
        print("No checkpoint found — starting from scratch (epoch 0)")
        return 0, None, None

    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        latest_av.download(path=CHECKPOINT_DIR)

        if not os.path.exists(CHECKPOINT_FILE):
            print("checkpoint.pth not found after download — starting from scratch")
            return 0, None, None

        ckpt = torch.load(CHECKPOINT_FILE, map_location="cuda")
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch} (checkpoint: {latest_av.fqn})")
        run.log_params({
            "resumed_from_epoch": start_epoch,
            "checkpoint_fqn": latest_av.fqn,
        })
        return start_epoch, ckpt.get("model_state"), ckpt.get("optimizer_state")

    except Exception as e:
        print(f"Failed to load checkpoint ({e}) — starting from scratch")
        return 0, None, None


def save_checkpoint(run, model, optimizer, epoch, loss):
    from truefoundry.ml import ArtifactPath

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
    }, CHECKPOINT_FILE)

    av = run.log_artifact(
        name=CHECKPOINT_NAME,
        artifact_paths=[ArtifactPath(src=CHECKPOINT_FILE, dest="checkpoint.pth")],
        metadata={
            "epoch": epoch,
            "loss": float(loss) if loss == loss else -1,
        }
    )
    print(f"Checkpoint saved — epoch={epoch}, fqn={av.fqn}")


# ── HuggingFace Geobody Dataset ───────────────────────────────────────────────

class GeobodyHFDataset(Dataset):
    """
    Loads the porestar/seismicfoundationmodel-geobody dataset from HuggingFace.
    Each item: seismic (numpy HxW) + label (numpy HxW, binary salt mask)
    Label: most common pixel value used as class (0=no salt, 1=salt)
    """
    def __init__(self, split="train"):
        from datasets import load_dataset
        print(f"Loading geobody dataset from HuggingFace (split={split})...")
        self.dataset = load_dataset(
            "porestar/seismicfoundationmodel-geobody",
            split=split
        ).with_format(type="numpy")
        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # seismic: numpy array HxW (grayscale image)
        seismic = item["seismic"].astype(np.float32)

        # Handle both grayscale (HxW) and RGB (HxWxC)
        if seismic.ndim == 3:
            seismic = seismic[:, :, 0]  # take first channel

        # Normalize to 0-1
        seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
        tensor = torch.from_numpy(seismic).unsqueeze(0)  # [1, H, W]

        # label: binary mask — use most common value as class (0=no salt, 1=salt)
        label = item["label"].astype(np.float32)
        if label.ndim == 3:
            label = label[:, :, 0]
        label_class = int(np.bincount(label.flatten().astype(int)).argmax())
        label_class = min(label_class, 1)  # binary: 0 or 1

        return tensor, torch.tensor(label_class, dtype=torch.long)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, classifier, dataloader, optimizer_enc, optimizer_cls, criterion):
    model.train()
    classifier.train()
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)

        optimizer_enc.step()
        optimizer_cls.step()

        loss_val = float(loss.item())
        if loss_val == loss_val:
            total_loss += loss_val
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def val_epoch(model, classifier, dataloader, criterion):
    model.eval()
    classifier.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            features = model.forward_features(inputs)
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss_val = float(loss.item())
            if loss_val == loss_val:
                total_loss += loss_val
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from truefoundry.ml import get_client, PyTorchFramework

    model_fqn         = os.environ["MODEL_DIR"]
    arch              = os.environ.get("SFM_ARCH", "vit_large_patch16")
    img_size          = int(os.environ.get("SFM_IMG_SIZE", "224"))
    epochs            = int(os.environ.get("EPOCHS", "10"))
    lr                = float(os.environ.get("LR", "1e-4"))
    num_classes       = int(os.environ.get("NUM_CLASSES", "2"))  # binary: salt or not
    batch_size        = int(os.environ.get("BATCH_SIZE", "8"))
    checkpoint_every  = int(os.environ.get("CHECKPOINT_EVERY", "1"))
    output_name       = os.environ.get("OUTPUT_MODEL_NAME", "sfm-large-geobody-finetuned")

    client = get_client()
    run = client.create_run(ml_repo="slb-pov", run_name="sfm-geobody-finetune")
    print(f"Started run: {run.run_id}")

    try:
        run.log_params({
            "base_model_fqn": str(model_fqn),
            "arch": str(arch),
            "img_size": int(img_size),
            "epochs": int(epochs),
            "lr": float(lr),
            "num_classes": int(num_classes),
            "batch_size": int(batch_size),
            "dataset": "porestar/seismicfoundationmodel-geobody",
        })

        # ── Step 1: Find and load latest checkpoint ──
        start_epoch, model_state, optimizer_state = try_load_checkpoint(client, run)

        # ── Step 2: Download base model ──
        print(f"Downloading base model from {model_fqn}...")
        os.makedirs("/tmp/sfm-model", exist_ok=True)
        download_info = client.get_model_version_by_fqn(model_fqn).download(path="/tmp/sfm-model")
        model_dir_path = download_info.download_dir

        # ── Step 3: Build model ──
        if arch not in MODEL_REGISTRY:
            raise ValueError(f"Unknown arch: {arch}")

        m = MODEL_REGISTRY[arch](
            num_classes=0, global_pool=False, in_chans=1, img_size=img_size
        )

        pth_files = glob.glob(os.path.join(model_dir_path, "*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth in {model_dir_path}")

        checkpoint = torch.load(pth_files[0], map_location="cuda")
        state_dict = checkpoint.get("model", checkpoint)
        m.load_state_dict(state_dict, strict=False)
        m.cuda()

        embed_dim  = 768 if "base" in arch else 1024
        classifier = nn.Linear(embed_dim, num_classes).cuda()

        optimizer_enc = torch.optim.AdamW(m.parameters(), lr=lr)
        optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=lr * 10)
        criterion     = nn.CrossEntropyLoss()

        # ── Step 4: Restore from checkpoint ──
        if model_state is not None:
            m.load_state_dict(model_state, strict=False)
            print("Restored model weights from checkpoint")
        if optimizer_state is not None:
            try:
                optimizer_enc.load_state_dict(optimizer_state)
                print("Restored optimizer state from checkpoint")
            except Exception as e:
                print(f"Could not restore optimizer ({e}), using fresh optimizer")

        # ── Step 5: Load datasets from HuggingFace ──
        train_dataset = GeobodyHFDataset(split="train")
        val_dataset   = GeobodyHFDataset(split="validation")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

        # ── Step 6: Train ──
        print(f"Training from epoch {start_epoch} to {epochs}...")
        best_val_acc = 0.0
        final_loss = 0.0

        for epoch in range(start_epoch, epochs):
            train_loss, train_acc = train_epoch(
                m, classifier, train_loader,
                optimizer_enc, optimizer_cls, criterion
            )
            val_loss, val_acc = val_epoch(m, classifier, val_loader, criterion)
            final_loss = train_loss

            print(f"Epoch {epoch+1}/{epochs} — "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if train_loss == train_loss and train_loss != float('inf'):
                run.log_metrics({
                    "train_loss": float(train_loss),
                    "train_accuracy": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # ── Save checkpoint every N epochs ──
            if (epoch + 1) % checkpoint_every == 0:
                save_checkpoint(run, m, optimizer_enc, epoch, train_loss)

        # ── Step 7: Save final model ──
        output_dir = "/output"
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            "model": m.state_dict(),
            "classifier": classifier.state_dict(),
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            "arch": arch,
            "img_size": img_size,
            "epochs_trained": epochs,
            "best_val_acc": best_val_acc,
        }, os.path.join(output_dir, "finetuned.pth"))

        run.log_metrics({
            "final_loss": float(final_loss),
            "best_val_accuracy": float(best_val_acc),
        })

        mv = run.log_model(
            name=output_name,
            model_file_or_folder=output_dir,
            framework=PyTorchFramework(),
            metadata={
                "base_model_fqn": str(model_fqn),
                "arch": str(arch),
                "img_size": int(img_size),
                "epochs_trained": int(epochs),
                "lr": float(lr),
                "num_classes": int(num_classes),
                "dataset": "porestar/seismicfoundationmodel-geobody",
                "best_val_accuracy": float(best_val_acc),
            }
        )
        print(f"Done — logged as: {mv.fqn}")

    finally:
        run.end()


if __name__ == "__main__":
    main()
