import os
import glob
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from truefoundry.ml import get_client, PyTorchFramework
import timm.models.vision_transformer

# Same VisionTransformer class as sfm_server.py
# Copy the entire class here
# ... (same as sfm_server.py)

def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            features = model.forward_features(inputs)
            # Add a classification head on top
            # In real training SLB would have their own head
            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    return model

def main():
    # Load model from ML Repo
    model_dir = os.environ["MODEL_DIR"]
    arch = os.environ.get("SFM_ARCH", "vit_base_patch16")
    img_size = int(os.environ.get("SFM_IMG_SIZE", "224"))
    epochs = int(os.environ.get("EPOCHS", "10"))
    lr = float(os.environ.get("LR", "1e-4"))
    output_name = os.environ.get("OUTPUT_MODEL_NAME", "sfm-base-finetuned")

    # Build model
    m = MODEL_REGISTRY[arch](
        num_classes=0,
        global_pool=False,
        in_chans=1,
        img_size=img_size
    )

    # Load checkpoint
    pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
    checkpoint = torch.load(pth_files[0], map_location="cuda")
    state_dict = checkpoint.get("model", checkpoint)
    m.load_state_dict(state_dict, strict=False)
    m.cuda()

    # Optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
    criterion = nn.MSELoss()  # placeholder — SLB would use their own loss

    # Dataset — placeholder using random data
    # SLB would replace this with their actual seismic dataset
    print("Creating placeholder dataset...")
    dataset = [(
        torch.randn(1, img_size, img_size),
        torch.randn(768)  # target features
    ) for _ in range(100)]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

    # Train
    print(f"Starting fine-tuning for {epochs} epochs...")
    m = train(m, dataloader, optimizer, criterion, epochs)

    # Save checkpoint
    output_path = "/output/finetuned.pth"
    os.makedirs("/output", exist_ok=True)
    torch.save({"model": m.state_dict()}, output_path)
    print(f"Saved checkpoint to {output_path}")

    # Log new version to ML Repo
    client = get_client()
    mv = client.log_model(
        ml_repo="slb-pov",
        name=output_name,
        model_file_or_folder="/output",
        framework=PyTorchFramework(),
        metadata={
            "base_model": os.environ.get("MODEL_DIR", ""),
            "epochs": epochs,
            "lr": lr,
            "arch": arch
        }
    )
    print(f"Logged fine-tuned model: {mv.fqn}")

if __name__ == "__main__":
    main()
