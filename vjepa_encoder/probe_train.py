import os
import torch
import torch.nn as nn
from truefoundry.ml import get_client, PyTorchFramework
from transformers import AutoModel, AutoVideoProcessor
import numpy as np

def main():
    model_dir = os.environ["MODEL_DIR"]
    embed_dim = int(os.environ.get("EMBED_DIM", "1024"))
    num_classes = int(os.environ.get("NUM_CLASSES", "10"))
    epochs = int(os.environ.get("EPOCHS", "10"))
    lr = float(os.environ.get("LR", "1e-3"))
    output_name = os.environ.get("OUTPUT_MODEL_NAME", "decoder-classify-trained")

    # Load frozen encoder
    print("Loading V-JEPA encoder (frozen)...")
    processor = AutoVideoProcessor.from_pretrained(model_dir)
    encoder = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    ).cuda().eval()

    # Freeze encoder — no gradient updates
    for param in encoder.parameters():
        param.requires_grad = False

    # Classification probe head
    head = nn.Linear(embed_dim, num_classes).cuda()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Placeholder dataset
    print("Creating placeholder dataset...")
    dataset = [(
        torch.randn(8, 3, 64, 64),  # 8 frames, RGB, 64x64
        torch.randint(0, num_classes, (1,)).item()  # label
    ) for _ in range(50)]

    # Train probe
    print(f"Training probe for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for frames, label in dataset:
            # Get embeddings from frozen encoder
            frames_np = (frames.numpy() * 255).clip(0, 255).astype(np.uint8)
            inputs = processor(frames_np, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = encoder.get_vision_features(**inputs)
                pooled = embeddings.mean(dim=1).float()

            # Train only the head
            optimizer.zero_grad()
            logits = head(pooled)
            loss = criterion(logits, torch.tensor([label]).cuda())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.4f}")

    # Save probe weights
    os.makedirs("/output", exist_ok=True)
    torch.save(head.state_dict(), "/output/probe.pth")
    print("Saved probe weights to /output/probe.pth")

    # Log to ML Repo
    client = get_client()
    mv = client.log_model(
        ml_repo="slb-pov",
        name=output_name,
        model_file_or_folder="/output",
        framework=PyTorchFramework(),
        metadata={
            "base_encoder": model_dir,
            "epochs": epochs,
            "lr": lr,
            "num_classes": num_classes,
            "training_type": "probe_only_frozen_encoder"
        }
    )
    print(f"Logged probe model: {mv.fqn}")

if __name__ == "__main__":
    main()
