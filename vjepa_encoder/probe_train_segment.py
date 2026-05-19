import os
import torch
import torch.nn as nn
import numpy as np


def main():
    from truefoundry.ml import get_client, PyTorchFramework
    from transformers import AutoModel, AutoVideoProcessor

    model_fqn = os.environ["MODEL_DIR"]
    embed_dim = int(os.environ.get("EMBED_DIM", "1024"))
    out_height = int(os.environ.get("OUT_HEIGHT", "64"))
    out_width = int(os.environ.get("OUT_WIDTH", "64"))
    epochs = int(os.environ.get("EPOCHS", "10"))
    lr = float(os.environ.get("LR", "1e-3"))
    output_name = os.environ.get("OUTPUT_MODEL_NAME", "decoder-segment-trained")

    client = get_client()

    run = client.create_run(
        ml_repo="slb-pov",
        run_name="vjepa-probe-segment"
    )
    print(f"Started run: {run.run_id}")

    try:
        run.log_params({
            "base_encoder_fqn": model_fqn,
            "embed_dim": embed_dim,
            "out_height": out_height,
            "out_width": out_width,
            "epochs": epochs,
            "lr": lr,
            "training_type": "segmentation_probe_frozen_encoder"
        })

        # Step 1 — Download encoder
        print(f"Downloading encoder from {model_fqn}...")
        os.makedirs("/tmp/vjepa-model", exist_ok=True)
        download_info = client.get_model_version_by_fqn(model_fqn).download(
            path="/tmp/vjepa-model"
        )
        model_dir = download_info.download_dir
        print(f"Encoder downloaded to {model_dir}")

        # Step 2 — Load frozen encoder
        print("Loading V-JEPA encoder (frozen)...")
        processor = AutoVideoProcessor.from_pretrained(model_dir)
        encoder = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        ).cuda().eval()

        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")

        # Step 3 — Segmentation head
        head = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_height * out_width)
        ).cuda()
        optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # Step 4 — Placeholder dataset
        print("Creating placeholder dataset...")
        dataset = [
            (
                torch.randint(0, 255, (8, 3, 64, 64), dtype=torch.uint8).numpy(),
                torch.randint(0, 2, (out_height * out_width,), dtype=torch.float32)
            )
            for _ in range(50)
        ]

        # Step 5 — Train
        print(f"Training segmentation probe for {epochs} epochs...")
        final_loss = 0
        for epoch in range(epochs):
            total_loss = 0
            head.train()

            for frames_np, mask_label in dataset:
                inputs = processor(frames_np, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    embeddings = encoder.get_vision_features(**inputs)
                    pooled = embeddings.mean(dim=1).float()

                optimizer.zero_grad()
                mask_pred = head(pooled).squeeze(0)
                loss = criterion(mask_pred, mask_label.cuda())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataset)
            final_loss = avg_loss
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            run.log_metrics({"train_loss": avg_loss}, step=epoch)

        run.log_metrics({"final_loss": final_loss})

        # Step 6 — Save
        output_dir = "/output"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(head.state_dict(), os.path.join(output_dir, "seg_head.pth"))
        print(f"Saved segmentation head to {output_dir}/seg_head.pth")

        # Step 7 — Log to ML Repo
        print(f"Logging segmentation model as {output_name}...")
        mv = run.log_model(
            name=output_name,
            model_file_or_folder=output_dir,
            framework=PyTorchFramework(),
            metadata={
                "base_encoder_fqn": model_fqn,
                "embed_dim": embed_dim,
                "out_height": out_height,
                "out_width": out_width,
                "epochs": epochs,
                "lr": lr,
                "training_type": "segmentation_probe_frozen_encoder"
            }
        )
        print(f"Done — logged as: {mv.fqn}")

    finally:
        run.end()


if __name__ == "__main__":
    main()
