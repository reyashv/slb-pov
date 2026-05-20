import os
import torch
import torch.nn as nn
import numpy as np


def main():
    from truefoundry.ml import get_client, PyTorchFramework
    from transformers import AutoModel, AutoVideoProcessor

    # Read env vars
    model_fqn = os.environ["MODEL_DIR"]
    embed_dim = int(os.environ.get("EMBED_DIM", "1024"))
    num_classes = int(os.environ.get("NUM_CLASSES", "10"))
    epochs = int(os.environ.get("EPOCHS", "10"))
    lr = float(os.environ.get("LR", "1e-3"))
    output_name = os.environ.get("OUTPUT_MODEL_NAME", "decoder-classify-trained")

    client = get_client()

    # Start a run — creates run ID that links to the model version
    run = client.create_run(
        ml_repo="slb-pov",
        run_name="vjepa-probe-train"
    )
    print(f"Started run: {run.run_id}")

    try:
        # Log parameters
        run.log_params({
            "base_encoder_fqn": model_fqn,
            "embed_dim": embed_dim,
            "num_classes": num_classes,
            "epochs": epochs,
            "lr": lr,
            "training_type": "probe_only_frozen_encoder"
        })

        # Step 1 — Download V-JEPA encoder from ML Repo
        print(f"Downloading encoder from {model_fqn}...")
        os.makedirs("/tmp/vjepa-model", exist_ok=True)
        download_info = client.get_model_version_by_fqn(model_fqn).download(
            path="/tmp/vjepa-model"
        )
        model_dir = download_info.download_dir
        print(f"Encoder downloaded to {model_dir}")

        # Step 2 — Load frozen V-JEPA encoder
        print("Loading V-JEPA encoder (will be frozen)...")
        processor = AutoVideoProcessor.from_pretrained(model_dir)
        encoder = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        ).cuda().eval()

        # Freeze encoder
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen — only probe head will be trained")

        # Step 3 — Build classification probe head
        head = nn.Linear(embed_dim, num_classes).cuda()
        optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Step 4 — Create placeholder dataset
        print("Creating placeholder dataset...")
        dataset = [
            (
                torch.randint(0, 255, (8, 3, 64, 64), dtype=torch.uint8).numpy(),
                torch.randint(0, num_classes, (1,)).item()
            )
            for _ in range(50)
        ]

        # Step 5 — Train probe
        print(f"Training probe for {epochs} epochs...")
        final_loss = 0
        for epoch in range(epochs):
            total_loss = 0
            head.train()

            for frames_np, label in dataset:
                inputs = processor(frames_np, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    embeddings = encoder.get_vision_features(**inputs)
                    pooled = embeddings.mean(dim=1).float()

                optimizer.zero_grad()
                logits = head(pooled)
                loss = criterion(logits, torch.tensor([label]).cuda())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataset)
            final_loss = avg_loss
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            # Log loss per epoch to the run
            run.log_metrics({"train_loss": avg_loss}, step=epoch)

        # Log final metrics
        run.log_metrics({"final_loss": final_loss})

        # Step 6 — Save probe weights
        output_dir = "/output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "probe.pth")
        torch.save(head.state_dict(), output_path)
        print(f"Saved probe weights to {output_path}")

        # Step 7 — Log to ML Repo with run_id for lineage
        print(f"Logging probe model as {output_name}...")
        mv = run.log_model(
            name=output_name,
            model_file_or_folder=output_dir,
            framework=PyTorchFramework(),
            metadata={
                "base_encoder_fqn": model_fqn,
                "embed_dim": embed_dim,
                "num_classes": num_classes,
                "epochs": epochs,
                "lr": lr,
                "training_type": "probe_only_frozen_encoder"
            }
        )
        print(f"Done — logged as: {mv.fqn}")

    finally:
        run.end()


if __name__ == "__main__":
    main()
