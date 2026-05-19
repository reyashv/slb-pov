import os
import torch
import torch.nn as nn
import numpy as np


def main():
    from truefoundry.ml import get_client, PyTorchFramework
    from transformers import AutoModel, AutoVideoProcessor

    model_fqn = os.environ["MODEL_DIR"]
    embed_dim = int(os.environ.get("EMBED_DIM", "1024"))
    num_classes = int(os.environ.get("NUM_CLASSES", "10"))
    epochs = int(os.environ.get("EPOCHS", "5"))
    encoder_lr = float(os.environ.get("ENCODER_LR", "1e-5"))
    head_lr = float(os.environ.get("HEAD_LR", "1e-3"))
    output_encoder_name = os.environ.get("OUTPUT_ENCODER_NAME", "vjepa2-vitl-finetuned")
    output_decoder_name = os.environ.get("OUTPUT_DECODER_NAME", "decoder-classify-joint")

    client = get_client()

    run = client.create_run(
        ml_repo="slb-pov",
        run_name="vjepa-joint-finetune"
    )
    print(f"Started run: {run.run_id}")

    try:
        run.log_params({
            "base_encoder_fqn": str(model_fqn),
            "embed_dim": int(embed_dim),
            "num_classes": int(num_classes),
            "epochs": int(epochs),
            "encoder_lr": float(encoder_lr),
            "head_lr": float(head_lr),
            "training_type": "joint_encoder_decoder"
        })

        # Step 1 — Download encoder
        print(f"Downloading encoder from {model_fqn}...")
        os.makedirs("/tmp/vjepa-model", exist_ok=True)
        download_info = client.get_model_version_by_fqn(model_fqn).download(
            path="/tmp/vjepa-model"
        )
        model_dir = download_info.download_dir

        # Step 2 — Load encoder — NOT frozen
        print("Loading V-JEPA encoder (will be fine-tuned)...")
        processor = AutoVideoProcessor.from_pretrained(model_dir)
        encoder = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        ).cuda().train()

        # Step 3 — Build head
        head = nn.Linear(embed_dim, num_classes).cuda()

        # Two separate optimizers — different learning rates
        encoder_optimizer = torch.optim.AdamW(
            encoder.parameters(), lr=encoder_lr
        )
        head_optimizer = torch.optim.Adam(
            head.parameters(), lr=head_lr
        )
        criterion = nn.CrossEntropyLoss()

        # Step 4 — Placeholder dataset
        print("Creating placeholder dataset...")
        dataset = [
            (
                torch.randint(0, 255, (8, 3, 64, 64), dtype=torch.uint8).numpy(),
                torch.randint(0, num_classes, (1,)).item()
            )
            for _ in range(50)
        ]

        # Step 5 — Joint training
        print(f"Joint fine-tuning for {epochs} epochs...")
        final_loss = 0.0
        for epoch in range(epochs):
            total_loss = 0.0
            encoder.train()
            head.train()

            for frames_np, label in dataset:
                inputs = processor(frames_np, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}

                encoder_optimizer.zero_grad()
                head_optimizer.zero_grad()

                embeddings = encoder.get_vision_features(**inputs)
                pooled = embeddings.mean(dim=1).float()

                logits = head(pooled)
                loss = criterion(logits, torch.tensor([label]).cuda())
                loss.backward()

                encoder_optimizer.step()
                head_optimizer.step()
                total_loss += float(loss.item())

            avg_loss = total_loss / len(dataset)
            final_loss = avg_loss
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            run.log_metrics({"train_loss": float(avg_loss)}, step=epoch)

        run.log_metrics({"final_loss": float(final_loss)})

        # Step 6 — Save both encoder and decoder
        output_dir = "/output"
        os.makedirs(f"{output_dir}/encoder", exist_ok=True)
        os.makedirs(f"{output_dir}/decoder", exist_ok=True)

        encoder.save_pretrained(f"{output_dir}/encoder")
        processor.save_pretrained(f"{output_dir}/encoder")
        print("Saved fine-tuned encoder")

        torch.save(head.state_dict(), f"{output_dir}/decoder/probe.pth")
        print("Saved fine-tuned decoder head")

        # Step 7 — Log both to ML Repo under same run
        print(f"Logging fine-tuned encoder as {output_encoder_name}...")
        mv_encoder = run.log_model(
            name=output_encoder_name,
            model_file_or_folder=f"{output_dir}/encoder",
            framework=PyTorchFramework(),
            metadata={
                "base_model_fqn": str(model_fqn),
                "training_type": "joint_encoder_decoder",
                "epochs": int(epochs),
                "encoder_lr": float(encoder_lr)
            }
        )
        print(f"Encoder logged as: {mv_encoder.fqn}")

        print(f"Logging fine-tuned decoder as {output_decoder_name}...")
        mv_decoder = run.log_model(
            name=output_decoder_name,
            model_file_or_folder=f"{output_dir}/decoder",
            framework=PyTorchFramework(),
            metadata={
                "base_encoder_fqn": str(model_fqn),
                "training_type": "joint_encoder_decoder",
                "epochs": int(epochs),
                "head_lr": float(head_lr)
            }
        )
        print(f"Decoder logged as: {mv_decoder.fqn}")
        print(f"Both logged under run: {run.run_id}")

    finally:
        run.end()


if __name__ == "__main__":
    main()
