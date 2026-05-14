from truefoundry.ml import get_client, PyTorchFramework

client = get_client()

variants = [
    ("sfm-base",      "./checkpoints/sfm-base/checkpoint.pth"),
    ("sfm-base-512",  "./checkpoints/sfm-base-512/checkpoint.pth"),
    ("sfm-large",     "./checkpoints/sfm-large/checkpoint.pth"),
    ("sfm-large-512", "./checkpoints/sfm-large-512/checkpoint.pth"),
]

for name, path in variants:
    mv = client.log_model(
        ml_repo="slb-pov",
        name=name,
        model_file_or_folder=path,
        framework=PyTorchFramework(),
        metadata={
            "source": "USTC SeismicFoundationModel",
            "input_size": "512x512" if "512" in name else "224x224"
        }
    )
    print(f"{name}: {mv.fqn}")
    # Save these FQNs — you'll paste them into the UI later