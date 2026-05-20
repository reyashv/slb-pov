from truefoundry.ml import get_client

client = get_client()

mv = client.log_artifact(
    ml_repo="slb-pov",
    name="sfm-facies-data",
    artifact_paths=[
        (r"Facies\seismic", "seismic"),
        (r"Facies\label", "label")
    ]
)
print(f"Uploaded: {mv.fqn}")
# Will print: artifact:slb-pilot/slb-pov/sfm-facies-data:1