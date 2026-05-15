import os
import sys
import subprocess
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Clone SFM repo if not already present so models_vit.py is available
SFM_REPO_PATH = "/app/SeismicFoundationModel"
if not os.path.exists(SFM_REPO_PATH):
    subprocess.run(
        ["git", "clone", "https://github.com/shenghanlin/SeismicFoundationModel.git", SFM_REPO_PATH],
        check=True
    )

sys.path.insert(0, f"{SFM_REPO_PATH}/SFM-Finetune")
import models_vit

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_dir = os.environ["MODEL_DIR"]
    arch = os.environ.get("SFM_ARCH", "vit_base_patch16")

    m = models_vit.__dict__[arch](num_classes=0, global_pool=False)

    checkpoint = torch.load(
        os.path.join(model_dir, "checkpoint.pth"),
        map_location="cuda"
    )
    # SFM checkpoints store weights under a 'model' key
    state_dict = checkpoint.get("model", checkpoint)
    m.load_state_dict(state_dict, strict=False)
    m.eval().cuda()
    model = m
    print(f"Loaded SFM arch={arch} from {model_dir}")
    yield

app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)

class InferRequest(BaseModel):
    data: List[float]   # 2D seismic slice flattened row by row
    height: int
    width: int

class InferResponse(BaseModel):
    features: List[List[float]]

@app.get("/health")
def health():
    return {"healthy": True}

@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    arr = np.array(req.data, dtype=np.float32).reshape(req.height, req.width)
    # [H, W] → [1, 1, H, W] (batch=1, channels=1)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        features = model.forward_features(tensor)
    return InferResponse(features=features.cpu().tolist())
