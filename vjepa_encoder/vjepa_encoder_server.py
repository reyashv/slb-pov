import os
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoModel, AutoVideoProcessor

model = None
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    model_dir = os.environ["MODEL_DIR"]
    print(f"Loading V-JEPA encoder from {model_dir}...")
    model = AutoModel.from_pretrained(model_dir).cuda().eval()
    processor = AutoVideoProcessor.from_pretrained(model_dir)
    print("V-JEPA encoder ready")
    yield

app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    # Each frame is a flat list of pixel values (height * width * channels)
    frames: List[List[float]]
    height: int
    width: int
    channels: int = 3

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int

@app.get("/health")
def health():
    return {"healthy": True}

@app.post("/embeddings", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    # Reconstruct each frame as a numpy array [H, W, C]
    frames_np = [
        np.array(f, dtype=np.float32).reshape(req.height, req.width, req.channels)
        for f in req.frames
    ]
    inputs = processor(frames_np, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Pool over patch tokens → one vector per video clip
    # last_hidden_state shape: [batch, num_patches, embed_dim]
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return EmbedResponse(
        embeddings=embeddings.cpu().tolist(),
        dim=embeddings.shape[-1]
    )