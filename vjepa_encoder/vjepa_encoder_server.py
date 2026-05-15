import os
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoVideoProcessor, AutoModel

model = None
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    model_dir = os.environ["MODEL_DIR"]
    print(f"Loading V-JEPA encoder from {model_dir}...")
    processor = AutoVideoProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    ).cuda().eval()
    print("V-JEPA encoder ready")
    yield

app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    # Each frame as flat list of pixel values 0-1 in H*W*C order
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
    # Convert each frame from flat H*W*C float 0-1 → [C, H, W] uint8
    frames_tchw = []
    for f in req.frames:
        arr_hwc = np.array(f, dtype=np.float32).reshape(req.height, req.width, req.channels)
        arr_uint8 = (arr_hwc * 255).clip(0, 255).astype(np.uint8)
        arr_chw = arr_uint8.transpose(2, 0, 1)  # [H,W,C] → [C,H,W]
        frames_tchw.append(arr_chw)

    # Stack to [T, C, H, W] — format V-JEPA processor expects
    video = np.stack(frames_tchw, axis=0)

    # Process frames
    video_inputs = processor(video, return_tensors="pt")
    video_inputs = {k: v.cuda() for k, v in video_inputs.items()}

    with torch.no_grad():
        embeddings = model.get_vision_features(**video_inputs)

    # Pool over patch tokens → [batch, embed_dim]
    pooled = embeddings.mean(dim=1).float()

    return EmbedResponse(
        embeddings=pooled.cpu().tolist(),
        dim=pooled.shape[-1]
    )
