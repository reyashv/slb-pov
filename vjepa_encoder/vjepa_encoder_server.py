import os
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

try:
    from transformers import AutoVideoProcessor
    PROCESSOR_CLASS = AutoVideoProcessor
except ImportError:
    from transformers import AutoProcessor
    PROCESSOR_CLASS = AutoProcessor

from transformers import AutoModel

model = None
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    model_dir = os.environ["MODEL_DIR"]
    print(f"Loading V-JEPA encoder from {model_dir}...")
    model = AutoModel.from_pretrained(model_dir).cuda().eval()
    processor = PROCESSOR_CLASS.from_pretrained(model_dir)
    print("V-JEPA encoder ready")
    yield

app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    frames: List[List[float]]  # each frame flat list of pixel values 0-1
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
    # Convert float32 0-1 → uint8 0-255 as processor expects
    frames_np = []
    for f in req.frames:
        arr = np.array(f, dtype=np.float32).reshape(req.height, req.width, req.channels)
        arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
        frames_np.append(arr_uint8)

    # processor expects list of numpy arrays [H, W, C] uint8
    inputs = processor(frames_np, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Handle different output formats
    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    elif hasattr(outputs, "pooler_output"):
        hidden = outputs.pooler_output.unsqueeze(1)
    else:
        # fallback — take first tensor output
        hidden = list(outputs.values())[0]

    # Pool over patch tokens → one vector per clip
    embeddings = hidden.mean(dim=1)

    return EmbedResponse(
        embeddings=embeddings.cpu().tolist(),
        dim=embeddings.shape[-1]
    )
