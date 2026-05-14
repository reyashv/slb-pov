import os
import torch
import httpx
import torch.nn as nn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

head = None
ENCODER_URL = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global head, ENCODER_URL
    ENCODER_URL = os.environ["ENCODER_URL"]
    embed_dim = int(os.environ.get("EMBED_DIM", "1536"))
    num_classes = int(os.environ.get("NUM_CLASSES", "10"))

    head = nn.Linear(embed_dim, num_classes).cuda()

    # Load trained probe weights if a checkpoint exists
    model_dir = os.environ.get("MODEL_DIR", "")
    if model_dir:
        probe_path = os.path.join(model_dir, "probe.pth")
        if os.path.exists(probe_path):
            head.load_state_dict(torch.load(probe_path, map_location="cuda"))
            print(f"Loaded probe weights from {probe_path}")

    head.eval()
    print(f"Classifier head ready — {embed_dim}d → {num_classes} classes")
    print(f"Encoder URL: {ENCODER_URL}")
    yield

app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)

class ClassifyRequest(BaseModel):
    frames: List[List[float]]
    height: int
    width: int
    channels: int = 3

class ClassifyResponse(BaseModel):
    class_id: int
    confidence: float
    logits: List[float]

@app.get("/health")
def health():
    return {"healthy": True}

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    # Step 1 — call the encoder over the internal cluster network
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                f"{ENCODER_URL}/embeddings",
                json=req.model_dump()
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Encoder returned error: {e}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Could not reach encoder: {e}")

    # Step 2 — run through the classification head
    embeddings = torch.tensor(
        resp.json()["embeddings"],
        dtype=torch.float32
    ).cuda()

    with torch.no_grad():
        logits = head(embeddings)
        probs = torch.softmax(logits, dim=-1)

    return ClassifyResponse(
        class_id=int(logits.argmax(-1).item()),
        confidence=float(probs.max().item()),
        logits=logits.squeeze(0).cpu().tolist()
    )