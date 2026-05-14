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
NUM_ANCHORS = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global head, ENCODER_URL, NUM_ANCHORS
    ENCODER_URL = os.environ["ENCODER_URL"]
    embed_dim = int(os.environ.get("EMBED_DIM", "1536"))
    NUM_ANCHORS = int(os.environ.get("NUM_ANCHORS", "10"))

    # Each anchor predicts [x, y, w, h, confidence] = 5 values
    head = nn.Linear(embed_dim, NUM_ANCHORS * 5).cuda()

    model_dir = os.environ.get("MODEL_DIR", "")
    if model_dir:
        ckpt_path = os.path.join(model_dir, "det_head.pth")
        if os.path.exists(ckpt_path):
            head.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
            print(f"Loaded detection head from {ckpt_path}")

    head.eval()
    print(f"Detection head ready — {NUM_ANCHORS} anchors")
    print(f"Encoder URL: {ENCODER_URL}")
    yield

app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)

class DetectRequest(BaseModel):
    frames: List[List[float]]
    height: int
    width: int
    channels: int = 3

class BoundingBox(BaseModel):
    x: float      # normalised 0-1
    y: float
    w: float
    h: float
    confidence: float

class DetectResponse(BaseModel):
    boxes: List[BoundingBox]

@app.get("/health")
def health():
    return {"healthy": True}

@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
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

    embeddings = torch.tensor(
        resp.json()["embeddings"],
        dtype=torch.float32
    ).cuda()

    with torch.no_grad():
        raw = head(embeddings).reshape(-1, NUM_ANCHORS, 5)
        coords = torch.sigmoid(raw[..., :4])   # normalised x,y,w,h
        scores = torch.sigmoid(raw[..., 4])    # confidence

    boxes = []
    for i in range(NUM_ANCHORS):
        boxes.append(BoundingBox(
            x=float(coords[0, i, 0]),
            y=float(coords[0, i, 1]),
            w=float(coords[0, i, 2]),
            h=float(coords[0, i, 3]),
            confidence=float(scores[0, i])
        ))

    return DetectResponse(boxes=boxes)