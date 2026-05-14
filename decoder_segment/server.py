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
OUT_H = None
OUT_W = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global head, ENCODER_URL, OUT_H, OUT_W
    ENCODER_URL = os.environ["ENCODER_URL"]
    embed_dim = int(os.environ.get("EMBED_DIM", "1536"))
    OUT_H = int(os.environ.get("OUT_HEIGHT", "224"))
    OUT_W = int(os.environ.get("OUT_WIDTH", "224"))

    # Simple MLP segmentation head — replace with UPerNet for production
    head = nn.Sequential(
        nn.Linear(embed_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, OUT_H * OUT_W)
    ).cuda()

    model_dir = os.environ.get("MODEL_DIR", "")
    if model_dir:
        ckpt_path = os.path.join(model_dir, "seg_head.pth")
        if os.path.exists(ckpt_path):
            head.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
            print(f"Loaded segmentation head from {ckpt_path}")

    head.eval()
    print(f"Segmentation head ready — output {OUT_H}x{OUT_W}")
    print(f"Encoder URL: {ENCODER_URL}")
    yield

app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)

class SegmentRequest(BaseModel):
    frames: List[List[float]]
    height: int
    width: int
    channels: int = 3

class SegmentResponse(BaseModel):
    # 2D mask returned as flat list — reshape to OUT_H x OUT_W on client side
    mask: List[float]
    out_height: int
    out_width: int

@app.get("/health")
def health():
    return {"healthy": True}

@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
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
        mask_logits = head(embeddings)
        mask = torch.sigmoid(mask_logits).squeeze(0)

    return SegmentResponse(
        mask=mask.cpu().tolist(),
        out_height=OUT_H,
        out_width=OUT_W
    )