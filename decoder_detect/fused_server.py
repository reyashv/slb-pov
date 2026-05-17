import os
import glob
import torch
import torch.nn as nn
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoVideoProcessor, AutoModel

encoder = None
processor = None
head = None
NUM_ANCHORS = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, processor, head, NUM_ANCHORS

    encoder_dir = os.environ["ENCODER_MODEL_DIR"]
    print(f"Loading V-JEPA encoder from {encoder_dir}...")
    processor = AutoVideoProcessor.from_pretrained(encoder_dir)
    encoder = AutoModel.from_pretrained(
        encoder_dir,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    ).to(DEVICE).eval()

    embed_dim = int(os.environ.get("EMBED_DIM", "1024"))
    NUM_ANCHORS = int(os.environ.get("NUM_ANCHORS", "10"))

    head = nn.Linear(embed_dim, NUM_ANCHORS * 5).to(DEVICE).eval()

    decoder_dir = os.environ.get("DECODER_MODEL_DIR", "")
    if decoder_dir:
        pth_files = glob.glob(os.path.join(decoder_dir, "*.pth"))
        if pth_files:
            head.load_state_dict(torch.load(pth_files[0], map_location=DEVICE))
            print(f"Loaded decoder weights from {pth_files[0]}")

    print(f"Fused detect ready — {NUM_ANCHORS} anchors, device={DEVICE}")
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
    x: float
    y: float
    w: float
    h: float
    confidence: float

class DetectResponse(BaseModel):
    boxes: List[BoundingBox]

class BatchDetectRequest(BaseModel):
    videos: List[DetectRequest]

class BatchDetectResponse(BaseModel):
    results: List[DetectResponse]

@app.get("/health")
def health():
    return {"healthy": True}

def _process_frames(req: DetectRequest):
    frames_tchw = []
    for f in req.frames:
        arr_hwc = np.array(f, dtype=np.float32).reshape(req.height, req.width, req.channels)
        arr_uint8 = (arr_hwc * 255).clip(0, 255).astype(np.uint8)
        frames_tchw.append(arr_uint8.transpose(2, 0, 1))
    return np.stack(frames_tchw, axis=0)

def _run_detect(video):
    inputs = processor(video, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        embeddings = encoder.get_vision_features(**inputs)
        pooled = embeddings.mean(dim=1).float()
        raw = head(pooled).reshape(-1, NUM_ANCHORS, 5)
        coords = torch.sigmoid(raw[..., :4])
        scores = torch.sigmoid(raw[..., 4])

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

@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    video = _process_frames(req)
    return _run_detect(video)

@app.post("/batch_detect", response_model=BatchDetectResponse)
def batch_detect(req: BatchDetectRequest):
    results = []
    for v in req.videos:
        video = _process_frames(v)
        results.append(_run_detect(video))
    return BatchDetectResponse(results=results)
