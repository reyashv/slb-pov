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
OUT_H = None
OUT_W = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, processor, head, OUT_H, OUT_W

    encoder_dir = os.environ["ENCODER_MODEL_DIR"]
    print(f"Loading V-JEPA encoder from {encoder_dir}...")
    processor = AutoVideoProcessor.from_pretrained(encoder_dir)
    encoder = AutoModel.from_pretrained(
        encoder_dir,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    ).to(DEVICE).eval()

    embed_dim = int(os.environ.get("EMBED_DIM", "1024"))
    OUT_H = int(os.environ.get("OUT_HEIGHT", "64"))
    OUT_W = int(os.environ.get("OUT_WIDTH", "64"))

    head = nn.Sequential(
        nn.Linear(embed_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, OUT_H * OUT_W)
    ).to(DEVICE).eval()

    decoder_dir = os.environ.get("DECODER_MODEL_DIR", "")
    if decoder_dir:
        pth_files = glob.glob(os.path.join(decoder_dir, "*.pth"))
        if pth_files:
            head.load_state_dict(torch.load(pth_files[0], map_location=DEVICE))
            print(f"Loaded decoder weights from {pth_files[0]}")

    print(f"Fused segment ready — output {OUT_H}x{OUT_W}, device={DEVICE}")
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
    mask: List[float]
    out_height: int
    out_width: int

class BatchSegmentRequest(BaseModel):
    videos: List[SegmentRequest]

class BatchSegmentResponse(BaseModel):
    results: List[SegmentResponse]

@app.get("/health")
def health():
    return {"healthy": True}

def _process_frames(req: SegmentRequest):
    frames_tchw = []
    for f in req.frames:
        arr_hwc = np.array(f, dtype=np.float32).reshape(req.height, req.width, req.channels)
        arr_uint8 = (arr_hwc * 255).clip(0, 255).astype(np.uint8)
        frames_tchw.append(arr_uint8.transpose(2, 0, 1))
    return np.stack(frames_tchw, axis=0)

@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest):
    video = _process_frames(req)
    video_inputs = processor(video, return_tensors="pt")
    video_inputs = {k: v.to(DEVICE) for k, v in video_inputs.items()}

    with torch.no_grad():
        embeddings = encoder.get_vision_features(**video_inputs)
        pooled = embeddings.mean(dim=1).float()
        mask_logits = head(pooled)
        mask = torch.sigmoid(mask_logits).squeeze(0)

    return SegmentResponse(
        mask=mask.cpu().tolist(),
        out_height=OUT_H,
        out_width=OUT_W
    )

@app.post("/batch_segment", response_model=BatchSegmentResponse)
def batch_segment(req: BatchSegmentRequest):
    results = []
    all_videos = [_process_frames(v) for v in req.videos]

    with torch.no_grad():
        for video in all_videos:
            inputs = processor(video, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            embeddings = encoder.get_vision_features(**inputs)
            pooled = embeddings.mean(dim=1).float()
            mask_logits = head(pooled)
            mask = torch.sigmoid(mask_logits).squeeze(0)
            results.append(SegmentResponse(
                mask=mask.cpu().tolist(),
                out_height=OUT_H,
                out_width=OUT_W
            ))

    return BatchSegmentResponse(results=results)
