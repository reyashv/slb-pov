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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, processor, head

    # Load encoder
    encoder_dir = os.environ["ENCODER_MODEL_DIR"]
    print(f"Loading V-JEPA encoder from {encoder_dir}...")
    processor = AutoVideoProcessor.from_pretrained(encoder_dir)
    encoder = AutoModel.from_pretrained(
        encoder_dir,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    ).to(DEVICE).eval()

    # Load classification head
    embed_dim = int(os.environ.get("EMBED_DIM", "1024"))
    num_classes = int(os.environ.get("NUM_CLASSES", "10"))
    head = nn.Linear(embed_dim, num_classes).to(DEVICE).eval()

    # Load trained probe weights if available
    decoder_dir = os.environ.get("DECODER_MODEL_DIR", "")
    if decoder_dir:
        pth_files = glob.glob(os.path.join(decoder_dir, "*.pth"))
        if pth_files:
            head.load_state_dict(torch.load(pth_files[0], map_location=DEVICE))
            print(f"Loaded decoder weights from {pth_files[0]}")

    print(f"Fused classify ready — device={DEVICE}")
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

class BatchClassifyRequest(BaseModel):
    videos: List[ClassifyRequest]

class BatchClassifyResponse(BaseModel):
    results: List[ClassifyResponse]

@app.get("/health")
def health():
    return {"healthy": True}

def _process_frames(req: ClassifyRequest):
    frames_tchw = []
    for f in req.frames:
        arr_hwc = np.array(f, dtype=np.float32).reshape(req.height, req.width, req.channels)
        arr_uint8 = (arr_hwc * 255).clip(0, 255).astype(np.uint8)
        frames_tchw.append(arr_uint8.transpose(2, 0, 1))
    return np.stack(frames_tchw, axis=0)

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    video = _process_frames(req)
    video_inputs = processor(video, return_tensors="pt")
    video_inputs = {k: v.to(DEVICE) for k, v in video_inputs.items()}

    with torch.no_grad():
        embeddings = encoder.get_vision_features(**video_inputs)
        pooled = embeddings.mean(dim=1).float()
        logits = head(pooled)
        probs = torch.softmax(logits, dim=-1)

    return ClassifyResponse(
        class_id=int(logits.argmax(-1).item()),
        confidence=float(probs.max().item()),
        logits=logits.squeeze(0).cpu().tolist()
    )

@app.post("/batch_classify", response_model=BatchClassifyResponse)
def batch_classify(req: BatchClassifyRequest):
    # All videos processed in one forward pass — no HTTP hop
    all_videos = [_process_frames(v) for v in req.videos]

    results = []
    # Process each video — batch them through encoder together
    all_inputs = [processor(v, return_tensors="pt") for v in all_videos]

    with torch.no_grad():
        for inputs in all_inputs:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            embeddings = encoder.get_vision_features(**inputs)
            pooled = embeddings.mean(dim=1).float()
            logits = head(pooled)
            probs = torch.softmax(logits, dim=-1)
            results.append(ClassifyResponse(
                class_id=int(logits.argmax(-1).item()),
                confidence=float(probs.max().item()),
                logits=logits.squeeze(0).cpu().tolist()
            ))

    return BatchClassifyResponse(results=results)
