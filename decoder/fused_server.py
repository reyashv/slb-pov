import os
import glob
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from transformers import AutoVideoProcessor, AutoModel
from heads import HEAD_REGISTRY

encoder = None
processor = None
head_obj = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEAD_TYPE = os.environ.get("HEAD_TYPE", "classify")
assert HEAD_TYPE in HEAD_REGISTRY, (
    f"HEAD_TYPE must be one of {list(HEAD_REGISTRY)}, got '{HEAD_TYPE}'"
)

cfg = HEAD_REGISTRY[HEAD_TYPE]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, processor, head_obj

    encoder_dir = os.environ["ENCODER_MODEL_DIR"]
    print(f"Loading V-JEPA encoder from {encoder_dir}...")
    processor = AutoVideoProcessor.from_pretrained(encoder_dir)
    encoder = AutoModel.from_pretrained(
        encoder_dir,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    ).to(DEVICE).eval()

    head_obj = cfg["head_cls"](DEVICE)

    decoder_dir = os.environ.get("DECODER_MODEL_DIR", "")
    if decoder_dir:
        # Try glob first (checkpoint pattern), then named weight file
        pth_files = glob.glob(os.path.join(decoder_dir, "*.pth"))
        if pth_files:
            head_obj.load_weights(decoder_dir)
        else:
            head_obj.load_weights(decoder_dir)

    print(f"Fused decoder ready — HEAD_TYPE={HEAD_TYPE}, device={DEVICE}")
    yield


app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"healthy": True}


def _process_frames(req) -> np.ndarray:
    frames_tchw = []
    for f in req.frames:
        arr_hwc = np.array(f, dtype=np.float32).reshape(req.height, req.width, req.channels)
        arr_uint8 = (arr_hwc * 255).clip(0, 255).astype(np.uint8)
        frames_tchw.append(arr_uint8.transpose(2, 0, 1))
    return np.stack(frames_tchw, axis=0)


def _encode(video: np.ndarray) -> torch.Tensor:
    inputs = processor(video, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = encoder.get_vision_features(**inputs)
        pooled = embeddings.mean(dim=1).float()
    return pooled


@app.post(cfg["route"], response_model=cfg["response_cls"])
def infer(req: cfg["request_cls"]):
    video = _process_frames(req)
    embeddings = _encode(video)
    return head_obj.infer(embeddings)


@app.post(cfg["batch_route"], response_model=cfg["batch_response_cls"])
def batch_infer(req: cfg["batch_request_cls"]):
    results = []
    for video_req in req.videos:
        video = _process_frames(video_req)
        embeddings = _encode(video)
        results.append(head_obj.infer(embeddings))
    return cfg["batch_response_cls"](results=results)
