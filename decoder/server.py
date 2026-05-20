import os
import torch
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from heads import HEAD_REGISTRY

head_obj = None
ENCODER_URL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEAD_TYPE = os.environ.get("HEAD_TYPE", "classify")
assert HEAD_TYPE in HEAD_REGISTRY, (
    f"HEAD_TYPE must be one of {list(HEAD_REGISTRY)}, got '{HEAD_TYPE}'"
)

cfg = HEAD_REGISTRY[HEAD_TYPE]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global head_obj, ENCODER_URL
    ENCODER_URL = os.environ["ENCODER_URL"]

    head_obj = cfg["head_cls"](DEVICE)

    model_dir = os.environ.get("MODEL_DIR", "")
    if model_dir:
        head_obj.load_weights(model_dir)

    print(f"Decoder ready — HEAD_TYPE={HEAD_TYPE}, device={DEVICE}")
    print(f"Encoder URL: {ENCODER_URL}")
    yield


app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"healthy": True}


async def _get_embeddings(req) -> torch.Tensor:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                f"{ENCODER_URL}/embeddings",
                json=req.model_dump()
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Encoder error: {e}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Could not reach encoder: {e}")
    return torch.tensor(resp.json()["embeddings"], dtype=torch.float32).to(DEVICE)


@app.post(cfg["route"], response_model=cfg["response_cls"])
async def infer(req: cfg["request_cls"]):
    embeddings = await _get_embeddings(req)
    return head_obj.infer(embeddings)


@app.post(cfg["batch_route"], response_model=cfg["batch_response_cls"])
async def batch_infer(req: cfg["batch_request_cls"]):
    results = []
    for video in req.videos:
        embeddings = await _get_embeddings(video)
        results.append(head_obj.infer(embeddings))
    return cfg["batch_response_cls"](results=results)
