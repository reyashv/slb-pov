"""
SFM encoder server using TensorRT FP16 engine.
Supports both single slice (/infer) and batch inference (/batch_infer).
"""

import os
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from truefoundry.ml import get_client

model = None
context = None
engine_global = None
DEVICE = "cuda"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, context, engine_global

    import tensorrt as trt

    engine_artifact_fqn = os.environ["TRT_ENGINE_FQN"]
    img_size = int(os.environ.get("SFM_IMG_SIZE", "224"))

    print(f"Downloading TRT engine from {engine_artifact_fqn}...")
    client = get_client()
    os.makedirs("/tmp/trt-engine", exist_ok=True)
    av = client.get_artifact_version_by_fqn(fqn=engine_artifact_fqn)
    av.download(path="/tmp/trt-engine")

    trt_path = "/tmp/trt-engine/sfm.trt"
    if not os.path.exists(trt_path):
        raise FileNotFoundError(f"sfm.trt not found after download")

    print("Loading TensorRT engine...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    engine_global = engine

    print(f"TRT engine ready — img_size={img_size}, precision=FP16")
    yield
    del context
    del engine


app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)


# ── Single slice inference ────────────────────────────────────────────────────

class InferRequest(BaseModel):
    data: List[float]
    height: int
    width: int


class InferResponse(BaseModel):
    features: List[float]


@app.get("/health")
def health():
    return {"healthy": True}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    arr = np.array(req.data, dtype=np.float16).reshape(1, 1, req.height, req.width)
    input_tensor = torch.from_numpy(arr).cuda()

    engine = context.engine
    output_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
    output_shape = (1,) + output_shape[1:]
    output_tensor = torch.empty(output_shape, dtype=torch.float16, device="cuda")

    context.set_tensor_address(engine.get_tensor_name(0), input_tensor.data_ptr())
    context.set_tensor_address(engine.get_tensor_name(1), output_tensor.data_ptr())

    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    features = output_tensor.squeeze(0).float().cpu().tolist()
    return InferResponse(features=features)


# ── Batch inference — N slices in one TRT forward pass ───────────────────────

class BatchInferRequest(BaseModel):
    slices: List[List[float]]
    height: int
    width: int


class BatchInferResponse(BaseModel):
    features: List[List[float]]
    batch_size: int


@app.post("/batch_infer", response_model=BatchInferResponse)
def batch_infer(req: BatchInferRequest):
    """
    True batch inference — processes N slices in a single TRT forward pass.
    More efficient than N separate /infer calls.
    """
    n = len(req.slices)

    # Stack all slices into batch tensor [N, 1, H, W] as FP16
    arrays = [
        np.array(s, dtype=np.float16).reshape(req.height, req.width)
        for s in req.slices
    ]
    batch = np.stack(arrays, axis=0)              # [N, H, W]
    input_tensor = torch.from_numpy(batch).unsqueeze(1).cuda()  # [N, 1, H, W]

    # Get output shape and allocate output tensor
    engine = context.engine
    output_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
    embed_dim = output_shape[1]
    output_tensor = torch.empty((n, embed_dim), dtype=torch.float16, device="cuda")

    # Set dynamic batch size and tensor addresses
    context.set_input_shape(engine.get_tensor_name(0), (n, 1, req.height, req.width))
    context.set_tensor_address(engine.get_tensor_name(0), input_tensor.data_ptr())
    context.set_tensor_address(engine.get_tensor_name(1), output_tensor.data_ptr())

    # Run inference
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    features = output_tensor.float().cpu().tolist()
    return BatchInferResponse(features=features, batch_size=n)
