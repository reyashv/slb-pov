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
DEVICE = "cuda"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, context

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

    print(f"TRT engine ready — img_size={img_size}, precision=FP16")
    yield
    del context
    del engine


app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)


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
    import tensorrt as trt

    # Prepare input
    arr = np.array(req.data, dtype=np.float16).reshape(1, 1, req.height, req.width)
    input_tensor = torch.from_numpy(arr).cuda()

    # Get output shape from engine
    engine = context.engine
    output_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
    output_shape = (1,) + output_shape[1:]  # set batch size to 1
    output_tensor = torch.empty(output_shape, dtype=torch.float16, device="cuda")

    # Set tensor addresses
    context.set_tensor_address(engine.get_tensor_name(0), input_tensor.data_ptr())
    context.set_tensor_address(engine.get_tensor_name(1), output_tensor.data_ptr())

    # Run inference
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    features = output_tensor.squeeze(0).float().cpu().tolist()
    return InferResponse(features=features)
