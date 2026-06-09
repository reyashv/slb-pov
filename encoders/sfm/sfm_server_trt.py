"""
sfm_server_trt.py — PyTriton TRT inference server for SFM

Endpoints:
  PyTriton (port 8000):
    Health: GET  http://host:8000/v2/health/ready
    Infer:  POST http://host:8000/v2/models/sfm_large/infer

  FastAPI (port 8080):
    Health:     GET  http://host:8080/health
    Full image: POST http://host:8080/infer_full_image
"""
import os
import math
import time
import threading
import numpy as np
import torch
from truefoundry.ml import get_client
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig


# ── TRT engine loading ────────────────────────────────────────────────────────

def load_trt_engine():
    import tensorrt as trt
    engine_artifact_fqn = os.environ["TRT_ENGINE_FQN"]
    img_size = int(os.environ.get("SFM_IMG_SIZE", "224"))
    print(f"Downloading TRT engine from {engine_artifact_fqn}...")
    client = get_client()
    os.makedirs("/tmp/trt-engine", exist_ok=True)
    av = client.get_artifact_version_by_fqn(fqn=engine_artifact_fqn)
    av.download(path="/tmp/trt-engine", overwrite=True)
    trt_path = "/tmp/trt-engine/sfm.trt"
    if not os.path.exists(trt_path):
        raise FileNotFoundError("sfm.trt not found after download")
    print("Loading TensorRT engine...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    print(f"TRT engine loaded — img_size={img_size}")
    return engine


# ── TRT inference helper (shared by PyTriton and FastAPI) ─────────────────────

def run_trt_inference(engine, input_batch_np):
    """
    Run TRT inference on a batch of tiles.
    input_batch_np: numpy array [N, 1, H, W] float32
    Returns: numpy array [N, embed_dim] float32
    """
    n = input_batch_np.shape[0]
    input_tensor = torch.from_numpy(input_batch_np.astype(np.float16)).cuda()
    context = engine.create_execution_context()
    embed_dim = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))[1]
    output_tensor = torch.empty((n, embed_dim), dtype=torch.float16, device="cuda")
    context.set_input_shape(
        engine.get_tensor_name(0),
        (n, 1, input_batch_np.shape[2], input_batch_np.shape[3])
    )
    context.set_tensor_address(engine.get_tensor_name(0), input_tensor.data_ptr())
    context.set_tensor_address(engine.get_tensor_name(1), output_tensor.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return output_tensor.float().cpu().numpy()


# ── PyTriton infer function (single tile / small batch) ───────────────────────

def make_infer_fn(engine):
    @batch
    def infer_fn(**inputs: np.ndarray):
        input_batch = inputs["INPUT"]
        features = run_trt_inference(engine, input_batch)
        return [features]
    return infer_fn


# ── FastAPI server for full image endpoint ────────────────────────────────────

def start_fastapi(engine, img_size, gpu_batch_size):
    from fastapi import FastAPI, Request, Response
    from pydantic import BaseModel
    from typing import List, Optional
    import uvicorn

    app = FastAPI(title="SFM Full Image Inference")

    class FullImageRequest(BaseModel):
        data_b64: str  # base64 encoded float32 array
        height: int
        width: int
        tile_size: Optional[int] = None

    class FullImageResponse(BaseModel):
        features_b64: str  # base64 encoded float32 result
        n_tiles: int
        tile_size: int
        tile_grid: List[int]
        embed_dim: int
        latency_ms: float
        gpu_time_ms: float

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/infer_full_image", response_model=FullImageResponse)
    def infer_full_image(req: FullImageRequest):
        import base64
        t_start = time.perf_counter()
        ts = req.tile_size or img_size

        # Decode base64 → numpy float32 array
        raw = base64.b64decode(req.data_b64)
        image = np.frombuffer(raw, dtype=np.float32).reshape(req.height, req.width)

        # Tile the image into ts x ts patches with zero-padding at edges
        n_rows = math.ceil(req.height / ts)
        n_cols = math.ceil(req.width / ts)
        tiles = []
        for r in range(n_rows):
            for c in range(n_cols):
                tile = np.zeros((ts, ts), dtype=np.float32)
                h_start = r * ts
                w_start = c * ts
                h_end = min(h_start + ts, req.height)
                w_end = min(w_start + ts, req.width)
                tile[:h_end - h_start, :w_end - w_start] = image[h_start:h_end, w_start:w_end]
                tiles.append(tile)

        # Stack into [N, 1, ts, ts] for TRT
        tiles_np = np.stack(tiles)[:, np.newaxis, :, :]  # [N, 1, ts, ts]

        # Batch inference through TRT engine
        t_gpu_start = time.perf_counter()
        all_features = []
        for i in range(0, len(tiles), gpu_batch_size):
            batch_input = tiles_np[i:i + gpu_batch_size]
            features = run_trt_inference(engine, batch_input)
            all_features.append(features)

        all_features = np.concatenate(all_features, axis=0)  # [N, embed_dim]
        t_gpu_end = time.perf_counter()

        t_end = time.perf_counter()

        # Encode features as base64
        features_bytes = all_features.astype(np.float32).tobytes()
        features_b64 = base64.b64encode(features_bytes).decode("ascii")

        return FullImageResponse(
            features_b64=features_b64,
            n_tiles=len(tiles),
            tile_size=ts,
            tile_grid=[n_rows, n_cols],
            embed_dim=all_features.shape[1],
            latency_ms=round((t_end - t_start) * 1000, 1),
            gpu_time_ms=round((t_gpu_end - t_gpu_start) * 1000, 1),
        )

    # ── SLB API Contract endpoint ─────────────────────────────
    class ROI(BaseModel):
        inline: Optional[List[int]] = None
        xline: Optional[List[int]] = None
        z: Optional[List[int]] = None

    class InputSpec(BaseModel):
        uri: Optional[str] = None          # S3/SDMS reference (future)
        data_b64: Optional[str] = None     # inline base64 encoded data
        height: Optional[int] = None       # required for inline
        width: Optional[int] = None        # required for inline
        roi: Optional[ROI] = None

    class SLBRequest(BaseModel):
        model: str = "seismic-fm"
        version: str = "1.0.0"
        input: InputSpec
        task: str = "embedding"
        output_format: str = "json"

    class SLBMetadata(BaseModel):
        tiles: int
        model_version: str
        tile_size: int
        embed_dim: int
        grid: List[int]
        gpu_time_ms: float

    class SLBResponse(BaseModel):
        status: str
        latency_ms: float
        output_uri: Optional[str] = None
        output_b64: Optional[str] = None
        metadata: SLBMetadata

    @app.post("/v1/infer", response_model=SLBResponse)
    def slb_infer(req: SLBRequest):
        import base64
        t_start = time.perf_counter()
        ts = img_size

        # Handle input source
        if req.input.uri:
            # S3/SDMS reference — not yet implemented
            if req.input.uri.startswith("s3://"):
                return SLBResponse(
                    status="error",
                    latency_ms=0,
                    metadata=SLBMetadata(tiles=0, model_version=req.version,
                                        tile_size=ts, embed_dim=0, grid=[0,0], gpu_time_ms=0),
                )
            return SLBResponse(
                status="error",
                latency_ms=0,
                metadata=SLBMetadata(tiles=0, model_version=req.version,
                                    tile_size=ts, embed_dim=0, grid=[0,0], gpu_time_ms=0),
            )
        elif req.input.data_b64 and req.input.height and req.input.width:
            # Inline base64 payload
            raw = base64.b64decode(req.input.data_b64)
            image = np.frombuffer(raw, dtype=np.float32).reshape(
                req.input.height, req.input.width
            )
        else:
            return SLBResponse(
                status="error",
                latency_ms=0,
                metadata=SLBMetadata(tiles=0, model_version=req.version,
                                    tile_size=ts, embed_dim=0, grid=[0,0], gpu_time_ms=0),
            )

        # Apply ROI crop if specified
        if req.input.roi:
            h_start = req.input.roi.inline[0] if req.input.roi.inline else 0
            h_end = req.input.roi.inline[1] if req.input.roi.inline else image.shape[0]
            w_start = req.input.roi.xline[0] if req.input.roi.xline else 0
            w_end = req.input.roi.xline[1] if req.input.roi.xline else image.shape[1]
            image = image[h_start:h_end, w_start:w_end]

        # Tile
        n_rows = math.ceil(image.shape[0] / ts)
        n_cols = math.ceil(image.shape[1] / ts)
        tiles = []
        for r in range(n_rows):
            for c in range(n_cols):
                tile = np.zeros((ts, ts), dtype=np.float32)
                h_s, w_s = r * ts, c * ts
                h_e = min(h_s + ts, image.shape[0])
                w_e = min(w_s + ts, image.shape[1])
                tile[:h_e - h_s, :w_e - w_s] = image[h_s:h_e, w_s:w_e]
                tiles.append(tile)
        tiles_np = np.stack(tiles)[:, np.newaxis, :, :]

        # GPU inference
        t_gpu = time.perf_counter()
        all_features = []
        for i in range(0, len(tiles), gpu_batch_size):
            batch_input = tiles_np[i:i + gpu_batch_size]
            features = run_trt_inference(engine, batch_input)
            all_features.append(features)
        all_features = np.concatenate(all_features, axis=0)
        t_gpu_end = time.perf_counter()

        # Encode output
        features_b64 = base64.b64encode(
            all_features.astype(np.float32).tobytes()
        ).decode("ascii")

        t_end = time.perf_counter()

        return SLBResponse(
            status="succeeded",
            latency_ms=round((t_end - t_start) * 1000, 1),
            output_b64=features_b64,
            metadata=SLBMetadata(
                tiles=len(tiles),
                model_version=req.version,
                tile_size=ts,
                embed_dim=all_features.shape[1],
                grid=[n_rows, n_cols],
                gpu_time_ms=round((t_gpu_end - t_gpu) * 1000, 1),
            ),
        )

    # ── Protobuf endpoint ─────────────────────────────────────
    @app.post("/infer_full_image_pb")
    async def infer_full_image_pb(request: Request):
        """
        Accepts protobuf-encoded InferFullImageRequest.
        Returns protobuf-encoded InferFullImageResponse.
        Content-Type: application/x-protobuf
        """
        from proto import sfm_inference_pb2

        raw = await request.body()
        req = sfm_inference_pb2.InferFullImageRequest()
        req.ParseFromString(raw)

        t_start = time.perf_counter()
        ts = req.tile_size if req.tile_size > 0 else img_size

        # Decode raw bytes → numpy
        image = np.frombuffer(req.data, dtype=np.float32).reshape(req.height, req.width)

        # Tile
        n_rows = math.ceil(req.height / ts)
        n_cols = math.ceil(req.width / ts)
        tiles = []
        for r in range(n_rows):
            for c in range(n_cols):
                tile = np.zeros((ts, ts), dtype=np.float32)
                h_start, w_start = r * ts, c * ts
                h_end = min(h_start + ts, req.height)
                w_end = min(w_start + ts, req.width)
                tile[:h_end - h_start, :w_end - w_start] = image[h_start:h_end, w_start:w_end]
                tiles.append(tile)

        tiles_np = np.stack(tiles)[:, np.newaxis, :, :]

        # GPU inference
        t_gpu = time.perf_counter()
        all_features = []
        for i in range(0, len(tiles), gpu_batch_size):
            batch_input = tiles_np[i:i + gpu_batch_size]
            features = run_trt_inference(engine, batch_input)
            all_features.append(features)
        all_features = np.concatenate(all_features, axis=0)
        t_gpu_end = time.perf_counter()
        t_end = time.perf_counter()

        # Build protobuf response
        resp = sfm_inference_pb2.InferFullImageResponse()
        resp.features = all_features.astype(np.float32).tobytes()
        resp.n_tiles = len(tiles)
        resp.tile_size = ts
        resp.embed_dim = all_features.shape[1]
        resp.grid_rows = n_rows
        resp.grid_cols = n_cols
        resp.latency_ms = round((t_end - t_start) * 1000, 1)
        resp.gpu_time_ms = round((t_gpu_end - t_gpu) * 1000, 1)

        return Response(
            content=resp.SerializeToString(),
            media_type="application/x-protobuf",
        )

    print(f"Starting FastAPI server on port 8080 (full image endpoint)...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    img_size = int(os.environ.get("SFM_IMG_SIZE", "224"))
    gpu_batch_size = int(os.environ.get("GPU_BATCH_SIZE", "16"))
    embed_dim = 1024  # vit_large embed dim

    engine = load_trt_engine()
    infer_fn = make_infer_fn(engine)

    # Start FastAPI in background thread for full image endpoint
    fastapi_thread = threading.Thread(
        target=start_fastapi,
        args=(engine, img_size, gpu_batch_size),
        daemon=True,
    )
    fastapi_thread.start()

    # Start PyTriton for single tile inference (main thread)
    print("Starting PyTriton server on port 8000...")
    with Triton(config=TritonConfig(http_port=8000, grpc_port=8001)) as triton:
        triton.bind(
            model_name="sfm_large",
            infer_func=infer_fn,
            inputs=[
                Tensor(name="INPUT", dtype=np.float32, shape=(1, img_size, img_size)),
            ],
            outputs=[
                Tensor(name="OUTPUT", dtype=np.float32, shape=(embed_dim,)),
            ],
            config=ModelConfig(
                max_batch_size=32,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=5000,
                ),
            ),
        )
        triton.serve()


if __name__ == "__main__":
    main()
