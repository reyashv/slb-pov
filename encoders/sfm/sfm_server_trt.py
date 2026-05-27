"""
sfm_server_trt.py — PyTriton TRT inference server for SFM

Port: 8000 (HTTP), 8001 (gRPC)
Health: GET http://host:8000/v2/health/ready
Infer:  POST http://host:8000/v2/models/sfm_large/infer
"""

import os
import numpy as np
import torch
from truefoundry.ml import get_client
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig


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


def make_infer_fn(engine):
    @batch
    def infer_fn(**inputs: np.ndarray):
        # @batch decorator handles batching — inputs is dict of numpy arrays
        # INPUT shape: [N, 1, H, W]
        input_batch = inputs["INPUT"]
        n = input_batch.shape[0]

        input_tensor = torch.from_numpy(input_batch.astype(np.float16)).cuda()

        context = engine.create_execution_context()
        embed_dim = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))[1]
        output_tensor = torch.empty((n, embed_dim), dtype=torch.float16, device="cuda")

        context.set_input_shape(engine.get_tensor_name(0), (n, 1, input_batch.shape[2], input_batch.shape[3]))
        context.set_tensor_address(engine.get_tensor_name(0), input_tensor.data_ptr())
        context.set_tensor_address(engine.get_tensor_name(1), output_tensor.data_ptr())

        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        features = output_tensor.float().cpu().numpy()  # [N, embed_dim]
        return [features]

    return infer_fn


def main():
    img_size = int(os.environ.get("SFM_IMG_SIZE", "224"))
    embed_dim = 1024  # vit_large embed dim

    engine = load_trt_engine()
    infer_fn = make_infer_fn(engine)

    print("Starting PyTriton server...")
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
