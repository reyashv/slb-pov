"""
convert_trt.py

Converts the SFM ViT model to a TensorRT FP16 engine.
Runs as a one-time TFY job. Saves the TRT engine to ML Repo.
"""

import os
import glob
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import timm.models.vision_transformer

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            return self.fc_norm(x)
        else:
            x = self.norm(x)
            return x[:, 0]

    def forward(self, x):
        return self.forward_features(x)


def vit_base_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_large_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

MODEL_REGISTRY = {
    "vit_base_patch16": vit_base_patch16,
    "vit_large_patch16": vit_large_patch16,
}


def main():
    import tensorrt as trt
    from truefoundry.ml import get_client, ArtifactPath

    model_fqn   = os.environ["MODEL_DIR"]
    arch        = os.environ.get("SFM_ARCH", "vit_base_patch16")
    img_size    = int(os.environ.get("SFM_IMG_SIZE", "224"))
    output_name = os.environ.get("OUTPUT_ARTIFACT_NAME", "sfm-base-trt-engine")

    client = get_client()
    run = client.create_run(ml_repo="slb-pov", run_name="sfm-trt-conversion")
    print(f"Started run: {run.run_id}")

    try:
        run.log_params({"model_fqn": model_fqn, "arch": arch,
                        "img_size": img_size, "precision": "fp16"})

        # Step 1: Download model
        print(f"Downloading model from {model_fqn}...")
        os.makedirs("/tmp/sfm-model", exist_ok=True)
        download_info = client.get_model_version_by_fqn(model_fqn).download(path="/tmp/sfm-model")
        model_dir = download_info.download_dir

        # Step 2: Load PyTorch model
        print(f"Loading {arch}...")
        m = MODEL_REGISTRY[arch](
            num_classes=0, global_pool=False, in_chans=1, img_size=img_size
        )
        pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth in {model_dir}")
        checkpoint = torch.load(pth_files[0], map_location="cuda")
        state_dict = checkpoint.get("model", checkpoint)
        m.load_state_dict(state_dict, strict=False)
        m = m.cuda().eval().half()
        print(f"Model loaded — arch={arch}, img_size={img_size}")

        # Step 3: Export to ONNX
        onnx_path = "/tmp/sfm.onnx"
        # Delete old files to force fresh build
        for f in [onnx_path, "/tmp/sfm.trt"]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Deleted old {f}")

        dummy_input = torch.randn(1, 1, img_size, img_size, dtype=torch.float16).cuda()
        print("Exporting to ONNX...")
        torch.onnx.export(
            m, dummy_input, onnx_path, opset_version=14,
            input_names=["input"], output_names=["features"],
            dynamic_axes={"input": {0: "batch_size"}, "features": {0: "batch_size"}},
            do_constant_folding=True,
        )
        print(f"ONNX exported to {onnx_path}")

        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified ✅")

        # Step 4: Build TensorRT engine — NO CACHE
        trt_path = "/tmp/sfm.trt"
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        print("Building TensorRT FP16 engine (this takes 5-15 minutes)...")
        builder = trt.Builder(TRT_LOGGER)
        # EXPLICIT_BATCH removed in TRT 10+ — explicit batch is default now
        try:
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        except AttributeError:
            flags = 0
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
        config.set_flag(trt.BuilderFlag.FP16)
        # ── DISABLE BUILDER CACHE to force fresh compilation on this GPU node ──
        config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

        profile = builder.create_optimization_profile()
        profile.set_shape("input",
            min=(1, 1, img_size, img_size),
            opt=(4, 1, img_size, img_size),
            max=(16, 1, img_size, img_size)
        )
        config.add_optimization_profile(profile)

        import time
        t0 = time.time()
        engine_bytes = builder.build_serialized_network(network, config)
        elapsed = time.time() - t0
        print(f"TRT build took {elapsed:.1f}s")

        if engine_bytes is None:
            raise RuntimeError("Failed to build TensorRT engine")

        with open(trt_path, "wb") as f:
            f.write(engine_bytes)
        print(f"TensorRT engine saved to {trt_path} ✅")

        # Step 5: Save to ML Repo
        print(f"Logging TRT engine to ML Repo as {output_name}...")
        av = run.log_artifact(
            name=output_name,
            artifact_paths=[ArtifactPath(src=trt_path, dest="sfm.trt")],
            metadata={"arch": arch, "img_size": img_size,
                      "precision": "fp16", "source_model_fqn": model_fqn}
        )
        print(f"Done — TRT engine logged as: {av.fqn}")
        run.log_metrics({"conversion_success": 1, "build_time_seconds": elapsed})

    except Exception as e:
        print(f"Conversion failed: {e}")
        run.log_metrics({"conversion_success": 0})
        raise
    finally:
        run.end()


if __name__ == "__main__":
    main()
