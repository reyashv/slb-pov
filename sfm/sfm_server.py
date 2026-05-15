import os
import glob
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import timm.models.vision_transformer

# ── Inline VisionTransformer from MAE/SFM ──────────────────────────────────
# This is the exact models_vit.py from facebookresearch/mae — copied here
# so we have zero dependency on external repos or runtime git clones.

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
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
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# ── Model registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "vit_base_patch16": vit_base_patch16,
    "vit_large_patch16": vit_large_patch16,
}

# ── FastAPI app ─────────────────────────────────────────────────────────────
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_dir = os.environ["MODEL_DIR"]
    arch = os.environ.get("SFM_ARCH", "vit_base_patch16")

    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arch: {arch}. Choose from {list(MODEL_REGISTRY.keys())}")

    m = MODEL_REGISTRY[arch](num_classes=0, global_pool=False, in_chans=1)

    # Find the .pth file — handles any filename TrueFoundry downloads
    pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(
            f"No .pth file in {model_dir}. Contents: {os.listdir(model_dir)}"
        )

    checkpoint_path = pth_files[0]
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    state_dict = checkpoint.get("model", checkpoint)
    msg = m.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded: {msg}")

    m.eval().cuda()
    model = m
    print(f"SFM ready — arch={arch}")
    yield


app = FastAPI(
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""),
    lifespan=lifespan
)


class InferRequest(BaseModel):
    data: List[float]
    height: int
    width: int


class InferResponse(BaseModel):
    features: List[float]   # CLS token — 1D vector


@app.get("/health")
def health():
    return {"healthy": True}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    arr = np.array(req.data, dtype=np.float32).reshape(req.height, req.width)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        features = model.forward_features(tensor)
    return InferResponse(features=features.squeeze(0).cpu().tolist())
