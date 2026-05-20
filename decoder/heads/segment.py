import os
import torch
import torch.nn as nn
from pydantic import BaseModel
from typing import List


class SegmentRequest(BaseModel):
    frames: List[List[float]]
    height: int
    width: int
    channels: int = 3


class SegmentResponse(BaseModel):
    # 2D mask returned as flat list — reshape to out_height x out_width on client
    mask: List[float]
    out_height: int
    out_width: int


class BatchSegmentRequest(BaseModel):
    videos: List[SegmentRequest]


class BatchSegmentResponse(BaseModel):
    results: List[SegmentResponse]


class SegmentHead:
    def __init__(self, device: str):
        embed_dim = int(os.environ.get("EMBED_DIM", "1536"))
        self.out_h = int(os.environ.get("OUT_HEIGHT", "64"))
        self.out_w = int(os.environ.get("OUT_WIDTH", "64"))
        # Simple MLP segmentation head — replace with UPerNet for production
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.out_h * self.out_w)
        ).to(device)
        self.device = device
        print(f"SegmentHead — {embed_dim}d → {self.out_h}x{self.out_w} mask")

    def load_weights(self, model_dir: str):
        ckpt_path = os.path.join(model_dir, "seg_head.pth")
        if os.path.exists(ckpt_path):
            self.head.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            print(f"Loaded segment weights from {ckpt_path}")
        self.head.eval()

    def infer(self, embeddings: torch.Tensor) -> SegmentResponse:
        with torch.no_grad():
            mask_logits = self.head(embeddings)
            mask = torch.sigmoid(mask_logits).squeeze(0)
        return SegmentResponse(
            mask=mask.cpu().tolist(),
            out_height=self.out_h,
            out_width=self.out_w
        )
