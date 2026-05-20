import os
import torch
import torch.nn as nn
from pydantic import BaseModel
from typing import List


class DetectRequest(BaseModel):
    frames: List[List[float]]
    height: int
    width: int
    channels: int = 3


class BoundingBox(BaseModel):
    x: float  # normalised 0-1
    y: float
    w: float
    h: float
    confidence: float


class DetectResponse(BaseModel):
    boxes: List[BoundingBox]


class BatchDetectRequest(BaseModel):
    videos: List[DetectRequest]


class BatchDetectResponse(BaseModel):
    results: List[DetectResponse]


class DetectHead:
    def __init__(self, device: str):
        embed_dim = int(os.environ.get("EMBED_DIM", "1536"))
        self.num_anchors = int(os.environ.get("NUM_ANCHORS", "10"))
        # Each anchor predicts [x, y, w, h, confidence] = 5 values
        self.head = nn.Linear(embed_dim, self.num_anchors * 5).to(device)
        self.device = device
        print(f"DetectHead — {embed_dim}d → {self.num_anchors} anchors")

    def load_weights(self, model_dir: str):
        ckpt_path = os.path.join(model_dir, "det_head.pth")
        if os.path.exists(ckpt_path):
            self.head.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            print(f"Loaded detect weights from {ckpt_path}")
        self.head.eval()

    def infer(self, embeddings: torch.Tensor) -> DetectResponse:
        with torch.no_grad():
            raw = self.head(embeddings).reshape(-1, self.num_anchors, 5)
            coords = torch.sigmoid(raw[..., :4])
            scores = torch.sigmoid(raw[..., 4])
        boxes = [
            BoundingBox(
                x=float(coords[0, i, 0]),
                y=float(coords[0, i, 1]),
                w=float(coords[0, i, 2]),
                h=float(coords[0, i, 3]),
                confidence=float(scores[0, i])
            )
            for i in range(self.num_anchors)
        ]
        return DetectResponse(boxes=boxes)
