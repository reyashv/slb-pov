import os
import torch
import torch.nn as nn
from pydantic import BaseModel
from typing import List


class ClassifyRequest(BaseModel):
    frames: List[List[float]]
    height: int
    width: int
    channels: int = 3


class ClassifyResponse(BaseModel):
    class_id: int
    confidence: float
    logits: List[float]


class BatchClassifyRequest(BaseModel):
    videos: List[ClassifyRequest]


class BatchClassifyResponse(BaseModel):
    results: List[ClassifyResponse]


class ClassifyHead:
    def __init__(self, device: str):
        embed_dim = int(os.environ.get("EMBED_DIM", "1536"))
        num_classes = int(os.environ.get("NUM_CLASSES", "10"))
        self.head = nn.Linear(embed_dim, num_classes).to(device)
        self.device = device
        print(f"ClassifyHead — {embed_dim}d → {num_classes} classes")

    def load_weights(self, model_dir: str):
        probe_path = os.path.join(model_dir, "probe.pth")
        if os.path.exists(probe_path):
            self.head.load_state_dict(torch.load(probe_path, map_location=self.device))
            print(f"Loaded classify weights from {probe_path}")
        self.head.eval()

    def infer(self, embeddings: torch.Tensor) -> ClassifyResponse:
        with torch.no_grad():
            logits = self.head(embeddings)
            probs = torch.softmax(logits, dim=-1)
        return ClassifyResponse(
            class_id=int(logits.argmax(-1).item()),
            confidence=float(probs.max().item()),
            logits=logits.squeeze(0).cpu().tolist()
        )
