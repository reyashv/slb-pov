import os
import asyncio
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Decoder internal cluster URLs — set as env vars in yaml
DECODER_CLASSIFY_URL = os.environ.get("DECODER_CLASSIFY_URL", "")
DECODER_DETECT_URL = os.environ.get("DECODER_DETECT_URL", "")
DECODER_SEGMENT_URL = os.environ.get("DECODER_SEGMENT_URL", "")

class InferRequest(BaseModel):
    frames: List[List[float]]
    height: int
    width: int
    channels: int = 3

class InferResponse(BaseModel):
    classify: Optional[dict] = None
    detect: Optional[dict] = None
    segment: Optional[dict] = None

app = FastAPI(root_path=os.getenv("TFY_SERVICE_ROOT_PATH", ""))

@app.get("/health")
def health():
    return {"healthy": True}

@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    payload = req.dict()

    async def call(url, endpoint):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(f"{url}/{endpoint}", json=payload)
                return endpoint, resp.json()
        except Exception as e:
            return endpoint, {"error": str(e)}

    results = await asyncio.gather(
        call(DECODER_CLASSIFY_URL, "classify"),
        call(DECODER_DETECT_URL, "detect"),
        call(DECODER_SEGMENT_URL, "segment"),
    )

    return InferResponse(**{name: result for name, result in results})
