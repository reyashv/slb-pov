from .classify import ClassifyHead, ClassifyRequest, ClassifyResponse
from .detect import DetectHead, DetectRequest, DetectResponse, BoundingBox
from .segment import SegmentHead, SegmentRequest, SegmentResponse

HEAD_REGISTRY = {
    "classify": {
        "head_cls": ClassifyHead,
        "request_cls": ClassifyRequest,
        "response_cls": ClassifyResponse,
        "route": "/classify",
        "batch_request_cls": None,  # defined in classify.py
        "batch_response_cls": None,
        "batch_route": "/batch_classify",
    },
    "detect": {
        "head_cls": DetectHead,
        "request_cls": DetectRequest,
        "response_cls": DetectResponse,
        "route": "/detect",
        "batch_request_cls": None,
        "batch_response_cls": None,
        "batch_route": "/batch_detect",
    },
    "segment": {
        "head_cls": SegmentHead,
        "request_cls": SegmentRequest,
        "response_cls": SegmentResponse,
        "route": "/segment",
        "batch_request_cls": None,
        "batch_response_cls": None,
        "batch_route": "/batch_segment",
    },
}

# Attach batch schemas after import to avoid circular refs
from .classify import BatchClassifyRequest, BatchClassifyResponse
from .detect import BatchDetectRequest, BatchDetectResponse
from .segment import BatchSegmentRequest, BatchSegmentResponse

HEAD_REGISTRY["classify"]["batch_request_cls"] = BatchClassifyRequest
HEAD_REGISTRY["classify"]["batch_response_cls"] = BatchClassifyResponse
HEAD_REGISTRY["detect"]["batch_request_cls"] = BatchDetectRequest
HEAD_REGISTRY["detect"]["batch_response_cls"] = BatchDetectResponse
HEAD_REGISTRY["segment"]["batch_request_cls"] = BatchSegmentRequest
HEAD_REGISTRY["segment"]["batch_response_cls"] = BatchSegmentResponse
