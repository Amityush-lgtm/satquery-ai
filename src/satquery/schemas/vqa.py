from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentTaskType(str, Enum):
    VQA = "vqa"
    GROUNDING = "grounding"
    BITEMPORAL_CHANGE = "bitemporal_change"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"


class BoundingBox(BaseModel):
    """Normalized spatial bounding box coordinates [0.0 - 1.0] for visual grounding."""
    ymin: float = Field(..., description="Top coordinate (0.0 to 1.0)")
    xmin: float = Field(..., description="Left coordinate (0.0 to 1.0)")
    ymax: float = Field(..., description="Bottom coordinate (0.0 to 1.0)")
    xmax: float = Field(..., description="Right coordinate (0.0 to 1.0)")
    label: Optional[str] = Field("target", description="Class or description of grounded entity")
    confidence: Optional[float] = Field(None, description="Confidence score for this detection")


class GeoMetadata(BaseModel):
    """Geospatial metadata extracted from input image."""
    crs: Optional[str] = Field(None, description="Coordinate Reference System e.g. EPSG:32633")
    transform: Optional[list] = Field(None, description="Affine Geotransform matrix")
    bounds: Optional[Dict[str, float]] = Field(None, description="Bounding coordinates (left, bottom, right, top)")
    shape: Optional[list] = Field(None, description="Image dimensions [bands, height, width]")
    count: Optional[int] = Field(None, description="Number of raster bands")
    driver: Optional[str] = Field(None, description="Image driver format (GTiff, PNG, JPEG)")
    dtypes: Optional[list] = Field(None, description="Data type of each band")
    nodata: Optional[Any] = Field(None, description="NoData pixel value")
    is_geospatial: bool = Field(False, description="True if georeferenced metadata is present")


class VQARequest(BaseModel):
    """Input payload for Visual Question Answering."""
    question: str = Field(..., description="Natural language question about the satellite image")
    image_path: Optional[str] = Field(None, description="Local path to satellite image file")


class VQAResponse(BaseModel):
    """Output response from VQA inference."""
    answer: str = Field(..., description="Model generated natural language answer")
    model: str = Field(..., description="Name and version of the vision-language model used")
    confidence: Optional[float] = Field(None, description="Calibrated confidence score if supported by model, null otherwise")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Image and execution metadata")
    execution_time_sec: Optional[float] = Field(None, description="Total inference time in seconds")
    boxes: Optional[List[BoundingBox]] = Field(default=None, description="Visual grounding bounding boxes if applicable")
    task_type: Optional[str] = Field("vqa", description="Identified remote sensing task type")


class AgentResponse(BaseModel):
    """Unified response from the Agentic Orchestrator."""
    answer: str = Field(..., description="Evidence-grounded natural language explanation")
    task: str = Field(..., description="Selected specialist task (vqa, grounding, bitemporal_change, optical_sar_fusion)")
    model: str = Field(..., description="Specialist model or fusion engine executed")
    tool_used: str = Field(..., description="Tool name invoked by the agent")
    confidence: Optional[float] = Field(None, description="Estimated confidence score")
    boxes: Optional[List[BoundingBox]] = Field(default=None, description="Visual grounding bounding boxes")
    change_map_url: Optional[str] = Field(default=None, description="Base64 PNG or URL of change heatmap overlay")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Primary and secondary image metadata")
    execution_time_sec: float = Field(..., description="Total execution latency in seconds")
    execution_trace: Dict[str, Any] = Field(default_factory=dict, description="Observable step-by-step agentic execution trace")


class ExecutionRecord(BaseModel):
    """Provenance trace schema for recording execution logs."""
    task: str
    model: str
    input: str
    question: str
    output: str
    confidence: Optional[float] = None
    execution_time_sec: float
    timestamp: str
    metadata: Dict[str, Any]
    boxes: Optional[List[Dict[str, Any]]] = None
