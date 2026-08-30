from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


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
