import base64
import io
import json
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from satquery.geo.image_loader import (
    CorruptedImageError,
    ImageValidationError,
    UnsupportedFormatError,
    load_image,
)
from satquery.schemas.vqa import VQAResponse
from satquery.utils.logging import PROVENANCE_FILE, get_logger, record_execution
from satquery.vqa.model import BaseVQAModel, get_vqa_model

logger = get_logger("satquery.api")


def pil_to_base64_png(img) -> str:
    """Encodes a PIL image to a Base64 PNG data URL."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading models once at application startup."""
    logger.info("Initializing SatQuery AI backend...")
    force_mock = os.getenv("SATQUERY_MOCK_MODEL", "0").lower() in {"1", "true", "yes"}
    try:
        # Pre-load the active VLM singleton
        app.state.model = get_vqa_model(force_mock=force_mock)
        logger.info(f"Loaded active model: {app.state.model.model_id}")
    except Exception as e:
        logger.error(f"Error initializing model on startup: {e}")
        # Fallback to mock for resilient startup
        app.state.model = get_vqa_model(force_mock=True)

    yield

    logger.info("Shutting down SatQuery AI backend...")


app = FastAPI(
    title="SatQuery AI API",
    description="Remote Sensing Visual Question Answering Backend for SIH26167",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_DIR = Path("outputs") / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing system status and active model."""
    model_id = getattr(app.state.model, "model_id", "uninitialized") if hasattr(app.state, "model") else "none"
    return {
        "status": "healthy",
        "service": "SatQuery AI",
        "version": "0.1.0",
        "active_model": model_id,
    }


@app.post("/preview")
async def preview_image_endpoint(
    image: UploadFile = File(..., description="Satellite image file to convert for browser preview"),
) -> Dict[str, Any]:
    """
    Accepts any supported satellite format (GeoTIFF, TIFF, PNG, JPG),
    normalizes it via the geospatial engine (percentile stretch & multi-band reduction),
    and returns a base64 encoded PNG for instant browser rendering alongside geospatial metadata.
    """
    suffix = Path(image.filename).suffix if image.filename else ".tif"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_UPLOAD_DIR)
    temp_path = Path(temp_file.name)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        try:
            geo_img = load_image(temp_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        except UnsupportedFormatError as e:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(e))
        except (CorruptedImageError, ImageValidationError) as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        geo_img.metadata["original_filename"] = image.filename
        preview_data_url = pil_to_base64_png(geo_img.pil_image)

        return {
            "status": "success",
            "filename": image.filename,
            "preview_url": preview_data_url,
            "metadata": geo_img.metadata,
        }
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.post("/vqa", response_model=VQAResponse)
async def vqa_endpoint(
    question: str = Form(..., description="Natural language question regarding the satellite image"),
    image: UploadFile = File(..., description="Satellite image file (GeoTIFF, TIFF, PNG, JPEG)"),
) -> VQAResponse:
    """
    Single-image Remote-Sensing Visual Question Answering.
    Accepts an uploaded satellite image and user query, processes it through the VLM,
    and returns answer, model identity, geospatial metadata, and latency.
    """
    start_time = time.time()
    clean_question = question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty or whitespace.",
        )

    # Save uploaded file safely to temporary disk
    suffix = Path(image.filename).suffix if image.filename else ".tif"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_UPLOAD_DIR)
    temp_path = Path(temp_file.name)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 1. Load and validate image & geospatial metadata
        try:
            geo_img = load_image(temp_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        except UnsupportedFormatError as e:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(e))
        except (CorruptedImageError, ImageValidationError) as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        # Update metadata filename with original upload name
        geo_img.metadata["original_filename"] = image.filename

        # 2. Acquire loaded model from app state
        model: BaseVQAModel = app.state.model

        # 3. Perform inference
        logger.info(f"API executing VQA query: '{clean_question}' on {image.filename} using {model.model_id}")
        answer_text, confidence = model.generate_answer(
            image=geo_img.pil_image,
            question=clean_question,
        )

        inference_time = time.time() - start_time

        # 4. Record provenance
        record_execution(
            task="vqa",
            model_name=model.model_id,
            image_path=image.filename or str(temp_path),
            question=clean_question,
            answer=answer_text,
            execution_time_sec=inference_time,
            metadata=geo_img.metadata,
            confidence=confidence,
        )

        return VQAResponse(
            answer=answer_text,
            model=model.model_id,
            confidence=confidence,
            metadata=geo_img.metadata,
            execution_time_sec=round(inference_time, 4),
        )

    finally:
        # Clean up temporary uploaded file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.get("/executions")
async def get_recent_executions(limit: int = 10) -> List[Dict[str, Any]]:
    """Returns the most recent execution provenance logs."""
    if not PROVENANCE_FILE.exists():
        return []

    records = []
    try:
        with open(PROVENANCE_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines[-limit:]):
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Error reading provenance records: {e}")

    return records


# Mount static files for lightweight Web UI
web_dir = Path("web")
if web_dir.exists():
    app.mount("/static", StaticFiles(directory="web"), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(web_dir / "index.html")
