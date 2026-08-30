import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from satquery.geo.image_loader import GeoImage, ImageValidationError, load_image
from satquery.schemas.vqa import VQAResponse
from satquery.utils.logging import get_logger, record_execution
from satquery.vqa.model import BaseVQAModel, get_vqa_model

logger = get_logger("satquery.vqa.inference")


def answer_question(
    image_path: Union[str, Path],
    question: str,
    model: Optional[BaseVQAModel] = None,
    model_id: Optional[str] = None,
    force_mock: bool = False,
) -> Dict[str, Any]:
    """
    Core VQA API for SatQuery AI.
    
    Processes a satellite image and natural language question, executes
    the Remote-Sensing VLM, records provenance, and returns structured results.

    Example:
        >>> result = answer_question(
        ...     image_path="data/samples/example.tif",
        ...     question="What is visible in this image?"
        ... )
        >>> print(result["answer"])
    """
    start_time = time.time()
    
    if not question or not str(question).strip():
        raise ValueError("Question cannot be empty.")

    clean_question = str(question).strip()
    logger.info(f"Received VQA query: '{clean_question}' on image: {image_path}")

    # 1. Load and inspect image
    geo_img: GeoImage = load_image(image_path)
    logger.info(
        f"Image loaded: shape={geo_img.shape}, geospatial={geo_img.is_geospatial}, CRS={geo_img.crs}"
    )

    # 2. Acquire model instance
    vlm_model = model or get_vqa_model(model_id=model_id, force_mock=force_mock)

    # 3. Generate answer
    logger.info(f"Running VLM inference using {vlm_model.model_id}...")
    answer_text, confidence = vlm_model.generate_answer(
        image=geo_img.pil_image,
        question=clean_question,
    )

    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.3f}s. Model: {vlm_model.model_id}")

    # 4. Record provenance / execution trace
    record_execution(
        task="vqa",
        model_name=vlm_model.model_id,
        image_path=str(image_path),
        question=clean_question,
        answer=answer_text,
        execution_time_sec=inference_time,
        metadata=geo_img.metadata,
        confidence=confidence,
    )

    # 5. Build structured response
    response = VQAResponse(
        answer=answer_text,
        model=vlm_model.model_id,
        confidence=confidence,
        metadata=geo_img.metadata,
        execution_time_sec=round(inference_time, 4),
    )

    return response.model_dump()
