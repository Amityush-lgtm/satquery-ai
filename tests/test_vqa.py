import pytest
from satquery.vqa.inference import answer_question
from satquery.vqa.model import MockVQAModel, get_vqa_model, load_model_config


def test_model_config_loading():
    cfg = load_model_config()
    assert "selected_model" in cfg
    assert "candidates" in cfg
    assert len(cfg["candidates"]) >= 2


def test_mock_model_generation(sample_png_image):
    model = get_vqa_model(force_mock=True)
    assert isinstance(model, MockVQAModel)

    result = answer_question(
        image_path=sample_png_image,
        question="What is visible in this image?",
        model=model,
    )

    assert "answer" in result
    assert "model" in result
    assert result["model"] == "mock-vlm-v1"
    assert result["confidence"] is None  # Confidence remains null for uncalibrated prototype
    assert "metadata" in result
    assert result["metadata"]["filename"] == "sample.png"


def test_vqa_geotiff_inference(sample_geotiff_image):
    result = answer_question(
        image_path=sample_geotiff_image,
        question="Describe the land cover and water bodies.",
        force_mock=True,
    )

    assert result["answer"] is not None
    assert len(result["answer"]) > 0
    assert result["metadata"]["filename"] == "sample_geo.tif"


def test_vqa_empty_question_raises(sample_png_image):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question(
            image_path=sample_png_image,
            question="   ",
            force_mock=True,
        )
