import numpy as np
import pytest
from PIL import Image

from satquery.analysis.grounding import detect_grounding_regions, parse_vlm_bounding_boxes
from satquery.schemas.vqa import BoundingBox


def test_parse_vlm_bounding_boxes():
    text = "Here is the region [0.15, 0.25, 0.65, 0.85] found in the scene."
    boxes = parse_vlm_bounding_boxes(text)
    assert len(boxes) == 1
    assert boxes[0].ymin == 0.15
    assert boxes[0].xmin == 0.25
    assert boxes[0].ymax == 0.65
    assert boxes[0].xmax == 0.85


def test_detect_grounding_water(sample_png_image):
    img = Image.open(sample_png_image)
    answer, boxes = detect_grounding_regions(img, "Locate the water body")
    assert len(boxes) > 0
    assert isinstance(boxes[0], BoundingBox)
    assert "water" in answer.lower() or "region" in answer.lower()


def test_detect_grounding_urban(sample_png_image):
    img = Image.open(sample_png_image)
    answer, boxes = detect_grounding_regions(img, "Find the built-up infrastructure and buildings")
    assert len(boxes) >= 1
    assert "built-up" in answer.lower() or "target" in answer.lower()
