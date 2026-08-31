import pytest
from PIL import Image

from satquery.agent.router import AgenticOrchestrator
from satquery.schemas.vqa import AgentTaskType
from satquery.vqa.model import MockVQAModel


@pytest.fixture
def orchestrator():
    model = MockVQAModel(model_id="mock-vlm-v1")
    return AgenticOrchestrator(vqa_model=model)


def test_router_classify_grounding(orchestrator):
    task = orchestrator.classify_task("Highlight the water body")
    assert task == AgentTaskType.GROUNDING


def test_router_classify_bitemporal(orchestrator):
    task = orchestrator.classify_task("What changed between these dates?", has_secondary_image=True)
    assert task == AgentTaskType.BITEMPORAL_CHANGE


def test_router_classify_crossmodal(orchestrator):
    task = orchestrator.classify_task("Use optical and SAR to find buildings", has_secondary_image=True)
    assert task == AgentTaskType.OPTICAL_SAR_FUSION


def test_router_classify_vqa(orchestrator):
    task = orchestrator.classify_task("What land cover types are visible?")
    assert task == AgentTaskType.VQA


def test_router_execute_grounding(orchestrator, sample_png_image):
    img = Image.open(sample_png_image)
    res = orchestrator.execute("Locate the water body", primary_image=img)
    assert res.task == "grounding"
    assert res.boxes is not None
    assert len(res.boxes) > 0
    assert res.execution_trace["tool_used"] == "VisualGroundingTool"


def test_router_execute_bitemporal(orchestrator, sample_png_image):
    img1 = Image.open(sample_png_image)
    img2 = Image.open(sample_png_image)
    res = orchestrator.execute("What changed between dates?", primary_image=img1, secondary_image=img2)
    assert res.task == "bitemporal_change"
    assert res.change_map_url is not None
    assert res.execution_trace["tool_used"] == "BiTemporalChangeTool"
