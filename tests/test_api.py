import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from satquery.api.main import app
from satquery.vqa.model import MockVQAModel


@pytest.fixture
def client():
    # Set mock model on app state for rapid deterministic tests
    app.state.model = MockVQAModel(model_id="mock-vlm-v1")
    with TestClient(app) as test_client:
        yield test_client


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "SatQuery AI"
    assert "active_model" in data


def test_api_vqa_valid_upload(client, sample_png_image):
    with open(sample_png_image, "rb") as f:
        response = client.post(
            "/vqa",
            data={"question": "What is visible in this satellite image?"},
            files={"image": ("test_patch.png", f, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["model"] == "mock-vlm-v1"
    assert data["confidence"] is None
    assert "metadata" in data
    assert data["metadata"]["original_filename"] == "test_patch.png"


def test_api_vqa_empty_question(client, sample_png_image):
    with open(sample_png_image, "rb") as f:
        response = client.post(
            "/vqa",
            data={"question": "   "},
            files={"image": ("test_patch.png", f, "image/png")},
        )

    assert response.status_code == 422


def test_api_vqa_unsupported_file(client, unsupported_format_file):
    with open(unsupported_format_file, "rb") as f:
        response = client.post(
            "/vqa",
            data={"question": "What is visible?"},
            files={"image": ("notes.txt", f, "text/plain")},
        )

    assert response.status_code in (400, 415)


def test_api_executions_endpoint(client):
    response = client.get("/executions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
