from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from satquery.analysis.spectral_indices import analyze_spectral_indices, compute_normalized_difference, colorize_index

def test_compute_normalized_difference():
    band_a = np.array([[100, 50], [200, 0]])
    band_b = np.array([[50, 50], [0, 100]])
    res = compute_normalized_difference(band_a, band_b)
    assert res.shape == (2, 2)
    assert np.all(res >= -1.0) and np.all(res <= 1.0)
    assert pytest.approx(res[0, 0], 0.01) == (100 - 50) / 150

def test_colorize_index():
    arr = np.linspace(-1, 1, 100).reshape(10, 10)
    img = colorize_index(arr, "RdYlGn")
    assert isinstance(img, Image.Image)
    assert img.size == (10, 10)
    assert img.mode == "RGB"

def test_analyze_spectral_indices_png(sample_png_image, tmp_path):
    res = analyze_spectral_indices(sample_png_image, output_dir=tmp_path)
    assert "indices" in res
    assert "ndvi" in res["indices"]
    assert "ndwi" in res["indices"]
    assert "nbr" in res["indices"]
    assert "summary" in res
    assert isinstance(res["indices"]["ndvi"]["mean"], float)
    assert (tmp_path / f"{sample_png_image.stem}_ndvi_map.png").exists()

def test_analyze_spectral_indices_geotiff(sample_geotiff_image, tmp_path):
    res = analyze_spectral_indices(sample_geotiff_image, output_dir=tmp_path)
    assert "metadata" in res
    assert res["metadata"]["crs"] is not None
    assert "ndvi" in res["indices"]
    assert (tmp_path / f"{sample_geotiff_image.stem}_ndvi_map.png").exists()
