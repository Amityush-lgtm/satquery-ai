from pathlib import Path
import pytest
from PIL import Image

from satquery.geo.bhoonidhi import BhoonidhiConnector, BhoonidhiProduct

def test_bhoonidhi_connector_list(tmp_path):
    conn = BhoonidhiConnector(data_root=tmp_path)
    products = conn.list_products()
    assert len(products) >= 4
    
    # Check satellite keys
    satellites = [p["satellite"] for p in products]
    assert any("Cartosat" in s for s in satellites)
    assert any("Resourcesat" in s for s in satellites)
    assert any("RISAT" in s for s in satellites)

def test_bhoonidhi_connector_filter(tmp_path):
    conn = BhoonidhiConnector(data_root=tmp_path)
    risat = conn.list_products(satellite="RISAT")
    assert len(risat) == 1
    assert "RISAT" in risat[0]["satellite"]

def test_bhoonidhi_load_scene(tmp_path):
    conn = BhoonidhiConnector(data_root=tmp_path)
    geo_img = conn.load_scene("ISRO_CARTOSAT2S_BLR_20240215")
    assert geo_img is not None
    assert isinstance(geo_img.pil_image, Image.Image)
    assert geo_img.pil_image.size == (512, 512)
