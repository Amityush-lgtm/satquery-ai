import os
os.environ["SATQUERY_MOCK_MODEL"] = "1"
import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_png_image(temp_dir):
    """Creates a standard 100x100 RGB PNG."""
    img_path = temp_dir / "sample.png"
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_geotiff_image(temp_dir):
    """Creates a 3-band georeferenced GeoTIFF with CRS and Transform."""
    img_path = temp_dir / "sample_geo.tif"
    width, height = 120, 120
    bands = 3
    # Synthetic multispectral data (16-bit)
    data = np.random.randint(200, 3500, (bands, height, width), dtype=np.uint16)

    if HAS_RASTERIO:
        transform = from_origin(13.4050, 52.5200, 10.0, 10.0)
        with rasterio.open(
            img_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=bands,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(data)
    else:
        # Fallback to PIL TIFF if rasterio is not available
        img = Image.fromarray((data[:3, :, :] / 16).astype(np.uint8).transpose(1, 2, 0))
        img.save(img_path)

    return img_path


@pytest.fixture
def corrupted_image_file(temp_dir):
    """Creates a corrupted image file with invalid bytes."""
    bad_path = temp_dir / "corrupted.tif"
    with open(bad_path, "wb") as f:
        f.write(b"NOT_A_VALID_TIFF_HEADER_1234567890")
    return bad_path


@pytest.fixture
def unsupported_format_file(temp_dir):
    """Creates an unsupported file extension."""
    txt_path = temp_dir / "notes.txt"
    with open(txt_path, "w") as f:
        f.write("This is a text file, not an image.")
    return txt_path
