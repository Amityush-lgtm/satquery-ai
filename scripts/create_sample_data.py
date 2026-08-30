import os
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

SAMPLES_DIR = Path("data") / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_geotiffs():
    print("Generating sample remote sensing datasets in data/samples/ ...")

    # 1. Sample Multi-band GeoTIFF: Agricultural Patch
    width, height = 256, 256
    bands = 3  # RGB composite
    
    # Synthetic reflectance simulation (12-bit / 16-bit typical S2 values: 500 - 4000)
    # Band 1 (Red): higher in soil/urban, lower in dense vegetation
    # Band 2 (Green): moderate in vegetation
    # Band 3 (Blue): lower
    np.random.seed(42)
    b_red = np.random.randint(600, 1800, (height, width), dtype=np.uint16)
    b_green = np.random.randint(800, 2400, (height, width), dtype=np.uint16)
    b_blue = np.random.randint(400, 1200, (height, width), dtype=np.uint16)
    
    # Add synthetic agricultural field patterns
    b_green[40:180, 50:200] += 1200  # Lush green parcel
    b_red[190:240, 20:120] += 1500   # Soil / fallow parcel

    data_agri = np.stack([b_red, b_green, b_blue], axis=0)
    agri_path = SAMPLES_DIR / "sample_agricultural.tif"

    if HAS_RASTERIO:
        transform = from_origin(-0.5510, 47.4520, 10.0, 10.0)
        with rasterio.open(
            agri_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=bands,
            dtype=data_agri.dtype,
            crs="EPSG:32630",
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(data_agri)
        print(f"Created GeoTIFF: {agri_path} (EPSG:32630)")
    else:
        # 8-bit fallback
        norm_img = ((data_agri / 4000.0) * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(np.transpose(norm_img, (1, 2, 0)))
        img.save(agri_path)
        print(f"Created TIFF: {agri_path}")

    # Copy as standard sample_patch.tif
    sample_patch_path = SAMPLES_DIR / "sample_patch.tif"
    if agri_path.exists():
        import shutil
        shutil.copyfile(agri_path, sample_patch_path)
        print(f"Created Primary Sample: {sample_patch_path}")

    # 2. Sample PNG: Urban patch
    urban_arr = np.random.randint(100, 220, (256, 256, 3), dtype=np.uint8)
    # Add road grid lines
    urban_arr[120:136, :, :] = 50
    urban_arr[:, 120:136, :] = 50
    urban_img = Image.fromarray(urban_arr)
    urban_path = SAMPLES_DIR / "sample_urban.png"
    urban_img.save(urban_path)
    print(f"Created Sample PNG: {urban_path}")


if __name__ == "__main__":
    create_sample_geotiffs()
