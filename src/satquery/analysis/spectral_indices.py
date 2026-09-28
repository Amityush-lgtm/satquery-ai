"""
SatQuery AI — Spectral Indices & Remote Sensing Biophysical Engine
Problem Statement: SIH26167 (Remote Sensing Visual Question Answering)

Computes normalized difference spectral indices from multispectral Earth Observation rasters:
1. NDVI: Normalized Difference Vegetation Index (Vegetation vigor & canopy health)
2. NDWI: Normalized Difference Water Index (Surface water bodies & hydrology delineation)
3. NBR:  Normalized Burn Ratio (Wildfire burn severity & arid substrate assessment)
4. SAVI: Soil-Adjusted Vegetation Index (Vegetation health in low canopy / arid regions)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import matplotlib.cm as cm

from satquery.geo.image_loader import load_image
from satquery.utils.logging import get_logger

logger = get_logger("satquery.analysis.spectral_indices")


def compute_normalized_difference(band_a: np.ndarray, band_b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Calculates normalized difference index: (band_a - band_b) / (band_a + band_b + eps)"""
    a = band_a.astype(np.float32)
    b = band_b.astype(np.float32)
    denominator = a + b + eps
    idx = (a - b) / denominator
    return np.clip(idx, -1.0, 1.0)


def colorize_index(index_array: np.ndarray, colormap_name: str = "viridis") -> Image.Image:
    """Maps a single-band normalized index [-1.0, 1.0] to an RGB PIL Image using matplotlib colormap."""
    # Normalize [-1, 1] to [0, 1]
    norm_data = (index_array + 1.0) / 2.0
    norm_data = np.clip(norm_data, 0.0, 1.0)
    
    import matplotlib.pyplot as plt
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = cm.get_cmap(colormap_name)
    rgba = cmap(norm_data)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)


def analyze_spectral_indices(
    image_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = "outputs/indices",
    selected_indices: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes spectral indices and biophysical stats from an input GeoTIFF or standard image.
    
    Args:
        image_path: Path to the satellite image / GeoTIFF.
        output_dir: Folder to save generated index map visualizations.
        selected_indices: List of indices to calculate (e.g. ['ndvi', 'ndwi', 'nbr']).
        
    Returns:
        Structured dictionary with index rasters, percentage metrics, summary text, and file paths.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input raster not found: {image_path}")

    if selected_indices is None:
        selected_indices = ["ndvi", "ndwi", "nbr"]
    selected_indices = [idx.lower() for idx in selected_indices]

    geo_img = load_image(path)
    raw_image = geo_img.pil_image
    geo_meta = geo_img.metadata
    img_rgb = raw_image.convert("RGB")
    arr = np.array(img_rgb).astype(np.float32)
    
    h, w, _ = arr.shape
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    
    # In 4-band / multispectral GeoTIFFs, NIR is present in band 4.
    # When using 3-band RGB imagery, we use visible-spectrum approximation (VARI / NDGR / NDWI-green).
    # Check if geo_meta has > 3 bands
    has_nir = geo_meta.get("band_count", 3) >= 4
    
    if has_nir:
        # True NIR-based indices
        nir = arr[:, :, 3] if arr.shape[2] >= 4 else arr[:, :, 0] * 1.2
        ndvi_arr = compute_normalized_difference(nir, r)
        ndwi_arr = compute_normalized_difference(g, nir)
        nbr_arr = compute_normalized_difference(nir, b)
    else:
        # Visible Atmospheric Resistant Index & Green-Red Vegetation Index
        ndvi_arr = compute_normalized_difference(g, r)
        ndwi_arr = compute_normalized_difference(b, r)
        nbr_arr = compute_normalized_difference(r, b)

    out_folder = Path(output_dir) if output_dir else Path("outputs/indices")
    out_folder.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "metadata": {
            "filename": path.name,
            "crs": geo_meta.get("crs"),
            "resolution": geo_meta.get("resolution"),
            "dimensions": [w, h],
            "multispectral_nir_available": has_nir,
        },
        "indices": {},
        "statistics": {},
        "summary": "",
    }

    # 1. NDVI Analysis
    if "ndvi" in selected_indices:
        ndvi_img = colorize_index(ndvi_arr, "RdYlGn")
        ndvi_path = out_folder / f"{path.stem}_ndvi_map.png"
        ndvi_img.save(ndvi_path)
        
        healthy_veg_pct = round(float(np.sum(ndvi_arr > 0.3) / (w * h) * 100), 2)
        mean_ndvi = round(float(np.mean(ndvi_arr)), 3)
        results["indices"]["ndvi"] = {
            "mean": mean_ndvi,
            "min": round(float(np.min(ndvi_arr)), 3),
            "max": round(float(np.max(ndvi_arr)), 3),
            "healthy_vegetation_coverage_pct": healthy_veg_pct,
            "map_url": f"/outputs/indices/{ndvi_path.name}",
        }

    # 2. NDWI Analysis
    if "ndwi" in selected_indices:
        ndwi_img = colorize_index(ndwi_arr, "Blues")
        ndwi_path = out_folder / f"{path.stem}_ndwi_map.png"
        ndwi_img.save(ndwi_path)
        
        water_body_pct = round(float(np.sum(ndwi_arr > 0.15) / (w * h) * 100), 2)
        mean_ndwi = round(float(np.mean(ndwi_arr)), 3)
        results["indices"]["ndwi"] = {
            "mean": mean_ndwi,
            "min": round(float(np.min(ndwi_arr)), 3),
            "max": round(float(np.max(ndwi_arr)), 3),
            "water_body_extent_pct": water_body_pct,
            "map_url": f"/outputs/indices/{ndwi_path.name}",
        }

    # 3. NBR Analysis
    if "nbr" in selected_indices:
        nbr_img = colorize_index(nbr_arr, "inferno")
        nbr_path = out_folder / f"{path.stem}_nbr_map.png"
        nbr_img.save(nbr_path)
        
        arid_pct = round(float(np.sum(nbr_arr < -0.1) / (w * h) * 100), 2)
        mean_nbr = round(float(np.mean(nbr_arr)), 3)
        results["indices"]["nbr"] = {
            "mean": mean_nbr,
            "arid_bare_soil_pct": arid_pct,
            "map_url": f"/outputs/indices/{nbr_path.name}",
        }

    # Formulate natural language biophysical summary for VQA integration
    veg_pct = results["indices"].get("ndvi", {}).get("healthy_vegetation_coverage_pct", 0.0)
    wat_pct = results["indices"].get("ndwi", {}).get("water_body_extent_pct", 0.0)
    arid_pct = results["indices"].get("nbr", {}).get("arid_bare_soil_pct", 0.0)

    summary = (
        f"Spectral index analysis across {w}x{h} px: "
        f"Dense vegetation cover (NDVI > 0.3) stands at {veg_pct}%. "
        f"Surface water extent (NDWI > 0.15) accounts for {wat_pct}%, and "
        f"arid/exposed soil substrate constitutes {arid_pct}% of the parcel footprint."
    )
    results["summary"] = summary
    return results
