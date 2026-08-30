# SatQuery AI — Prototype 1
### Smart India Hackathon 2026 · Problem Statement ID: SIH26167
**Team:** Saverra · **Institution:** IIT Madras BS Degree Programme  
**Theme:** Space Technology · **Category:** Software

---

## 🛰️ 1. Project Overview

**SatQuery AI** is an intelligent, agentic vision-language assistant for Earth Observation (EO) and remote sensing data. 

**Prototype 1** establishes the core baseline pipeline:
$$\text{Satellite Image (GeoTIFF, TIFF, PNG, JPEG)} + \text{Natural Language Query} \xrightarrow{\text{Remote-Sensing VLM}} \text{Evidence-Grounded Answer}$$

### Key Capabilities in Prototype 1:
1. **Geospatial Raster Ingestion:** Reads **GeoTIFF**, TIFF, PNG, and JPEG files with full extraction of Coordinate Reference System (`CRS`), Affine geotransform, bounding box, and radiometric bands.
2. **Dynamic 2%–98% Percentile Normalization:** Automatically converts 16-bit and multi-band satellite rasters into clean RGB composites suitable for Vision-Language Models.
3. **Live Browser GeoTIFF Preview:** Server-side `/preview` endpoint renders GeoTIFFs directly into the web browser while populating geospatial metadata drawers.
4. **VLM Integration (`Qwen2-VL-2B-Instruct`):** Integrated open-source remote-sensing vision-language model with dynamic resolution handling and fallback Mock VLM mode.
5. **Interactive Web UI & REST API:** Modern responsive interface with instant previews, sample presets, latency monitoring, and auditable execution logging (`outputs/executions.jsonl`).
6. **Automated Test Suite:** 16 unit and integration tests passing 100%.

---

## 💻 2. Team Setup Guide (Get It Running on Your System)

Follow these steps to set up and run SatQuery AI on your machine.

### Prerequisites:
- **Operating System:** Windows 10/11, macOS, or Ubuntu/Linux
- **Python:** Python 3.10 or 3.11 installed
- **Git:** Git CLI installed
- **Hardware:** Works on both CPU and GPU (NVIDIA GPU with CUDA recommended for real VLM weights; CPU/Mock mode supported for edge/lightweight testing).

---

### Step 1: Clone Repository
```powershell
git clone https://github.com/Amityush-lgtm/satquery-ai.git
cd satquery-ai
```

---

### Step 2: Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If you see an execution policy error on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On macOS / Linux (Bash/Zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

---

### Step 4: Generate Sample GeoTIFF & Image Datasets
Run the script to generate sample multi-band GeoTIFFs and test imagery in `data/samples/`:
```powershell
python scripts/create_sample_data.py
```
This generates:
- `data/samples/sample_patch.tif` (Multi-band GeoTIFF)
- `data/samples/sample_agricultural.tif` (GeoTIFF)
- `data/samples/sample_urban.png` (High-resolution optical PNG)

---

## 🚀 3. How to Run the Application

### Option A: Launch Interactive Web Application (Recommended)

#### Mode 1: Fast Testing / Mock Mode (Starts Instantly, No GPU or Download Needed)
If you do not have a dedicated GPU or want to test the Web UI without waiting for 4.5 GB model weights to download:
```powershell
# Windows PowerShell
$env:SATQUERY_MOCK_MODEL="1"
python scripts/run_server.py

# macOS / Linux Bash
SATQUERY_MOCK_MODEL=1 python scripts/run_server.py
```

#### Mode 2: Production VLM Mode (Downloads and Uses `Qwen2-VL-2B-Instruct`)
```powershell
python scripts/run_server.py
```
*(Note: On first run, it will automatically download ~4.5 GB weights from Hugging Face).*

Once started, open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

### Option B: Command-Line Interface (CLI)

You can run single queries directly from the terminal:

```powershell
# Run with fast mock model:
python -m satquery.vqa --image data/samples/sample_patch.tif --question "What is visible in this satellite image?" --mock

# Run with full Qwen2-VL model:
python -m satquery.vqa --image data/samples/sample_patch.tif --question "What land cover types are present?"
```

---

### Option C: Run Automated Tests

To verify that all components (image loader, VQA model, API endpoints, GeoTIFF preview) are working:
```powershell
pytest -v
```
All 16 tests should pass with green checks.

---

## 📂 4. Project Structure

```
satquery-ai/
├── configs/
│   └── model.yaml                # Model candidate matrix & hyperparameters
├── data/
│   └── samples/                  # Sample GeoTIFFs, PNGs, and annotations
├── notebooks/
│   └── 01_bigearthnet_exploration.ipynb  # BigEarthNet dataset schema analysis
├── outputs/
│   ├── executions.jsonl          # Provenance audit logs
│   └── temp_uploads/             # Temp storage for normalized previews
├── scripts/
│   ├── create_sample_data.py     # Script to generate sample GeoTIFFs
│   └── run_server.py             # Server launcher with hot-reload
├── src/
│   └── satquery/
│       ├── api/
│       │   └── main.py           # FastAPI backend (/health, /preview, /vqa, /executions)
│       ├── geo/
│       │   └── image_loader.py   # Multi-band GeoTIFF loader & percentile normalizer
│       ├── schemas/
│       │   └── vqa.py            # Pydantic schemas
│       ├── utils/
│       │   └── logging.py        # Audit & provenance logger
│       └── vqa/
│           ├── inference.py      # Core VQA inference controller
│           └── model.py          # Model loader (Qwen2-VL & MockVQAModel)
├── tests/                        # 16 automated pytest unit tests
├── web/
│   ├── index.html                # Frontend UI
│   ├── style.css                 # Styling & dark-mode theme
│   └── app.js                    # Dynamic preview, VQA dispatcher, metadata drawer
├── pytest.ini                    # Pytest configuration
├── pyproject.toml                # Package configuration
├── requirements.txt              # Project dependencies
└── README.md
```

---

## 🗺️ 5. Next Milestones (Prototypes 2–8)

- **Prototype 2:** Visual Grounding (overlaying bounding boxes on UI for objects/water/roads).
- **Prototype 3:** Bi-Temporal Change Detection & CDVQA (before/after comparison).
- **Prototype 4:** Optical + SAR Cross-Modal Fusion (Sentinel-1 SAR + Sentinel-2 Optical).
- **Prototype 5:** Agentic Query-Driven Model & Tool Routing.
- **Prototype 6:** Domain Adaptation & Fine-Tuning on BigEarthNet.txt.
- **Prototype 7:** Bhoonidhi & ISRO/SAC API Data Connector.
- **Prototype 8:** Final Unified SIH Demonstration System.

---

## 👥 Team Details (Team Saverra)
- **Kumar Amityush** (Team Leader) — *AI/ML & Overall Development*
- **Shreya Shrikant Jadhav** — *Research, Testing & Evaluation*
- **Rashes Kumar Tripathy** — *Vision Language Models & Fine-tuning*
- **Samadrita Dutta Gupta** — *Backend & Integration*
- **Shivam Kumar** — *Frontend & Visualisation*
- **Pavitra Patel** — *Remote Sensing & Image Processing*
