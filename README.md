# EO-Flood-Ops

A Python package for flood extent modeling using Earth Observation (EO) data and threshold-based approaches.

## Overview

EO-Flood-Ops provides tools for processing satellite imagery (particularly SAR data) to detect and map flood extents. The package includes threshold-based models for identifying inundated areas and utilities for working with geospatial raster data.

## Features

- **Threshold-based flood detection**: Automated learning and application of optimal SAR thresholds for flood mapping
- **Geospatial utilities**: Tools for processing GeoTIFF files, clipping, masking, and generating time series data
- **Model persistence**: Save and load trained models for operational deployment
- **Executable generation**: Build standalone executables for deployment in production environments

## Installation

### Development Installation

1. Clone the repository:
```bash
git clone https://github.com/Deltares-research/EO-Flood-Ops.git
cd EO-Flood-Ops
```

2. Create and activate a virtual environment (recommended):
```bash
# Using uv (recommended)
uv sync

# Or using standard Python
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install the package with dependencies:
```bash
# Using uv (faster)
uv pip install -e .

# Or using pip
pip install -e .
```

## Usage

### Running the Threshold Model

The main entry point is `run_threshold_model.py`, which applies a trained threshold model to input data:

```bash
python src/eo_flood_ops/run_threshold_model.py \
    --input path/to/input_data.tif \
    --model path/to/trained_model.pkl \
    --output path/to/output_flood_map.tif
```

**Arguments:**
- `--input`: Path to the input SAR/EO data (GeoTIFF format)
- `--model`: Path to the pickled trained threshold model
- `--output`: Path where the output flood extent map will be saved


## Creating a Standalone Executable

For operational deployment without Python dependencies, you can build a standalone executable using PyInstaller.

### Prerequisites

PyInstaller is included in the project dependencies. Ensure you have activated your virtual environment.

### Build Instructions

**Using the provided spec file** (recommended):

```powershell
# Make sure you're in the project root directory
cd C:\Users\jong\Projects\EO-Flood-Ops

# Build the executable
python -m PyInstaller threshold_model.spec --clean
```


- The spec file includes hidden imports for `rasterio.sample` and `rasterio.vrt` to avoid runtime import errors
- The `--clean` flag ensures a fresh build by removing cached files
- The executable will be created in the `dist/` directory


### Troubleshooting PyInstaller

**ModuleNotFoundError for rasterio modules:**
- Add the missing module to `hiddenimports` in `threshold_model.spec`
- Common modules: `rasterio.sample`, `rasterio.vrt`, `rasterio._shim`, `rasterio.crs`


## Project Structure

```
EO-Flood-Ops/
├── src/
│   └── eo_flood_ops/
│       ├── __init__.py
│       ├── run_threshold_model.py      # Main CLI script
│       ├── thresholding_model.py       # Threshold model implementation
│       ├── manifold_model.py           # Manifold-based model
│       ├── model_utils.py              # Model utilities and data structures
│       └── general_utils.py            # Geospatial utility functions
├── notebooks/                          # Jupyter notebooks for testing/demos
├── scripts/                            # Additional scripts
├── threshold_model.spec                # PyInstaller specification file
├── pyproject.toml                      # Project metadata and dependencies
└── README.md                           # This file
```

## Dependencies

Key dependencies include:
- `rasterio` - Geospatial raster I/O
- `geopandas` - Geospatial data handling
- `numpy` & `scipy` - Numerical computing
- `scikit-image` - Image processing
- `xarray` - Multi-dimensional arrays
- `matplotlib` - Visualization
- `PyInstaller` - Executable generation

See `pyproject.toml` for the complete list.


