# EO-Flood-Ops
A Python package for **flood extent modeling** using Earth Observation (EO) data and threshold-based machine learning (ML) approaches. 

EO-Flood-Ops provides tools to process satellite imagery, particularly **SAR data**, to detect and map flood extents.  
It is suitable for both **research** and **operational flood monitoring** applications.


## Overview

This package implements two main ML flood modeling approaches:

- **Thresholding Model:** Computes flood inundation extent.  
- **Manifold Model:** Computes both inundation **extent** and **depth**, providing an alternative to hydraulic flood modeling.  

Both models were originally developed by **Google** and are described in their paper: [Nevo et al., 2022](https://hess.copernicus.org/articles/26/4013/2022/hess-26-4013-2022.html).

> **Note:** The original code for these models is licensed under the **Apache License, Version 2.0**. You can view the license [here](https://www.apache.org/licenses/LICENSE-2.0).

EO-Flood-Ops provides tools to **prepare all necessary datasets** — including water levels, DEMs, and satellite-derived flood extents — for training the models. The package also demonstrates how to **apply these models operationally** to generate flood extent predictions.

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

### Training a model

You can train and use different types of flood prediction models using the provided classes in the `eo_flood_ops` package:

- **Thresholding Model:** Use the `ThresholdingModel` class from `eo_flood_ops.thresholding_model`.  
- **Manifold Model:** Use the `ManifoldModel` class from `eo_flood_ops.manifold_model`.

Data required for model training — such as **water levels**, **satellite-derived flood extents**, and **digital elevation models (DEM)** — can be prepared using the utility functions available in `eo_flood_ops.general_utils`.

> Note: A generic model training function is currently under development (WIP).

For examples on how to train and use the models, please refer to the **notebooks** section of this repository.


### Running a model

The main entry point is `run_model.py`, which applies a trained threshold model or manifold model to input data:

```bash
python src/eo_flood_ops/run_model.py \
    --input path/to/input_water_level_data.csv \
    --model path/to/trained_model.pkl \
    --output path/to/output_flood_map.tif
```
**Arguments:**
- `--input`: Path to the input water level data (CSV format)
- `--model`: Path to the pickled trained model
- `--output`: Path where the output flood extent map will be saved

Note that the input CSV file can be exported from **DELFT-FEWS** and must follow this structure:

| GMT+7 | ID6     |
|-------|---------|
|       | H.obs   |
| 2023-11-14 01:00:00 | 1.12 |
| 2023-11-14 02:00:00 | 1.03 |


where:
- The first row specifies the *time zone and station ID.  
- The second row contains the column headers (`H.obs` = observed water level).  
- Subsequent rows contain timestamped water level observations.  

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
python -m PyInstaller eo_flood_ops_model.spec --clean
```


- The spec file includes hidden imports for `rasterio.sample` and `rasterio.vrt` to avoid runtime import errors
- The `--clean` flag ensures a fresh build by removing cached files
- The executable will be created in the `dist/` directory


### Troubleshooting PyInstaller

**ModuleNotFoundError for rasterio modules:**
- Add the missing module to `hiddenimports` in `eo_flood_ops_model.spec`
- Common modules: `rasterio.sample`, `rasterio.vrt`, `rasterio._shim`, `rasterio.crs`


## Project Structure

```
EO-Flood-Ops/
├── src/
│   └── eo_flood_ops/
│       ├── __init__.py
│       ├── run_model.py                # Main CLI script
│       ├── thresholding_model.py       # Threshold model implementation
│       ├── manifold_model.py           # Manifold-based model
│       ├── model_utils.py              # Model utilities and data structures
│       └── general_utils.py            # Geospatial utility functions
├── notebooks/                          # Jupyter notebooks for testing/demos
├── scripts/                            # Additional scripts
├── eo_flood_ops_model.spec             # PyInstaller specification file
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


