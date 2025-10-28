import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from eo_flood_ops.thresholding_model import ThresholdingModel
from eo_flood_ops.general_utils import (
    tif_to_clipped_masked_array,
    tif_to_clipped_array,
    find_closest_valid,
    generate_wet_dry_timeseries_ds,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run threshold-based flood extent model."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to pickled model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output data.",
    )

    args = parser.parse_args()
