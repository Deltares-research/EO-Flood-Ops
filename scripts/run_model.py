import argparse
import os
import pickle
import pandas as pd
import warnings
from affine import Affine

from eo_flood_ops.general_utils import generate_timeseries_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run threshold-based flood extent model."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input water level data.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to pickled model.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to the directory to save the output data.",
    )

    args = parser.parse_args()

    waterlevels_fn = args.input
    model_path = args.model
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    # Read timezone info from the first line
    with open(waterlevels_fn, "r") as f:
        tz_line = f.readline().strip()  # e.g. "GMT+7,ID6"

    # Split by comma and extract the first part ("GMT+7")
    tz_part = tz_line.split(",")[0].strip()  # "GMT+7"

    # Parse numeric offset (e.g. +7)
    offset_hours = int(tz_part.replace("GMT", ""))

    # Read actual data (skip the first line only) 
    df_water = pd.read_csv(waterlevels_fn, skiprows=[0])
    df_water.rename(columns={df_water.columns[0]: "datetime", df_water.columns[1]: "water_level"}, inplace=True)

    # Parse datetimes and convert to UTC 
    df_water["datetime"] = pd.to_datetime(df_water["datetime"], format="%Y-%m-%d %H:%M:%S")

    # Assign timezone (GMT+7) then convert to UTC (GMT+0)
    # NOTE: for "GMT+7", you must localize with "Etc/GMT-7" because of reversed sign convention
    df_water["datetime"] = (
        df_water["datetime"]
        .dt.tz_localize(f"Etc/GMT{-offset_hours}")
        .dt.tz_convert("UTC")
    )

    # Check for missing or invalid values (-999)
    invalid_rows = df_water[df_water["water_level"] == -999]

    if not invalid_rows.empty:
        warnings.warn(
            f"Removed {len(invalid_rows)} row(s) with missing value (-999) at timestamps: {invalid_rows['datetime'].to_list()}",
            UserWarning
        )
        # Drop invalid rows
        df_water = df_water[df_water["water_level"] != -999]

    gauge_levels = df_water["water_level"].to_numpy()
    timestamps = df_water["datetime"]

    # Load the trained model back
    with open(model_path, "rb") as f:
        test_tm = pickle.load(f)
    print("Model successfully loaded!")

    # NOTE here we are hardcoding the transform for the Hoi An test case.
    # TODO: read the transform from a file e.g. in FEWS inputs.
    transform = Affine(
        0.00026949458523585647, 0.0, 108.04873355319717,
        0.0, -0.00026949458523585647, 16.08101139560879
    )

    ds = generate_timeseries_ds(
        gauge_levels,
        timestamps,
        test_tm,
        transform,
    )

    # Convert time to naive datetime64 (UTC)
    ds = ds.assign_coords(
        time=pd.DatetimeIndex(ds.time.values).tz_localize(None)
    )

    # Save to NetCDF
    # the filename part from the first line, e.g. "ID6"
    output_fn = tz_line.split(",")[-1].strip() 
    output_path = f"{output_dir}/output_{output_fn}.nc"
    ds.to_netcdf(output_path)