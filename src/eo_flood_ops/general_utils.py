import rasterio
import numpy as np
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
import warnings
from shapely.geometry import box
from rasterio.features import geometry_mask
import xarray as xr


def tif_to_clipped_masked_array(
    tif_path: str, geojson_path: str
) -> tuple[np.ma.MaskedArray, rasterio.Affine, rasterio.crs.CRS]:
    """
    Clip a binary TIFF with a polygon from a GeoJSON file
    and return it as a masked array of True/False (wet/dry),
    with everything outside the polygon masked.

    Args:
        tif_path: Path to the .tif file.
        geojson_path: Path to the polygon (GeoJSON or shapefile).

    Returns:
        Tuple of:
            - np.ma.MaskedArray of clipped region (True=wet, False=dry)
            - Affine transform of the clipped raster
            - CRS of the raster
    """
    # Load polygon
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        raise ValueError("Input GeoJSON has no CRS.")

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError("Input TIFF has no CRS.")
        if src.crs.to_epsg() != 4326:
            raise ValueError(f"Input TIFF must be in EPSG:4326, found {src.crs}")
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # Crop to bounding box first
        minx, miny, maxx, maxy = gdf.total_bounds
        bbox_geom = [box(minx, miny, maxx, maxy).__geo_interface__]
        clipped_data, clipped_transform = mask(src, bbox_geom, crop=True)
        data = clipped_data[0]

        # Mask everything outside the polygon
        polygon_mask = geometry_mask(
            geometries=gdf.geometry,
            out_shape=data.shape,
            transform=clipped_transform,
            invert=True,  # True inside polygon
        )

        # Convert binary 1/0 to True/False
        binary_mask = data == 1

        # Apply polygon mask: everything outside polygon is masked
        masked_array = np.ma.masked_array(binary_mask, mask=~polygon_mask)

    return masked_array, clipped_transform, src.crs


def tif_to_clipped_array(
    tif_path: str, geojson_path: str
) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    """
    Clip DEM to the bounding box of the AOI but set everything outside the AOI polygon to NaN.
    Returns a numpy array (DEM values) with NaNs where data are outside the polygon.
    """
    gdf = gpd.read_file(geojson_path)

    with rasterio.open(tif_path) as src:
        if gdf.crs is None:
            raise ValueError("Input GeoJSON has no CRS.")
        if src.crs is None:
            raise ValueError("Input TIFF has no CRS.")
        if src.crs.to_epsg() != 4326:
            raise ValueError(f"Input TIFF must be in EPSG:4326, found {src.crs}")
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # Get the bounding box geometry (still needed to crop to a small raster)
        minx, miny, maxx, maxy = gdf.total_bounds
        bbox_geom = [box(minx, miny, maxx, maxy).__geo_interface__]

        # Crop the DEM to the bounding box area
        clipped_data, clipped_transform = mask(
            src, bbox_geom, crop=True, filled=True, nodata=np.nan
        )
        data = clipped_data[0]

        # Now create a polygon mask (True inside polygon, False outside)

        polygon_mask = geometry_mask(
            geometries=gdf.geometry,
            out_shape=data.shape,
            transform=clipped_transform,
            invert=True,  # True inside polygon
        )

        # Set everything outside polygon to NaN
        data = np.where(polygon_mask, data, np.nan)

    return data, clipped_transform, src.crs


# Function to find closest valid value
def find_closest_valid(df, dt, station):
    # Compute absolute difference
    df["diff"] = (df["datetime"] - dt).abs()

    # Sort by time difference
    df_sorted = df.sort_values("diff")

    # Iterate to find first non -999 value
    for idx, row in df_sorted.iterrows():
        value = row[station]
        if value != -999:
            time_diff = row["datetime"] - dt
            abs_diff = abs(time_diff)

            # Check thresholds
            if abs_diff > pd.Timedelta(days=1):
                raise ValueError(
                    f"Closest valid obs for {station} is {abs_diff} away "
                    f"(at {row['datetime']}). Image timestamp: {dt}"
                )
            elif abs_diff > pd.Timedelta(hours=5):
                warnings.warn(
                    f"Closest valid obs for {station} is {abs_diff} away "
                    f"(at {row['datetime']}). Image timestamp: {dt}"
                )

            if idx != df_sorted.index[0]:
                print(
                    "Closest timestamp initially had -999, using next closest valid value."
                )
            return row["datetime"], value

    # If all values are -999
    return None, None


def generate_wet_dry_timeseries_ds(water_levels, timestamps, tm, transform, crs):
    """
    Generate a CF-compliant xarray Dataset of wet/dry masks (boolean) or
    water depths (float) for a time series of water levels using a trained model.
    Includes water_level as a coordinate.

    Args:
        water_levels (list or array): Water levels to infer wet/dry masks or depths.
        timestamps (list or pd.DatetimeIndex): Corresponding timestamps.
        tm: Trained model with method tm.infer(level).
            - Returns boolean arrays for thresholding models.
            - Returns masked float arrays for manifold models.
        transform (Affine): Rasterio affine transform of the raster.
        crs (rasterio.crs.CRS): CRS of the raster (must be EPSG:4326).

    Returns:
        xr.Dataset: CF-compliant Dataset with dimensions (time, lat, lon)
    """

    def array_to_da(array, is_bool):
        """Convert boolean/float/masked array to xarray.DataArray with lat/lon."""
        if not isinstance(array, np.ma.MaskedArray):
            masked = np.ma.masked_array(array)
        else:
            masked = array

        nrows, ncols = masked.shape
        xs = np.arange(ncols)
        ys = np.arange(nrows)
        lon, _ = rasterio.transform.xy(transform, 0, xs, offset="center")
        _, lat = rasterio.transform.xy(transform, ys, 0, offset="center")
        lon = np.array(lon)
        lat = np.array(lat)

        # Ensure latitude is sorted ascending
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            masked = masked[::-1, :]

        # Convert data
        if is_bool:
            data = masked.astype(np.float32)  # True/False → 1.0/0.0
        else:
            data = masked.filled(np.nan).astype(np.float32)

        da = xr.DataArray(
            data,
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )

        if is_bool:
            da.attrs.update({"long_name": "wet_dry_mask", "units": "1"})
        else:
            da.attrs.update({"long_name": "water_depth", "units": "m"})

        da.attrs["grid_mapping"] = "spatial_ref"
        return da

    # Detect model type (boolean vs float)
    test_out = tm.infer(water_levels[0])
    is_bool = np.issubdtype(test_out.dtype, np.bool_)

    arrays = []
    for lvl in water_levels:
        arr = tm.infer(lvl)
        da = array_to_da(arr, is_bool)
        arrays.append(da)

    # Stack along time
    varname = "wet_dry" if is_bool else "water_depth"
    ts_da = xr.concat(arrays, dim=pd.Index(timestamps, name="time"))

    # Build Dataset
    ds = xr.Dataset({varname: ts_da})

    # Add water_level as coordinate along time
    ds = ds.assign_coords({"water_level": ("time", water_levels)})
    ds["water_level"].attrs.update({"long_name": "water_level", "units": "m"})

    # Add CF-compliant spatial_ref
    ds["spatial_ref"] = xr.DataArray(
        0,
        attrs={
            "spatial_ref": crs.to_wkt(),
            "grid_mapping_name": "latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
    )
    ds[varname].attrs["grid_mapping"] = "spatial_ref"

    # Coordinate metadata
    ds["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    ds["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    ds.raster.set_crs("EPSG:4326")

    return ds


def masked_array_to_da(
    masked_array: np.ma.MaskedArray, transform: rasterio.Affine, crs: rasterio.crs.CRS
) -> xr.DataArray:
    """
    Convert a single masked wet/dry raster (masked array from tif_to_clipped_masked_array)
    into an xarray.DataArray with lat/lon coordinates.

    Args:
        masked_array (np.ma.MaskedArray): Input masked wet/dry array (True=wet, False=dry).
        transform (Affine): Rasterio affine transform.
        crs (rasterio.crs.CRS): CRS of the raster (must be EPSG:4326).

    Returns:
        xr.DataArray: DataArray with dimensions (lat, lon).
    """
    nrows, ncols = masked_array.shape
    xs = np.arange(ncols)
    ys = np.arange(nrows)

    # lon from col centers, lat from row centers
    lon, _ = rasterio.transform.xy(transform, 0, xs, offset="center")
    _, lat = rasterio.transform.xy(transform, ys, 0, offset="center")
    lon = np.array(lon)
    lat = np.array(lat)

    # Convert masked_array → float with NaNs
    data = masked_array.astype(np.float32)
    data = np.ma.filled(data, np.nan)

    da = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "wet_dry_mask",
            "units": "1",
            "grid_mapping": "spatial_ref",
        },
    )

    # Add CF-compliant spatial_ref
    da = da.to_dataset(name="wet_dry")["wet_dry"]
    da.attrs["grid_mapping"] = "spatial_ref"
    da.coords["spatial_ref"] = xr.DataArray(
        0,
        attrs={
            "spatial_ref": crs.to_wkt(),
            "grid_mapping_name": "latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
    )

    # Metadata
    da["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    da["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})

    return da


def skill(da_sim, da_obs, da_msk, hmin=0.15):
    """
    Compute skill metrics (CSI, hit rate, false alarm rate, bias) and
    a confusion map between simulated and observed inundation.

    Args:
        da_sim (xr.DataArray): Simulated water depth (float).
        da_obs (xr.DataArray): Observed wet/dry mask (binary).
        da_msk (xr.DataArray): Mask (True = invalid, False = valid).
        hmin (float): Depth threshold [m] for considering a cell wet in simulation.

    Returns:
        ds_skill (xr.Dataset): Skill metrics.
        da_cm (xr.DataArray): Confusion map with codes:
                              3=true positive, 2=false positive, 1=false negative, 0=true negative.
    """
    # detect spatial dims automatically
    spatial_dims = da_sim.dims  # e.g. ('lat', 'lon') or ('y', 'x')

    # thresholding: sim > hmin = wet, obs > 0 = wet
    sim = (da_sim > hmin).where(~da_msk, False)
    obs = (da_obs > 0).where(~da_msk, False)

    ds = xr.Dataset(
        dict(
            true_pos=np.logical_and(sim, obs),
            false_neg=np.logical_and(~sim, obs),
            false_pos=np.logical_and(sim, ~obs),
        )
    )

    # totals
    ntot = np.logical_or(sim, obs).where(~da_msk, False).sum(spatial_dims)
    nobs = obs.where(~da_msk, False).sum(spatial_dims)
    nsim = sim.where(~da_msk, False).sum(spatial_dims)

    # metrics
    hit_rate = ds["true_pos"].sum(spatial_dims) / nobs
    false_rate = ds["false_pos"].sum(spatial_dims) / nsim
    csi = ds["true_pos"].sum(spatial_dims) / ntot
    bias = ds["false_pos"].sum(spatial_dims) / ds["false_neg"].sum(spatial_dims)

    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1_score = 2 * precision * recall / (precision + recall)

    # F1 score using ds
    precision = ds["true_pos"].sum(spatial_dims) / (
        ds["true_pos"].sum(spatial_dims) + ds["false_pos"].sum(spatial_dims)
    )
    recall = ds["true_pos"].sum(spatial_dims) / (
        ds["true_pos"].sum(spatial_dims) + ds["false_neg"].sum(spatial_dims)
    )
    f1_score = 2 * precision * recall / (precision + recall)

    ds_skill = xr.merge(
        [
            csi.rename("C"),  # Critical Success Index
            hit_rate.rename("H"),  # Hit rate
            false_rate.rename("F"),  # False alarm ratio
            bias.rename("E"),  # Bias
            f1_score.rename("F1"),  # F1 Score
        ]
    )

    # confusion map: 3=true pos, 2=false pos, 1=false neg, 0=true neg
    da_cm = (ds["true_pos"] * 3 + ds["false_pos"] * 2 + ds["false_neg"] * 1).astype(
        np.uint8
    )

    # or leave both masked an true neg as 0
    da_cm = da_cm.where(~da_msk, np.nan)

    # preserve CRS if available
    if hasattr(da_sim, "raster"):
        da_cm.raster.set_crs(da_sim.raster.crs)
        da_cm.raster.set_nodata(0)

    return ds_skill, da_cm
