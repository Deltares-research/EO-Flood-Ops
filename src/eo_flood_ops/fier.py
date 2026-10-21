import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import hydroeval as he
import pyproj
from datetime import datetime
from pystac_client import Client
import odc.stac as odc_stac
from collections.abc import Sequence

def reof(
    da: xr.DataArray,
    n_modes: int = 4,
    significance_test: bool = False,
    time_dim: str = "time",
    y_dim: str = "lat",
    x_dim: str = "lon",
    method: str = "varimax",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> xr.Dataset:
    """
    Perform Rotated Empirical Orthogonal Function (REOF) on multi-temporal satellite images

    args:
        da (xr.DataArray): DataArray of spatial temporal values with dims (time_dim, y_dim, x_dim)
        n_modes (int): number of eof modes to use when significance_test is False. default = 4
        significance_test (bool): if True, determine n_modes using a significance test. default = False
        time_dim (str): name of the time dimension/coordinate. default = "time"
        y_dim (str): name of the latitude/y dimension/coordinate. default = "lat"
        x_dim (str): name of the longitude/x dimension/coordinate. default = "lon"
        method (str): rotation method passed to `_ortho_rotation`, "varimax" or "quartimax". default = "varimax"
        tol (float): convergence tolerance passed to `_ortho_rotation`. default = 1e-6
        max_iter (int): maximum number of rotation iterations passed to `_ortho_rotation`. default = 100

    returns:
        reof_ds (xr.Dataset): Dataset of REOF results with spatial modes, temporal modes, mean of data, and
        explained variances as variables
    """
    if method not in ["varimax", "quartimax"]:
        raise ValueError("method must be 'varimax' or 'quartimax'")
    if not np.isscalar(tol) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a finite value greater than zero")
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool):
        raise TypeError("max_iter must be an integer")
    if max_iter < 1:
        raise ValueError("max_iter must be at least one")
    if not isinstance(significance_test, (bool, np.bool_)):
        raise TypeError("significance_test must be a boolean")

    time_size = da.sizes[time_dim]
    y_size = da.sizes[y_dim]
    x_size = da.sizes[x_dim]
    space_size = y_size * x_size

    if significance_test:
        print("Performing significance test to determine the number of EOF modes...")
        n_modes = _sig_eof_test(
            da=da,
            time_dim=time_dim,
            y_dim=y_dim,
            x_dim=x_dim,
        )
        print(f"Number of EOF modes determined by significance test: {n_modes}")

    print(f"Performing REOF analysis using the first {n_modes} modes...")

    temporal_mean = np.nanmean(da.values, axis=0)

    flat_da = da.values.reshape((time_size, space_size))
    aoi_mask_flat = ~(np.isnan(flat_da).all(axis=0))
    flat_da = flat_da[:, aoi_mask_flat]
    flat_da = flat_da - flat_da.mean(axis=0, keepdims=True)

    # Record the original index of each pixels inside the AOI
    # to recover the flattened data back to geographical dimension
    flat2geo = np.arange(np.prod(space_size))[aoi_mask_flat].reshape(1, -1)

    # sklearn PCA
    pca = PCA(n_components=n_modes)
    pca.fit_transform(flat_da)
    expvar = pca.explained_variance_ratio_
    eofs = (pca.components_).T

    rotated_eofs = _ortho_rotation(
        eofs, method=method, 
        tol=tol, 
        max_iter=max_iter
    )

    # project the original time series data on the rotated eofs
    rotated_pcs = np.dot(flat_da, rotated_eofs)

    # get variance of each rotated mode
    rotated_var = np.var(rotated_pcs, axis=0)

    # get cumulative variance of all rotated modes
    total_rotated_var = rotated_var.cumsum()[-1]

    # get variance fraction of each rotated mode
    rotated_var_frac = (rotated_var / total_rotated_var) * np.sum(expvar)

    # recover the flattened, AOI-only EOF array back to the geographical dimension
    rotated_eofs = rotated_eofs.T
    rotated_pcs = rotated_pcs.T

    # index of explained variance in the descending order
    sort_rotated_var_frac = np.argsort(-1 * rotated_var_frac)  
    rotated_var_frac = rotated_var_frac[sort_rotated_var_frac]

    # sort modes based on variance fraction of REOF
    indx_rotated_var_frac_sort = np.expand_dims(sort_rotated_var_frac, axis=-1)  # Mode X 1
    rotated_pcs = np.take_along_axis(rotated_pcs, indx_rotated_var_frac_sort, axis=0)

    rotated_eofs = np.take_along_axis(rotated_eofs, indx_rotated_var_frac_sort, axis=0)
    fill_eofs = np.ones((n_modes, space_size)) * np.nan
    for ct_c in np.arange(rotated_eofs.shape[1]):
        fill_eofs[:, flat2geo[0, ct_c]] = rotated_eofs[:, ct_c]
    rec_eofs_img = fill_eofs.reshape(n_modes, y_size, x_size)

    # define data with variable attributes
    data_vars = {
        "spatial_modes": (["mode", y_dim, x_dim], rec_eofs_img),
        "temporal_modes": (["mode", time_dim], rotated_pcs),
        "temporal_mean": ([y_dim, x_dim], temporal_mean),
        "explained_var": (["mode"], rotated_var_frac),
    }

    # define coordinates
    coords = {
        time_dim: ([time_dim], da[time_dim].values),
        y_dim: ([y_dim], da[y_dim].values, {"units": "degrees North"}),
        x_dim: ([x_dim], da[x_dim].values, {"units": "degrees East"}),
        "mode": (["mode"], np.arange(n_modes) + 1),
    }

    # create dataset
    reof_ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
    )

    return reof_ds

def find_hydro_mode(
    ds_eof: xr.Dataset,
    da_hydro: xr.DataArray,
    r_thrd: float = 0.5,
    p_thrd: float = 0.05,
    time_dim: str = "time",
    site_dim: str = "site",
    warn_thrd: pd.Timedelta = pd.Timedelta(hours=5),
    error_thrd: pd.Timedelta = pd.Timedelta(days=1),
) -> xr.Dataset:
    """
    Calculate correlation between temporal patterns and hydrological data.
    This helps determine water-related mode. By default,  >=0.5 is considered to be correlated.

    args:
       ds_eof: Dataset with REOF results produced by the `reof()` function
       da_hydro: DataArray with hydrological data
       r_thrd: Threshold (default: 0.5) of correlation coefficient to decide which modes are water-related
       p_thrd: Threshold (default: 0.05) of p-value below which a correlation is considered significant
       time_dim: name of the time dimension/coordinate shared between ds_eof and da_hydro. default = "time"
       site_dim: name of the site/gauge dimension/coordinate in da_hydro. default = "site"
       warn_thrd: Warning threshold for time difference between satellite and hydro data (default: 5 hours)
       error_thrd: Error threshold for time difference between satellite and hydro data (default: 1 day)
    output (site X modes):
       site: Names of selected sites of the modes
       best_r: The highest correlations between RTPCs and hydrological data
       best_p: Corresponding p-values of best_r
    """
    # "mode" is always the dim name produced by reof(), 
    # so it is not parameterized here
    mode_dim = "mode"

    # get number of mode
    mode_num = ds_eof.sizes[mode_dim]
    # get number of hydrological data sites
    site_num = da_hydro.sizes[site_dim]

    r = np.zeros((site_num, mode_num))
    p = np.zeros((site_num, mode_num))

    time_tpc = ds_eof.temporal_modes.sel({mode_dim: int(1)})[time_dim]
    # hydro measurements rarely land on the exact satellite acquisition timestamps,
    # so match each satellite timestamp to the closest valid hydro observation instead
    hydro_common = _find_closest_valid_hydro(
        da_hydro=da_hydro, 
        times= time_tpc.values, 
        time_dim=time_dim, 
        site_dim=site_dim,
        warn_thrd = warn_thrd,
        error_thrd = error_thrd,
    )
    hydro_site = hydro_common[site_dim].values
    hydro_values = hydro_common.values

    n = hydro_values.shape[-1]
    p_dist = stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)

    # Instead of using FOR, see if it is possible to vectorize the process
    for ct_mode in range(mode_num):
        # get mode of tpc
        tpc = ds_eof.temporal_modes.sel({mode_dim: int(ct_mode+1)})
        tpc = tpc.expand_dims(dim=site_dim, axis=0).values.reshape(1,-1)

        r_da = np.concatenate((tpc, hydro_values), axis=0)
        r_mode = np.expand_dims((pd.DataFrame(np.transpose(r_da)).corr(method='pearson').values[1:,0]), axis=1)
        p_mode = 2*p_dist.cdf(-abs(r_mode))

        if ct_mode==0:
            r = r_mode
            p = p_mode
        else:
            r = np.concatenate((r,r_mode), axis=1)
            p = np.concatenate((p,p_mode), axis=1)

    indx_max_r_site = np.nanargmax(np.abs(r), axis=0)
    r_temp = r[( indx_max_r_site, list(range(mode_num)) )]
    p_temp = p[( indx_max_r_site, list(range(mode_num)) )]

    mode_out = ((np.logical_and(np.abs(r_temp) >= r_thrd, p_temp <= p_thrd).astype(int)).nonzero())[0]

    site_out = indx_max_r_site[mode_out]
    r_out = r_temp[mode_out]
    p_out = p_temp[mode_out]

    tpc_hydro_r = xr.Dataset(

        data_vars=dict(
            site=([mode_dim], hydro_site[site_out]),
            best_r=([mode_dim], r_out),
            best_p=([mode_dim], p_out)
        ),

        coords={
            mode_dim: ([mode_dim], ds_eof[mode_dim].values[mode_out]),
        },

    )

    return tpc_hydro_r

def combine_eof_hydro(
    ds_eof: xr.Dataset,
    da_hydro: xr.DataArray,
    tpc_hydro_r: xr.Dataset,
    time_dim: str = "time",
    site_dim: str = "site",
    degree: int = 2,
) -> xr.Dataset:
    """
    Combine REOF/EOF results for the water-related modes (output of `find_hydro_mode`)
    with their matched hydrological data, and fit a polynomial relating each mode's
    temporal pattern to its water level.

    args:
        ds_eof: Dataset with EOF or REOF results (output of `reof`)
        da_hydro: DataArray with hydrological data
        tpc_hydro_r: Dataset with the water-related modes (output of `find_hydro_mode`),
            with variables `site` and `best_r`
        time_dim: name of the time dimension/coordinate shared between ds_eof and da_hydro. default = "time"
        site_dim: name of the site/gauge dimension/coordinate in da_hydro. default = "site"
        degree: degree of the polynomial fit relating water level to temporal mode amplitude. default = 2

    returns:
        xr.Dataset: ds_eof subset to the water-related modes, merged with each mode's
        matched hydro series (as `hydro_var`), plus `site`, `best_r`, and per-mode
        polynomial `coefficients`
    """
    mode_dim = "mode"

    eof_sel = ds_eof.sel({mode_dim: tpc_hydro_r[mode_dim].values})

    # match each mode's RTPC timestamps to the closest valid observation at its best-matching gauge
    hydro_sel = xr.concat(
        [
            _find_closest_valid_hydro(
                da_hydro.sel({site_dim: [site]}),
                eof_sel[time_dim].values,
                time_dim=time_dim,
                site_dim=site_dim,
            ).sel({site_dim: site})
            for site in tpc_hydro_r.site.values
        ],
        dim=mode_dim,
    )
    hydro_sel = hydro_sel.assign_coords({mode_dim: tpc_hydro_r[mode_dim].values})
    hydro_sel.name = "hydro_var"

    eof_hydro = xr.merge([eof_sel, hydro_sel]).astype(np.float32)
    eof_hydro["site"] = ([mode_dim], tpc_hydro_r.site.values)
    eof_hydro["best_r"] = ([mode_dim], tpc_hydro_r.best_r.values)

    n_sel_modes = eof_hydro.sizes[mode_dim]
    n_coef = degree + 1

    coef_mat = np.zeros([n_sel_modes, n_coef])
    for i in range(n_sel_modes):
        mode_ds = eof_hydro.isel({mode_dim: i})
        q = mode_ds["hydro_var"]
        tpc = mode_ds["temporal_modes"]

        p = np.polynomial.Polynomial.fit(q, tpc, degree)
        coef_mat[i, :] = p.convert().coef

    eof_hydro["degree"] = np.arange(n_coef)
    eof_hydro["coefficients"] = ([mode_dim, "degree"], coef_mat.astype(np.float32))

    return eof_hydro

def synthesize(
        reof_ds: xr.Dataset, 
        da_hydro: xr.DataArray, 
        site_dim: str = "site",
        *,
        threshold: float | None = None,
) -> xr.DataArray:
    """Reconstruct an image using the site-matched polynomial for each mode.

    Args:
        reof_ds: Output of combine_eof_hydro(). It must contain ``site``
            indexed by ``mode`` and ``coefficients`` indexed by ``mode`` and
            ``degree``.
        da_hydro: Hydrological data indexed by ``site_dim`` (also used for training the polynomial). A single time step is
            expected, for example ``da_hydro.sel(time=date_str)``.
        site_dim: Name of the site dimension in ``da_hydro``.
        threshold: Optional calibrated cutoff. If provided, return 0 for land,
            1 for water (score >= threshold), and NaN for missing pixels.
    """
    if not isinstance(da_hydro, xr.DataArray):
        raise TypeError("da_hydro must be an xarray.DataArray indexed by site_dim")
    if site_dim not in da_hydro.dims:
        raise ValueError(f"da_hydro must contain the site dimension {site_dim!r}")
    if "site" not in reof_ds or "coefficients" not in reof_ds:
        raise ValueError("reof_ds must contain 'site' and 'coefficients'")

    modes = reof_ds["mode"].values
    predicted_pcs = []
    used_modes = []

    for mode in modes:
        site = reof_ds["site"].sel(mode=mode).item()
        try:
            da_hydro_value = da_hydro.sel({site_dim: site}).item()
        except KeyError:
            continue

        if not np.isfinite(da_hydro_value):
            continue

        coefficients = reof_ds["coefficients"].sel(mode=mode).values
        pc_value = np.polynomial.polynomial.polyval(da_hydro_value, coefficients)
        predicted_pcs.append(pc_value)
        used_modes.append(mode)

    if not used_modes:
        raise ValueError("No mode sites were found in da_hydro with finite values")

    pcs = xr.DataArray(
        predicted_pcs,
        dims="mode",
        coords={"mode": used_modes},
    )
    reconstructed = pcs.dot(reof_ds["spatial_modes"].sel(mode=used_modes))
    reconstructed = reconstructed + reof_ds["temporal_mean"]

    reconstructed = reconstructed.astype(np.float32).rename("synthesized")
    if threshold is not None:
        return correct_binary_image(reconstructed, threshold)
    return reconstructed


def correct_binary_image(
    predicted_image: xr.DataArray,
    threshold: float = 0.5,
) -> xr.DataArray:
    """Classify scores as land (0) or water (1), preserving nonfinite pixels as NaN."""
    if not isinstance(predicted_image, xr.DataArray):
        raise TypeError("predicted_image must be an xarray.DataArray")
    if not np.isscalar(threshold) or not np.isfinite(threshold):
        raise ValueError("threshold must be a finite scalar")
    return (
        (predicted_image.astype(np.float64) >= threshold)
        .astype(np.float32)
        .where(np.isfinite(predicted_image))
        .rename("floodmap")
    )


def calibrate_synthesis_threshold(
    reof_ds: xr.Dataset,
    training_images: xr.DataArray,
    time_dim: str = "time",
) -> xr.Dataset:
    """Synthesize every training image and maximize pooled water F1.

    Uses the per-mode water levels saved as ``hydro_var`` by
    ``combine_eof_hydro``, not the observed temporal-mode amplitudes.
    Images must have exactly the training times and spatial grid, with only
    0 (land), 1 (water), or NaN (NoData). Every image needs valid observations
    and every mode needs a finite water level at every training time.

    All distinct score boundaries are evaluated using ``score >= threshold``;
    scores are not clipped to [0, 1]. Ties favor the candidate closest to 0.5,
    then the smaller threshold. F1 is zero when its denominator is zero.
    Requires at least one observed water pixel. All valid pixel/date pairs
    are pooled, so this is not a mean of per-image F1 scores.

    Returns the scalar ``best_threshold``, threshold/F1/confusion-count curves,
    continuous ``reconstructed`` images, ``binary`` images, and per-image
    confusion counts and F1. NoData is excluded from all metrics.
    Reconstruction and the exact sorted search run in memory.
    This is in-sample calibration, not an independent validation score.
    """
    if not isinstance(training_images, xr.DataArray):
        raise TypeError("training_images must be an xarray.DataArray")
    if time_dim not in training_images.dims or training_images.sizes[time_dim] == 0:
        raise ValueError("training_images must contain a nonempty time dimension")
    required = {"hydro_var", "site", "coefficients", "spatial_modes", "temporal_mean"}
    missing = required.difference(reof_ds.variables)
    if missing:
        raise ValueError(f"reof_ds is missing required variables: {sorted(missing)}")
    if reof_ds.sizes.get("mode", 0) == 0:
        raise ValueError("reof_ds must contain at least one mode")
    expected_dims = {time_dim, *reof_ds["temporal_mean"].dims}
    if set(training_images.dims) != expected_dims:
        raise ValueError("training_images dimensions must match time and the model spatial grid")
    xr.align(training_images, reof_ds, join="exact", copy=False)
    if reof_ds["hydro_var"].dims != ("mode", time_dim):
        raise ValueError("hydro_var must have dimensions ('mode', time_dim)")
    if not np.isfinite(reof_ds["hydro_var"].values).all():
        raise ValueError("Every training time must have finite hydro_var for every mode")

    images = training_images.transpose(time_dim, *reof_ds["temporal_mean"].dims)
    observed = np.asarray(images.values)
    valid = ~np.isnan(observed)
    if not np.isin(observed[valid], [0, 1]).all():
        raise ValueError(
            "training_images must contain only 0, 1, or NaN; mask NoData codes "
            "(e.g. 255) before training REOF and calibrating"
        )
    if not valid.reshape(images.sizes[time_dim], -1).any(axis=1).all():
        raise ValueError("Every training image must contain observed pixels")
    if not np.any(observed[valid] == 1):
        raise ValueError("Water F1 calibration requires at least one observed water pixel")

    sites, site_indices = np.unique(reof_ds["site"].values, return_index=True)
    for site in sites:
        levels = reof_ds["hydro_var"].isel(
            mode=np.flatnonzero(reof_ds["site"].values == site)
        ).values
        if not np.all(levels == levels[0]):
            raise ValueError(f"Inconsistent training water levels for site {site!r}")

    reconstructed = xr.concat(
        [
            synthesize(
                reof_ds,
                xr.DataArray(
                    reof_ds["hydro_var"].isel({time_dim: index}).values[site_indices],
                    dims="site",
                    coords={"site": sites},
                ),
            )
            for index in range(images.sizes[time_dim])
        ],
        dim=images[time_dim],
    ).transpose(*images.dims)
    if not np.isfinite(reconstructed.values[valid]).all():
        raise ValueError("Synthesis produced nonfinite scores at observed training pixels")

    scores = reconstructed.values[valid].astype(np.float64)
    actual = observed[valid].astype(np.int64)
    order = np.argsort(scores)
    scores, actual = scores[order], actual[order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    thresholds = np.r_[scores[starts], np.nextafter(scores[-1], np.inf)]
    starts = np.r_[starts, scores.size]
    positives_before = np.r_[0, np.cumsum(actual)][starts]
    tp = actual.sum() - positives_before
    fp = scores.size - starts - tp
    fn = positives_before
    denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp, denominator,
        out=np.zeros(thresholds.size, dtype=float),
        where=denominator != 0,
    )
    best_indices = np.flatnonzero(f1 == f1.max())
    best = best_indices[np.argmin(np.abs(thresholds[best_indices] - 0.5))]
    best_threshold = float(thresholds[best])
    binary = correct_binary_image(reconstructed, best_threshold).where(images.notnull())
    spatial_dims = list(reof_ds["temporal_mean"].dims)
    image_tp = ((binary == 1) & (images == 1)).sum(spatial_dims)
    image_fp = ((binary == 1) & (images == 0)).sum(spatial_dims)
    image_fn = ((binary == 0) & (images == 1)).sum(spatial_dims)
    image_denominator = 2 * image_tp + image_fp + image_fn
    return xr.Dataset(
        {
            "best_threshold": best_threshold,
            "best_f1": float(f1[best]),
            "f1": ("threshold", f1),
            "true_positives": ("threshold", tp),
            "false_positives": ("threshold", fp),
            "false_negatives": ("threshold", fn),
            "reconstructed": reconstructed,
            "binary": binary,
            "image_true_positives": image_tp,
            "image_false_positives": image_fp,
            "image_false_negatives": image_fn,
            "image_f1": (2 * image_tp / image_denominator.where(image_denominator != 0)).fillna(0),
        },
        coords={"threshold": thresholds},
        attrs={"objective": "pooled_water_f1", "classification_rule": "score >= threshold"},
    )

def plot_fier_diagnostics(
    eof_hydro: xr.Dataset,
    spatial_vmin: float = -0.01,
    spatial_vmax: float = 0.01,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (15, 10),
):
    """Plot spatial modes, normalized time series, and polynomial fits."""
    mode_dim = "mode"
    modes = eof_hydro[mode_dim].values
    n_modes = len(modes)
    if n_modes == 0:
        raise ValueError("eof_hydro must contain at least one mode")

    spatial_modes = eof_hydro["spatial_modes"]
    hydro_var = eof_hydro["hydro_var"]
    temporal_modes = eof_hydro["temporal_modes"]
    coefficients = eof_hydro["coefficients"]
    time = eof_hydro["time"]
    sites = eof_hydro["site"]
    best_r = eof_hydro["best_r"]

    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(
        3,
        n_modes,
        height_ratios=[2.4, 1.5, 1.5],
        hspace=0.55,
        wspace=0.3,
        left=0.06,
        right=0.88,
        bottom=0.09,
        top=0.96,
    )
    axes = np.empty((3, n_modes), dtype=object)
    for row in range(3):
        for column in range(n_modes):
            axes[row, column] = figure.add_subplot(grid[row, column])

    colormap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    date_locator = mpl.dates.AutoDateLocator()
    date_formatter = mpl.dates.DateFormatter("%Y-%m")

    for index, mode in enumerate(modes):
        display_mode = index + 1
        site = sites.isel({mode_dim: index}).item()
        mode_r = float(best_r.isel({mode_dim: index}).item())

        spatial_modes.isel({mode_dim: index}).fillna(0).plot(
            ax=axes[0, index],
            cmap=colormap,
            vmin=spatial_vmin,
            vmax=spatial_vmax,
            add_colorbar=False,
        )
        axes[0, index].set_title("", loc="center")
        axes[0, index].set_title("", loc="right")
        axes[0, index].set_title(
            f"RSM-{display_mode:02d}",
            loc="left",
            pad=8,
        )

        temporal_values = temporal_modes.isel({mode_dim: index})
        hydro_values = hydro_var.isel({mode_dim: index})
        temporal_normalized = stats.zscore(temporal_values, nan_policy="omit")
        hydro_normalized = stats.zscore(hydro_values, nan_policy="omit")
        axes[1, index].plot(time.values, temporal_normalized, label="RTPC")
        axes[1, index].plot(time.values, hydro_normalized, label="Hydrological variable")
        axes[1, index].set_title(
            f"RTPC-{display_mode:02d} ({site}, r={mode_r:.2f})",
            loc="left",
            pad=8,
        )
        axes[1, index].set_ylabel("Normalized value")
        axes[1, index].set_xlabel("Date", labelpad=8)
        axes[1, index].xaxis.set_major_locator(date_locator)
        axes[1, index].xaxis.set_major_formatter(date_formatter)
        axes[1, index].tick_params(axis="x", rotation=35, pad=4)
        if index == 0:
            axes[1, index].legend(loc="best")

        polynomial = np.polynomial.Polynomial(
            coefficients.isel({mode_dim: index}).values
        )
        hydro_values_np = np.asarray(hydro_values.values, dtype=float)
        temporal_values_np = np.asarray(temporal_values.values, dtype=float)
        valid = np.isfinite(hydro_values_np) & np.isfinite(temporal_values_np)
        if not np.any(valid):
            raise ValueError(f"Mode {mode} has no finite regression data")

        fitted_values = polynomial(hydro_values_np[valid])
        nse = float(he.nse(fitted_values, temporal_values_np[valid]))
        x_values = np.linspace(
            float(np.min(hydro_values_np[valid])),
            float(np.max(hydro_values_np[valid])),
            500,
        )
        axes[2, index].plot(
            hydro_values_np[valid],
            temporal_values_np[valid],
            "C0o",
            alpha=0.7,
            label="Data",
        )
        axes[2, index].plot(
            x_values,
            polynomial(x_values),
            "r--",
            label=f"Fitted model (NSE: {nse:.2f})",
        )
        axes[2, index].set_xlabel(
            f"Hydrological variable ({site})",
            labelpad=8,
        )
        axes[2, index].set_ylabel(f"RTPC-{display_mode:02d}")
        axes[2, index].legend(loc="best")

    colorbar_axis = figure.add_axes([0.90, 0.70, 0.018, 0.20])
    norm = mpl.colors.Normalize(vmin=spatial_vmin, vmax=spatial_vmax)
    figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=colormap),
        cax=colorbar_axis,
    )
    return figure, axes

def _sig_eof_test(
    da: xr.DataArray,
    option: str = "monte_carlo",
    monte_carlo_iter: int = 5,
    time_dim: str = "time",
    y_dim: str = "lat",
    x_dim: str = "lon",
):
    """
    Determine the number of significant EOF modes to retain for rotation.

    args:
        da (xr.DataArray): Input DataArray containing satellite imagery
        option (str): Significant test to use: "monte_carlo" or "north"
        monte_carlo_iter (int): Iteration time for the Monte Carlo test
        time_dim (str): Name of the time dimension. default = "time"
        y_dim (str): Name of the latitude/y dimension. default = "lat"
        x_dim (str): Name of the longitude/x dimension. default = "lon"

    output:
        sig_mode (int): Number of modes to retain for the rotation of EOF analysis results
    """
    if option not in ["monte_carlo", "north"]:
        raise ValueError("option must be 'monte_carlo' or 'north'")

    if monte_carlo_iter < 1:
        raise ValueError("monte_carlo_iter must be at least one")

    time_size = da.sizes[time_dim]
    y_size = da.sizes[y_dim]
    x_size = da.sizes[x_dim]
    space_size = y_size * x_size

    flat_da = da.values.reshape((time_size, space_size))
    aoi_mask_flat = ~(np.isnan(flat_da).all(axis=0))
    flat_da = flat_da[:, aoi_mask_flat]
    flat_da = flat_da - flat_da.mean(axis=0, keepdims=True)

    if flat_da.shape[1] == 0:
        raise ValueError("The input DataArray contains no valid spatial pixels")

    # ----- sklearn PCA (eigenvalues of real data) -----
    pca = PCA(n_components=time_size)
    pca.fit_transform(flat_da)
    real_lamb = pca.explained_variance_

    if option == "monte_carlo":
        mc_lamb = np.full((time_size, monte_carlo_iter), np.nan)
        rng = np.random.default_rng()

        # ----- Monte Carlo simulation -----
        for i in range(monte_carlo_iter):
            # ----- Randomize observations by spatially shuffling the data -----
            obs_temp = rng.permuted(flat_da, axis=1)

            pca = PCA(n_components=time_size)
            pca.fit_transform(obs_temp)
            mc_lamb[:, i] = pca.explained_variance_

        mc_lamb = np.transpose(mc_lamb)
        mean_mc_lamb = np.mean(mc_lamb, axis=0)
        sig_bool = real_lamb > mean_mc_lamb

        sig_mode = time_size
        for i in range(time_size):
            if not sig_bool[i]:
                sig_mode = i
                break

    else:
        lamb_err = real_lamb * np.sqrt(2 / time_size)
        lower_lamb = real_lamb - lamb_err
        upper_lamb = real_lamb + lamb_err

        sig_bool = lower_lamb[:-1] > upper_lamb[1:]

        sig_mode = time_size
        for i in range(len(sig_bool)):
            if not sig_bool[i]:
                sig_mode = i
                break

    return sig_mode

def _ortho_rotation(
    components, 
    method="varimax", 
    tol=1e-6, 
    max_iter=100
):
    """Return orthogonally rotated component loadings."""
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    variance = 0

    for _ in range(max_iter):
        components_rotated = np.dot(components, rotation_matrix)
        if method == "varimax":
            correction = components_rotated * np.transpose(
                (components_rotated**2).sum(axis=0) / nrow
            )
        else:
            correction = 0

        u, singular_values, vh = np.linalg.svd(
            np.dot(components.T, components_rotated**3 - correction),
            full_matrices=False,
        )
        rotation_matrix = np.dot(u, vh)
        variance_new = np.sum(singular_values)
        if variance != 0 and variance_new < variance * (1 + tol):
            break
        variance = variance_new

    return np.dot(components, rotation_matrix)

def _find_closest_valid_hydro(
    da_hydro: xr.DataArray,
    times: np.ndarray,
    time_dim: str = "time",
    site_dim: str = "site",
    warn_thrd: pd.Timedelta = pd.Timedelta(hours=5),
    error_thrd: pd.Timedelta = pd.Timedelta(days=1),
    verbose: bool = False,
) -> xr.DataArray:
    """
    For each requested timestamp, find the closest valid (non-NaN) hydrological
    observation per site.

    args:
        da_hydro: DataArray with hydrological data, with dims (site_dim, time_dim)
        times: requested timestamps (e.g. satellite acquisition times) to match against
        time_dim: name of the time dimension/coordinate. default = "time"
        site_dim: name of the site/gauge dimension/coordinate. default = "site"
        warn_thrd: warn if the closest valid observation is farther than this. default = 5 hours
        error_thrd: raise if the closest valid observation is farther than this. default = 1 day
        verbose: if True, print site, image time and matched observation time for each match. default = False

    returns:
        xr.DataArray with dims (site_dim, time_dim) aligned to `times`, holding the
        closest valid observation per site
    """
    query_times = np.asarray(times, dtype="datetime64[ns]")
    sites = da_hydro[site_dim].values

    matched = np.full((len(sites), len(query_times)), np.nan)

    for si, site in enumerate(sites):
        valid = da_hydro.sel({site_dim: site}).dropna(time_dim)
        valid_times = valid[time_dim].values
        if valid_times.size == 0:
            continue

        if valid_times.size == 1:
            nearest_idx = np.zeros(len(query_times), dtype=int)
        else:
            idx = np.clip(np.searchsorted(valid_times, query_times), 1, valid_times.size - 1)
            left_diff = np.abs(query_times - valid_times[idx - 1])
            right_diff = np.abs(valid_times[idx] - query_times)
            nearest_idx = np.where(right_diff < left_diff, idx, idx - 1)

        gaps = pd.to_timedelta(np.abs(query_times - valid_times[nearest_idx]))
        for ti, gap in enumerate(gaps):
            if verbose:
                print(
                    f"site={site}  image_time={query_times[ti]}  "
                    f"closest_obs_time={valid_times[nearest_idx[ti]]}  gap={gap}"
                )
            if gap > error_thrd:
                raise ValueError(
                    f"Closest valid obs for {site} is {gap} away "
                    f"(at {valid_times[nearest_idx[ti]]}). Requested timestamp: {query_times[ti]}"
                )
            if gap > warn_thrd:
                warnings.warn(
                    f"Closest valid obs for {site} is {gap} away "
                    f"(at {valid_times[nearest_idx[ti]]}). Requested timestamp: {query_times[ti]}"
                )
        matched[si] = valid.values[nearest_idx]

    return xr.DataArray(
        matched,
        dims=(site_dim, time_dim),
        coords={site_dim: sites, time_dim: query_times},
    )

def download_gfm(
    bbox: tuple[float, float, float, float],
    bands: Sequence[str],
    start_time: datetime,
    end_time: datetime,
    *,
    crs: str | pyproj.CRS | None = None,
    resolution: float | None = None,
    groupby: str = "solar_day",
) -> xr.Dataset:
    """Get GFM data for the specified bounding box and time range.
    
    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box in the format (minx, miny, maxx, maxy).
    bands : Sequence[str]
        List of GFM data bands to retrieve.
    start_time : datetime
        Start of the time range for the search.
    end_time : datetime
        End of the time range for the search.
    crs : str | pyproj.CRS | None, optional
        Coordinate reference system for the output data. If None, uses native CRS.
    resolution : float | None, optional
        Resolution for the output data. Must be provided together with `crs`. 
        CRS and Resolution must match. If None, uses native resolution (20 meters).
    groupby : str, optional
        Grouping method for the loaded data, default is "solar_day".

    Returns
    -------
    xr.Dataset
        The loaded GFM data as an xarray Dataset.

    """
    if (crs is None) != (resolution is None):
        raise ValueError("Provide both crs and resolution, or neither. " \
        "Note that crs and resolution should match, e.g., crs='EPSG:4326' with resolution=20/111_000 " \
        "considering 1 degree ≈ 111 km and the raw GFM data resolution is 20 meters.")

    api_url = "https://stac.eodc.eu/api/v1"
    eodc_catalog = Client.open(api_url)
    collection_id = "GFM"

    # Perform a search in the catalog with the specified parameters
    items = list(eodc_catalog.search(
        collections=[collection_id], 
        bbox=bbox, 
        datetime=(start_time, end_time),
    ).items())

    # Print the number of items found for debugging purposes
    print(f"Found {len(items)} GFM items.")

    if not items:
        raise ValueError("No GFM items found for the bbox and time range.")

    for item in items:
        data_keys = {
            key for key, asset in item.assets.items()
            if "data" in (asset.roles or [])
        }
        missing = set(bands) - data_keys
        if missing:
            raise ValueError(
                f"Unavailable data bands for {item.id}: {sorted(missing)}. "
                f"Available data keys: {sorted(data_keys)}"
            )

    if crs is None:
        crs = pyproj.CRS.from_wkt(items[0].properties["proj:wkt2"])
        resolution = items[0].properties["gsd"]

    data = odc_stac.load(
        items, 
        bbox=bbox, 
        bands=bands, 
        crs=crs, 
        resolution=resolution,
        groupby=groupby,
    )

    # Check if all pixels contain no data (255) and drop the corresponding time slices
    for band in bands:
        spatial_dims = [dim for dim in data[band].dims if dim != "time"]
        no_data = (data[band] == 255).all(dim=spatial_dims).compute()

        if no_data.any().item():
            no_data_dates = data.time.where(no_data, drop=True)
            print(
                f"All pixels are no data for band {band} on dates: "
                f"{no_data_dates.values}. Dropping these time slices."
            )
            data = data.isel(time=~no_data)
    
    return data