import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import hydroeval as he

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
    hydro_common = _find_closest_valid_hydro(da_hydro, time_tpc.values, time_dim=time_dim, site_dim=site_dim)
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
