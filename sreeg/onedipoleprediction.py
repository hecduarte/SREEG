import numpy as np
from .equipotential import equipotentials
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from sreeg.electrodes import evaluate_dipole_at_electrodes

def dipole_orientation_by_nodal_plane(mesh, *, n_div_list=(3, 4, 5), min_pts=12, min_len=0.05):
    """
    Estimate dipole orientation from the central nodal plane (equipotential = 0),
    using multiple subdivisions and filtering out short/isolated polylines.
    """

    all_pts_xy = []

    # --- Collect nodal points across multiple resolutions ---
    for n_div in n_div_list:
        eq = equipotentials(mesh, n_div=n_div, min_len=min_len)
        nodal_lines = eq["polylines"].get(0.0, [])
        for poly in nodal_lines:
            if len(poly) >= 3:  # keep only non-trivial segments
                all_pts_xy.append(poly)

    if not all_pts_xy:
        return {
            "direction": np.full(3, np.nan),
            "score": 0.0,
            "method": "dipole_orientation_by_nodal_plane/fit-curve-robust",
            "details": {"num_points": 0, "plane_normal_raw": np.full(3, np.nan)},
        }

    pts_xy = np.vstack(all_pts_xy)

    # --- Map back to 3D sphere via nearest vertices ---
    verts_xy = mesh["verts_xy"]
    verts = mesh["verts"]

    tree = cKDTree(verts_xy)
    _, idx = tree.query(pts_xy, k=1)
    pts3d = verts[idx]

    if len(pts3d) < min_pts:
        return {
            "direction": np.full(3, np.nan),
            "score": float(len(pts3d)),
            "method": "dipole_orientation_by_nodal_plane/fit-curve-robust",
            "details": {"num_points": len(pts3d), "plane_normal_raw": np.full(3, np.nan)},
        }

    # --- Robust plane fit (unweighted) ---
    centroid = pts3d.mean(axis=0)
    X = pts3d - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1, :]
    normal /= np.linalg.norm(normal) + 1e-12

    # --- Ensure orientation consistency with electrode field ---
    ref_vec = pts3d.mean(axis=0)  # proxy reference from geometry
    if np.dot(normal, ref_vec) < 0:
        normal = -normal

    return {
        "direction": normal,
        "score": float(len(pts3d)),
        "method": "dipole_orientation_by_nodal_plane/fit-curve-robust",
        "details": {"num_points": len(pts3d), "plane_normal_raw": normal},
    }


def dipole_position_by_planes(
    positions: np.ndarray,
    values: np.ndarray,
    orientation: np.ndarray,
    *,
    n_planes: int = 7,
    band_halfwidth: float = 0.01,   # [m] half-thickness for each parallel band
    loss: str = "zMSE",             # {"MSE","zMSE"}
    sigma: float = 0.33
) -> dict:
    """
    Estimate dipole position using parallel planes orthogonal to the nodal axis (orientation):
      1) Slice electrodes into n_planes bands along 'orientation' (always uses electrodes).
      2) In each band, compute a weighted centroid (|V| weights) as local source proxy.
      3) Fit a 3D axis to those centroids via PCA.
      4) Search along that axis for the position that best matches the electrode pattern,
         fixing the orientation and fitting only a scale α (amplitude) at each test point.
         Comparison metric selectable via 'loss' = {"MSE","zMSE"}.

    Returns
    -------
    {
      "position": (3,),        # best position on the fitted axis
      "axis_point": (3,),      # PCA axis point (centroid)
      "axis_dir": (3,),        # PCA axis direction (unit)
      "local_points": (K,3),   # centroids per band used
      "alpha": float,          # scale at best position
      "loss": float,           # loss at best position
      "method": "dipole_position_by_planes",
      "details": {...}
    }
    """
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float)
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    def _zscore(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        mu, sd = x.mean(), x.std()
        return (x - mu) / (sd + 1e-12)

    positions = np.asarray(positions, float)
    values    = np.asarray(values, float)
    u_axis    = _unit(np.asarray(orientation, float))

    # --- 1) Parallel bands (always from electrodes) ---
    s = positions @ u_axis  # signed distance along axis
    s_min, s_max = float(s.min()), float(s.max())
    # Center band centers around the mid-span for symmetry
    centers = np.linspace(s_min, s_max, int(max(3, n_planes)))
    
    local_pts = []
    for c in centers:
        idx = np.where(np.abs(s - c) <= float(band_halfwidth))[0]
        if idx.size < 3:
            continue
        w = np.abs(values[idx])
        if w.sum() <= 0:
            continue
        p = (positions[idx] * w[:, None]).sum(axis=0) / (w.sum() + 1e-12)
        local_pts.append(p)
    local_pts = np.array(local_pts, float)

    if local_pts.shape[0] < 2:
        # Fallback: use strongest electrodes to build a crude axis
        j = int(np.argmax(np.abs(values)))
        local_pts = positions[[j], :]
        # Pick a second point slightly shifted along orientation
        local_pts = np.vstack([local_pts, local_pts[0] + 0.01 * u_axis])

    # --- 2) PCA axis through local points ---
    cen = local_pts.mean(axis=0)
    Xc  = local_pts - cen
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    axis_dir = _unit(Vt[0])

    # Ensure axis_dir is aligned with nodal orientation (same/antipodal)
    if np.dot(axis_dir, u_axis) < 0:
        axis_dir = -axis_dir

    # --- 3) 1D search along axis using electrode fit (always uses electrodes) ---
    # Define a reasonable search interval from projections of electrodes and local points
    t_proj_e = (positions - cen) @ axis_dir
    t_proj_lp = (local_pts - cen) @ axis_dir
    t_lo = float(min(t_proj_e.min(), t_proj_lp.min()))
    t_hi = float(max(t_proj_e.max(), t_proj_lp.max()))
    # pad 10% on each side
    pad = 0.1 * (t_hi - t_lo + 1e-12)
    t_lo -= pad; t_hi += pad

    v_meas = values.astype(float)

    def _fit_alpha_and_loss_at(pos: np.ndarray) -> tuple[float, float]:
        v_pred = evaluate_dipole_at_electrodes(
            {"location": pos, "orientation": u_axis},
            positions,
            sigma=sigma
        ).astype(float)
        # Best α in least squares
        denom = float(np.dot(v_pred, v_pred)) + 1e-12
        alpha = float(np.dot(v_meas, v_pred) / denom)
        if loss.upper() == "MSE":
            L = float(np.mean((v_meas - alpha * v_pred) ** 2))
        elif loss.upper() == "ZMMSE" or loss.upper() == "ZMSE":  # tolerate typo
            L = float(np.mean((_zscore(v_meas) - _zscore(alpha * v_pred)) ** 2))
        else:  # default robust choice
            L = float(np.mean((_zscore(v_meas) - _zscore(alpha * v_pred)) ** 2))
        return alpha, L

    # Coarse grid + local refine (1D)
    T = np.linspace(t_lo, t_hi, 41)
    losses = []
    alphas = []
    for t in T:
        a, L = _fit_alpha_and_loss_at(cen + t * axis_dir)
        alphas.append(a); losses.append(L)
    losses = np.asarray(losses); alphas = np.asarray(alphas)
    k0 = int(np.argmin(losses))
    t0 = T[k0]

    # Local Nelder–Mead around t0
    def _loss_scalar(tt: np.ndarray) -> float:
        _, L = _fit_alpha_and_loss_at(cen + float(tt[0]) * axis_dir)
        return L

    res = minimize(_loss_scalar, x0=np.array([t0], float), method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-8})

    t_star = float(res.x[0])
    pos_star = cen + t_star * axis_dir
    alpha_star, loss_star = _fit_alpha_and_loss_at(pos_star)

    return {
        "position": pos_star.astype(float),
        "axis_point": cen.astype(float),
        "axis_dir": axis_dir.astype(float),
        "local_points": local_pts.astype(float),
        "alpha": float(alpha_star),
        "loss": float(loss_star),
        "method": "dipole_position_by_planes",
        "details": {
            "n_local_points": int(local_pts.shape[0]),
            "search_interval": (float(t_lo), float(t_hi)),
            "init_t": float(t0),
            "init_loss": float(losses[k0]),
            "loss_metric": "MSE" if loss.upper()=="MSE" else "zMSE",
            "sigma": float(sigma),
        }
    }


def dipole_orientation_by_fast_grid(
    positions: np.ndarray,
    values: np.ndarray,
    pos_est: np.ndarray,
    *,
    n_orient: int = 128,
    sigma: float = 0.33,
    loss: str = "zMSE",
) -> dict:
    """
    Fast orientation search on a fixed dipole position using a Fibonacci sphere grid.

    Parameters
    ----------
    positions : ndarray, shape (N,3)
        Electrode positions.
    values : ndarray, shape (N,)
        Measured potentials at electrodes.
    pos_est : ndarray, shape (3,)
        Estimated dipole position.
    n_orient : int, optional
        Number of orientation candidates (default 128).
    sigma : float, optional
        Spatial spread parameter for forward model.
    loss : {"MSE","zMSE"}, optional
        Loss function for fit.

    Returns
    -------
    result : dict
        {
          "direction": (3,),  # best orientation (unit vector)
          "alpha": float,     # scaling factor
          "loss": float,      # loss at best orientation
          "method": "dipole_orientation_by_fast_grid",
          "details": {
              "matrix": ndarray (M, 6),
              "matrix_columns": ("ox","oy","oz","loss","alpha","idx")
          }
        }
    """
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    def _zscore(x: np.ndarray) -> np.ndarray:
        mu, sd = x.mean(), x.std()
        return (x - mu) / (sd + 1e-12)

    # --- Fibonacci sphere sampling ---
    def _fibonacci_sphere(n: int) -> np.ndarray:
        k = np.arange(n, dtype=float) + 0.5
        phi = 2.0 * np.pi * k / ((1 + np.sqrt(5)) / 2.0)  # golden angle
        z = 1.0 - 2.0 * k / n
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.column_stack([x, y, z])

    orientations = _fibonacci_sphere(n_orient)

    values = values.astype(float)
    cand_rows = []

    best_loss, best_u, best_alpha = np.inf, None, 0.0

    for idx, u in enumerate(orientations):
        u_hat = _normalize(u)
        v_pred = evaluate_dipole_at_electrodes(
            {"location": pos_est, "orientation": u_hat},
            positions,
            sigma=sigma
        ).astype(float)

        denom = float(np.dot(v_pred, v_pred)) + 1e-12
        alpha = float(np.dot(values, v_pred) / denom)

        if loss.upper() == "MSE":
            L = float(np.mean((values - alpha * v_pred) ** 2))
        else:  # default zMSE
            L = float(np.mean((_zscore(values) - _zscore(alpha * v_pred)) ** 2))

        cand_rows.append([u_hat[0], u_hat[1], u_hat[2], L, alpha, idx])

        if L < best_loss:
            best_loss, best_u, best_alpha = L, u_hat, alpha

    M = np.asarray(cand_rows, dtype=float)
    mat_cols = ("ox", "oy", "oz", "loss", "alpha", "idx")

    return {
        "direction": best_u,
        "alpha": best_alpha,
        "loss": best_loss,
        "method": "dipole_orientation_by_fast_grid",
        "details": {
            "matrix": M,
            "matrix_columns": mat_cols,
        },
    }


def dipole_position_by_fast_grid(
    positions: np.ndarray,
    values: np.ndarray,
    orientation: np.ndarray,
    pos_est: np.ndarray,
    *,
    # --- coarse pre-pass (slightly wider, low cost) ---
    coarse_line_span: float = 0.05,
    coarse_n_line: int = 9,
    coarse_radial_offsets: tuple[float, ...] = (0.0, 0.006, -0.006),
    # --- main (focused) pass ---
    axis_dir: np.ndarray | None = None,
    line_span: float = 0.02,
    n_line: int = 11,
    radial_offsets: tuple[float, ...] = (0.0, 0.003, -0.003),
    # optional fine refinement around the best of the focused pass
    two_pass: bool = False,
    shrink: float = 0.5,
    sigma: float = 0.33,
    loss: str = "zMSE",
) -> dict:
    """
    Fast position refinement with a lightweight coarse→focused strategy.
    1) Coarse pre-pass: wider 1D line around pos_est with small lateral offsets.
    2) Focused pass: stencil around the coarse best (axis_dir re/estimated).
    3) Optional fine pass: shrunken stencil around the focused best.

    Returns
    -------
    dict
        {
          "position": (3,),
          "alpha": float,
          "loss": float,
          "axis_dir": (3,),
          "method": "dipole_position_by_fast_grid",
          "details": {
            "matrix": (M,8),
            "matrix_columns": ("x","y","z","loss","alpha","t","dx","dy"),
            "coarse": {...}, "focused": {...}, "fine": {... or None},
            "sigma": float, "loss_metric": "MSE" | "zMSE"
          }
        }
    """
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float)
        n = np.linalg.norm(v)
        return v if n == 0.0 else v / n

    def _z(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return (x - x.mean()) / (x.std() + 1e-12)

    positions = np.asarray(positions, float)
    values    = np.asarray(values,    float)
    u_ori     = _unit(orientation)
    p0        = np.asarray(pos_est,   float)

    # axis estimation helper (weighted PCA on |V|)
    def _estimate_axis_dir() -> np.ndarray:
        w = np.abs(values)
        w_sum = float(w.sum()) + 1e-12
        cen = (positions * (w[:, None] / w_sum)).sum(axis=0)
        X = positions - cen
        Xw = X * np.sqrt((w[:, None] / w_sum))
        _, _, Vt = np.linalg.svd(Xw, full_matrices=False)
        return Vt[0]

    # orthonormal frame around an axis
    def _orthonormal_frame(u_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base = np.array([0.0, 0.0, 1.0]) if abs(np.dot(u_axis, [0, 0, 1])) < 0.9 else np.array([1.0, 0.0, 0.0])
        u_perp1 = _unit(np.cross(u_axis, base))
        u_perp2 = _unit(np.cross(u_axis, u_perp1))
        return u_perp1, u_perp2

    # evaluate one stencil
    def _evaluate_stencil(center: np.ndarray, u_axis: np.ndarray, span: float, npts: int, roff: tuple[float, ...]):
        u_perp1, u_perp2 = _orthonormal_frame(u_axis)
        ts = np.linspace(-float(span), float(span), int(max(3, npts)))
        rows = []
        best = (np.inf, center, 0.0)  # (loss, pos, alpha)
        for t in ts:
            base_pt = center + t * u_axis
            for dx in roff:
                for dy in roff:
                    pos = base_pt + dx * u_perp1 + dy * u_perp2
                    v_pred = evaluate_dipole_at_electrodes(
                        {"location": pos, "orientation": u_ori},
                        positions,
                        sigma=sigma
                    ).astype(float)

                    denom = float(np.dot(v_pred, v_pred)) + 1e-12
                    alpha = float(np.dot(values, v_pred) / denom)

                    if loss.upper() == "MSE":
                        L = float(np.mean((values - alpha * v_pred) ** 2))
                    else:  # zMSE
                        L = float(np.mean((_z(values) - _z(alpha * v_pred)) ** 2))

                    rows.append([pos[0], pos[1], pos[2], L, alpha, t, dx, dy])
                    if L < best[0]:
                        best = (L, pos, alpha)
        return np.asarray(rows, float), best

    matrices = []
    info = {"coarse": None, "focused": None, "fine": None}

    # --- (1) Coarse pre-pass (axis from PCA; wider span, low npts) ---
    u_axis_coarse = _unit(_estimate_axis_dir())
    M_coarse, best_coarse = _evaluate_stencil(
        center=p0,
        u_axis=u_axis_coarse,
        span=coarse_line_span,
        npts=coarse_n_line,
        roff=coarse_radial_offsets
    )
    matrices.append(M_coarse)
    info["coarse"] = {
        "axis_dir": u_axis_coarse.astype(float),
        "line_span": float(coarse_line_span),
        "n_line": int(max(3, coarse_n_line)),
        "radial_offsets": tuple(float(r) for r in coarse_radial_offsets),
        "best_loss": float(best_coarse[0]),
    }

    # --- (2) Focused pass (axis: provided or re/estimated; centered at coarse best) ---
    u_axis_focused = _unit(axis_dir) if axis_dir is not None else _unit(_estimate_axis_dir())
    M_focused, best_focused = _evaluate_stencil(
        center=best_coarse[1],
        u_axis=u_axis_focused,
        span=line_span,
        npts=n_line,
        roff=radial_offsets
    )
    matrices.append(M_focused)
    info["focused"] = {
        "axis_dir": u_axis_focused.astype(float),
        "line_span": float(line_span),
        "n_line": int(max(3, n_line)),
        "radial_offsets": tuple(float(r) for r in radial_offsets),
        "best_loss": float(best_focused[0]),
    }

    best_all = best_focused
    u_axis_final = u_axis_focused

    # --- (3) Optional fine pass (shrunk around focused best) ---
    if two_pass:
        span2 = line_span * float(shrink)
        n2 = max(3, int(np.ceil(n_line * float(shrink))))
        roff2 = tuple(float(r) * float(shrink) for r in radial_offsets)
        M_fine, best_fine = _evaluate_stencil(
            center=best_focused[1],
            u_axis=u_axis_focused,
            span=span2,
            npts=n2,
            roff=roff2
        )
        matrices.append(M_fine)
        info["fine"] = {
            "axis_dir": u_axis_focused.astype(float),
            "line_span": float(span2),
            "n_line": int(n2),
            "radial_offsets": roff2,
            "best_loss": float(best_fine[0]),
        }
        if best_fine[0] < best_all[0]:
            best_all = best_fine

    M = np.vstack(matrices) if len(matrices) else np.zeros((0, 8), float)

    return {
        "position": best_all[1].astype(float),
        "alpha": float(best_all[2]),
        "loss": float(best_all[0]),
        "axis_dir": u_axis_final.astype(float),
        "method": "dipole_position_by_fast_grid",
        "details": {
            "matrix": M,
            "matrix_columns": ("x", "y", "z", "loss", "alpha", "t", "dx", "dy"),
            "coarse": info["coarse"],
            "focused": info["focused"],
            "fine": info["fine"],
            "sigma": float(sigma),
            "loss_metric": "MSE" if loss.upper() == "MSE" else "zMSE",
        },
    }


def dipole_amplitude_by_fit(
    positions: np.ndarray,
    values: np.ndarray,
    location: np.ndarray,
    orientation: np.ndarray,
    *,
    sigma: float = 0.33,
    loss: str = "zMSE",
) -> dict:
    """
    Estimate dipole amplitude (scaling factor) given a fixed position and orientation.
    The forward model is evaluated for a unit dipole, and the best scale α is found
    by least squares fit against electrode data.

    Parameters
    ----------
    positions : ndarray, shape (N,3)
        Electrode positions.
    values : ndarray, shape (N,)
        Measured electrode potentials.
    location : ndarray, shape (3,)
        Dipole position.
    orientation : ndarray, shape (3,)
        Dipole orientation (unit vector).
    sigma : float, optional
        Spatial spread parameter for forward model.
    loss : {"MSE","zMSE"}, optional
        Loss function for fit.

    Returns
    -------
    result : dict
        {
          "alpha": float,     # estimated amplitude
          "loss": float,      # loss at best fit
          "method": "dipole_amplitude_by_fit",
          "details": {
              "v_pred": ndarray,   # predicted electrode pattern (unit dipole)
              "v_meas": ndarray    # measured electrode pattern
          }
        }
    """
    def _z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-12)

    values = np.asarray(values, float)

    # Predicted pattern for unit amplitude dipole
    v_pred = evaluate_dipole_at_electrodes(
        {"location": np.asarray(location, float),
         "orientation": np.asarray(orientation, float)},
        np.asarray(positions, float),
        sigma=sigma
    ).astype(float)

    # Best alpha in least squares
    denom = float(np.dot(v_pred, v_pred)) + 1e-12
    alpha = float(np.dot(values, v_pred) / denom)
    if alpha < 0:
        alpha = -alpha
        orientation = -orientation

    # Compute loss
    if loss.upper() == "MSE":
        L = float(np.mean((values - alpha * v_pred) ** 2))
    else:  # default zMSE
        L = float(np.mean((_z(values) - _z(alpha * v_pred)) ** 2))

    return {
        "alpha": alpha,
        "loss": L,
        "method": "dipole_amplitude_by_fit",
        "details": {
            "v_pred": v_pred,
            "v_meas": values
        }
    }


def onedipole_frequency_by_fft(values: np.ndarray, fs: float, ch: int = 0) -> dict:
    """
    Estimate the dominant frequency from one electrode channel,
    only if temporal information is available.

    Parameters
    ----------
    values : ndarray, shape (N, T) or (N,)
        Electrode activity matrix (N channels × T time samples),
        or static frame (N,).
    fs : float
        Sampling frequency in Hz.
    ch : int, optional
        Channel index to analyze (default=0).

    Returns
    -------
    result : dict
        {
          "frequency": float or nan,  # dominant frequency [Hz] or nan if not available
          "channel": int,             # electrode index used
          "method": "onedipole_frequency_by_fft",
          "status": str               # "ok" if computed, "no-temporal-data" otherwise
        }
    """
    values = np.asarray(values, float)

    # --- Case 1: no temporal dimension ---
    if values.ndim == 1:
        return {
            "frequency": float("nan"),
            "channel": ch,
            "method": "onedipole_frequency_by_fft",
            "status": "no-temporal-data"
        }

    # --- Case 2: temporal dimension available ---
    if values.ndim == 2:
        if not (0 <= ch < values.shape[0]):
            raise IndexError(f"Channel index {ch} out of range (0..{values.shape[0]-1})")

        signal = values[ch]
        T = signal.shape[0]

        freqs = np.fft.rfftfreq(T, d=1.0/fs)
        psd = np.abs(np.fft.rfft(signal))**2

        if psd[1:].size == 0:
            return {
                "frequency": float("nan"),
                "channel": ch,
                "method": "onedipole_frequency_by_fft",
                "status": "no-peak-found"
            }

        k = int(np.argmax(psd[1:])) + 1  # skip DC
        return {
            "frequency": float(freqs[k]),
            "channel": ch,
            "method": "onedipole_frequency_by_fft",
            "status": "ok"
        }

    # --- Case 3: unexpected dimensions ---
    raise ValueError("values must be (N,) or (N,T)")
