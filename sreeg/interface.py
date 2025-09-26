class BrainBox:
    """Encapsulates a 3D brain box visualization."""

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax

    def show(self):
        """Display the brain box figure."""
        import matplotlib.pyplot as plt
        self.fig.tight_layout()
        plt.show()


def brain_box(
    dipoles=None,
    size=0.20,
    theta=22,
    phi=30,
    show=False
):
    """
    Create a 3D brain box. Optionally plot dipoles (with optional per-dipole colors).

    Parameters
    ----------
    dipoles : list or None
        Iterable of dipole dicts or (dipole, color) pairs.
        Each dipole dict must have at least: 'location' (3,), 'orientation' (3,).
        If a dipole dict includes 'size', the box size will adopt the max over them.
    size : float
        Default head size [m] used if dipoles do not provide one (fallback).
    theta : float
        Elevation angle (degrees).
    phi : float
        Azimuth angle (degrees).
    show : bool
        If True, display immediately and return None.

    Returns
    -------
    BrainBox or None
        BrainBox object (with fig, ax, .show()) or None if show=True.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sreeg.simulation import plot_dipole_3d

    # Resolve effective size: start from argument, upgrade with any dipole['size']
    eff_size = float(size)
    dp_list, color_list = [], []
    if dipoles is not None:
        for item in dipoles:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                d = item[0]
                c = item[1] if len(item) >= 2 else None
            else:
                d, c = item, None
            dp_list.append(d)
            color_list.append(c)
            if isinstance(d, dict) and ("size" in d):
                try:
                    eff_size = max(eff_size, float(d["size"]))
                except Exception:
                    pass

    # Figure and axis
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=theta, azim=phi)

    # Delegate drawing (background + dipoles + projections) to plot_dipole_3d
    dipoles_with_colors = []
    palette = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    for idx, d in enumerate(dp_list):
        if not isinstance(d, dict):
            continue
        col = color_list[idx] if color_list[idx] is not None else palette[idx % len(palette)]
        dipoles_with_colors.append({**d, "color": col})

    plot_dipole_3d(
        dipoles=dipoles_with_colors,
        background=True,
        realism='vacuum',
        size=eff_size,
        figsize=(6, 6),
        show=False,
        ax=ax
    )

    box = BrainBox(fig, ax)
    if show:
        box.show()
        return None
    return box

# --- Composite dipole support (opt-in, non-breaking) ------------------------

class Dipole(dict):
    """
    Dict-compatible dipole with '+' operator to compose multi-dipole signals.
    Immutable-by-convention: do not mutate fields after creation.
    """
    def __add__(self, other):
        if isinstance(other, Dipole):
            return Signal([self, other])
        if isinstance(other, Signal):
            return Signal([self] + other.dipoles)
        raise TypeError("Can only add Dipole or Signal")

class Signal:
    """
    Composite signal: ordered list of Dipole objects.
    Supports associative '+' composition and iteration.
    """
    __slots__ = ("dipoles",)

    def __init__(self, dipoles):
        # Minimal validation: non-empty sequence
        self.dipoles = list(dipoles)
        if not self.dipoles:
            raise ValueError("Signal requires at least one Dipole.")
        # Temporal compatibility (length only; dt alignment is handled downstream)
        lengths = { (len(d.get("signal")) if "signal" in d and d["signal"] is not None else None)
                    for d in self.dipoles }
        known = {L for L in lengths if L is not None}
        if len(known) > 1:
            raise ValueError("All dipoles must share the same time length when temporal.")

    def __add__(self, other):
        if isinstance(other, Dipole):
            return Signal(self.dipoles + [other])
        if isinstance(other, Signal):
            return Signal(self.dipoles + other.dipoles)
        raise TypeError("Can only add Dipole or Signal")

_ENABLE_SIGNAL_OP = False
def enable_signal_operator():
    """
    Opt-in: make `dipole_in_brain` return a Dipole (sum-enabled).
    Without calling this, the API remains unchanged.
    """
    global _ENABLE_SIGNAL_OP
    _ENABLE_SIGNAL_OP = True
    

def dipole_in_brain(
    location="random",
    orientation="random",
    frequency=10.0,
    amplitude=1.0,
    phase=0.0,
    duration=1.0,
    sfreq=1000,
    size=0.20,
    arbitrary=False
):
    """
    Generate a synthetic dipole inside the brain volume.

    Parameters
    ----------
    location : array-like (3,) or "random"
        Dipole location in meters. If "random", a random location inside the head is used.
    orientation : array-like (3,) or "random"
        Dipole orientation vector. If "random", a random unit vector is used.
    frequency : float
        Dipole oscillation frequency [Hz].
    amplitude : float
        Signal amplitude (default 1.0).
    phase : float
        Signal phase [radians].
    duration : float
        Duration of the signal [s].
    sfreq : float
        Sampling frequency [Hz].
    size : float
        Physical head size [m] (default 0.20).
    arbitrary : bool
        If True, new random location/orientation are drawn on every call.

    Returns
    -------
    dict
        Dipole dictionary with the following keys:
        - 'time' : ndarray, time vector [s]
        - 'signal' : ndarray, dipole time series (official field)
        - 'amplitude' : ndarray, legacy alias for 'signal' (kept for backward compatibility)
        - 'location' : ndarray(3,), dipole position [m]
        - 'orientation' : ndarray(3,), unit orientation vector
        - 'frequency' : float, oscillation frequency [Hz]
        - 'phase' : float, initial phase [rad]
        - 'size' : float, effective head size [m]
    """
    import numpy as np

    # --- Time vector ---
    time = np.arange(0, duration, 1.0 / sfreq)

    # --- Random generation if requested ---
    radius = size / 2.0
    if isinstance(location, str) and location == "random":
        loc = np.random.uniform(-radius, radius, 3)
        while np.linalg.norm(loc) > radius:
            loc = np.random.uniform(-radius, radius, 3)
    else:
        loc = np.array(location, dtype=float)
        if np.linalg.norm(loc) > radius:
            loc = loc / np.linalg.norm(loc) * radius
            import warnings
            warnings.warn(
                f"[dipole_in_brain] Location adjusted to fit inside head radius {radius:.3f} m.",
                UserWarning
            )

    if isinstance(orientation, str) and orientation == "random":
        vec = np.random.normal(size=3)
    else:
        vec = np.array(orientation, dtype=float)
    vec = vec / (np.linalg.norm(vec) + 1e-12)

    # --- Signal ---
    signal = amplitude * np.sin(2 * np.pi * frequency * time + phase)

    return Dipole({
        "time": time,
        "signal": signal,      # time series (official)
        "amplitude": signal,   # legacy alias for backward-compat (to be deprecated)
        "location": loc,
        "orientation": vec,
        "frequency": float(frequency),
        "phase": float(phase),
        "size": float(size),
    }) if _ENABLE_SIGNAL_OP else {
        "time": time,
        "signal": signal,      # time series (official)
        "amplitude": signal,   # legacy alias for backward-compat (to be deprecated)
        "location": loc,
        "orientation": vec,
        "frequency": float(frequency),
        "phase": float(phase),
        "size": float(size),
    }


def dipole_to_electrodes(dipole, montage="biosemi32", sigma=0.33):
    """
    Project a dipole source onto EEG electrodes and return activity with metadata.

    Parameters
    ----------
    dipole : dict
        Dictionary with dipole parameters:
        - 'location' : array-like, shape (3,)
            Dipole position [m].
        - 'orientation' : array-like, shape (3,)
            Dipole orientation (will be normalized).
        - 'amplitude' : float or array, optional
            If float, scale factor for the spatial pattern.
            If array, time course of the dipole (shape (Nt,)).
            If missing, unit amplitude is assumed.
    montage : str, optional
        Electrode montage name (e.g., "biosemi32", "biosemi64", "standard_1020").
        Default is "biosemi32".
    sigma : float, optional
        Conductivity of the medium [S/m]. Default is 0.33.

    Returns
    -------
    ElectrodeActivity
        Array-like object with potentials at each electrode [V]:
        - shape (N,) if amplitude is scalar or missing.
        - shape (N, Nt) if amplitude is temporal array.
        Attributes:
            .montage   -> str, montage name
            .electrodes -> list of str, electrode names
    """
    import numpy as np
    from sreeg.electrodes import get_electrode_positions

    class ElectrodeActivity(np.ndarray):
        """Array subclass carrying montage metadata."""

        def __new__(cls, input_array, montage, electrodes):
            obj = np.asarray(input_array).view(cls)
            obj.montage = montage
            obj.electrodes = electrodes
            return obj

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self.montage = getattr(obj, "montage", None)
            self.electrodes = getattr(obj, "electrodes", None)

    # --- Composite Signal support (opt-in, non-breaking) --------------------
    # If a composite Signal is provided, project each Dipole and sum channel-wise.
    if isinstance(dipole, Signal):
        acc = None
        montage_out = None
        electrodes_out = None
        time_out = None
        freq_out = None

        for d in dipole.dipoles:
            cur = dipole_to_electrodes(d, montage=montage, sigma=sigma)  # recursion (dict/Dipole path)
            arr = np.asarray(cur, dtype=float)
            if acc is None:
                acc = arr.copy()
                montage_out = getattr(cur, "montage", montage)
                electrodes_out = getattr(cur, "electrodes", None)
                time_out = getattr(cur, "time", getattr(d, "time", None))
                freq_out = getattr(cur, "frequency", None)
            else:
                if acc.shape != arr.shape:
                    raise ValueError(
                        f"[dipole_to_electrodes] Shape mismatch while summing composite signal: "
                        f"expected {acc.shape}, got {arr.shape}."
                    )
                acc += arr

        ea = ElectrodeActivity(acc, montage_out, electrodes_out)
        if time_out is not None:
            setattr(ea, "time", time_out)
        if freq_out is not None:
            setattr(ea, "frequency", freq_out)
        return ea

    # Transparently accept Dipole by unwrapping to dict (legacy path)
    if isinstance(dipole, Dipole):
        dipole = dict(dipole)

    # Electrode positions and names from montage
    pos, names = get_electrode_positions(montage)

    # Dipole geometry
    r0 = np.asarray(dipole["location"], dtype=float)
    p = np.asarray(dipole["orientation"], dtype=float)
    p = p / np.linalg.norm(p)

    # Spatial projection
    diff = pos - r0
    norm = np.linalg.norm(diff, axis=1)
    norm[norm < 1e-9] = np.inf  # avoid singularity
    dot = diff @ p
    G = (1.0 / (4 * np.pi * sigma)) * (dot / norm**3)

    # Signal / amplitude handling
    if "signal" in dipole:  # preferred (time series)
        sig = np.asarray(dipole["signal"], float)
        activity = G[:, None] * sig[None, :]
    else:
        amp = dipole.get("amplitude", 1.0)
        if np.isscalar(amp):
            activity = G * float(amp)
        else:
            amp = np.asarray(amp, float)
            activity = G[:, None] * amp[None, :]

    ea = ElectrodeActivity(activity, montage, names)
    ea.frequency = dipole.get("frequency", None)
    return ea

class BrainActivity:
    """Encapsulates scalp activity visualization and data for prediction."""

    def __init__(self, mesh, positions, electrodes, values, fig, ax):
        self.mesh = mesh              # dict (equipotential mesh info)
        self.positions = positions    # ndarray (N,3)
        self.electrodes = electrodes  # list of str
        self.values = values          # ndarray (N,)
        self.fig = fig                # matplotlib Figure
        self.ax = ax                  # matplotlib Axes

    def show(self):
        """Re-display the scalp activity figure (optional)."""
        import matplotlib.pyplot as plt
        self.fig.tight_layout()
        plt.show()


def activity_brain(activity, montage=None, when=None, reducer="mean",
                   cmap="RdBu_r", subdiv=3, show=False):
    """
    Compute and plot brain activity map from electrode data.

    Parameters
    ----------
    activity : ndarray or dict-like
        Electrode potentials: shape (N,) or (N, Nt).
        If dict-like from dipole_to_electrodes(), must include 'electrodes'.
    montage : str, optional
        Electrode montage name (required if activity has no metadata).
    when : int or float, optional
        Temporal index or time (if activity has time).
    reducer : str, optional
        Reduction method if activity is temporal. Default "mean".
    cmap : str, optional
        Colormap for topomap. Default "RdBu_r".
    subdiv : int, optional
        Subdivision level for interpolation. Default 3.
    show : bool, optional
        If True, display immediately and return None.

    Returns
    -------
    BrainActivity or None
        Object containing mesh, positions, electrodes, values, fig, ax,
        with a .show() method, or None if show=True.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sreeg.electrodes import get_electrode_positions
    from equipotential import plot_equipotential_map

    # --- Resolve metadata ---
    if isinstance(activity, dict) and "activity" in activity:
        values = np.asarray(activity["activity"])
        electrodes = activity.get("electrodes", None)
        montage = activity.get("montage", montage)
        time_vector = activity.get("time", None)
    elif hasattr(activity, "montage") and hasattr(activity, "electrodes"):
        # ElectrodeActivity object
        values = np.asarray(activity)
        electrodes = getattr(activity, "electrodes", None)
        montage = getattr(activity, "montage", montage)
        time_vector = getattr(activity, "time", None)
    else:
        values = np.asarray(activity)
        electrodes, time_vector = None, None

    if montage is None:
        raise ValueError("Montage must be provided if not present in activity.")

    # --- Electrode positions ---
    positions, electrodes = get_electrode_positions(montage)

    # --- Handle temporal dimension ---
    selected_t = None
    selected_ch = None
    
    if values.ndim == 2:
        if when is not None:
            if isinstance(when, int):
                selected_t = int(when)
            elif isinstance(when, float) and time_vector is not None:
                selected_t = int(np.argmin(np.abs(time_vector - when)))
            else:
                raise ValueError("Invalid 'when': must be int index or float time (with metadata).")
            values_used = values[:, selected_t]
        else:
            # Align snapshot with prediction time: global max |values|
            selected_ch, selected_t = np.unravel_index(np.argmax(np.abs(values)), values.shape)
            selected_t = int(selected_t)
            values_used = values[:, selected_t]
    else:
        values_used = values

    # --- Call equipotential map (already draws fig/ax) ---
    mesh = plot_equipotential_map(
        positions, values_used,
        radius=0.1, cmap=cmap, subdiv=subdiv
    )

    fig, ax = plt.gcf(), plt.gca()
    brain_act = BrainActivity(mesh, positions, electrodes, values_used, fig, ax)
    brain_act.t_index = selected_t      # time index used for the snapshot (int or None)
    brain_act.ch_index = selected_ch    # channel index where max was found (int or None)
    brain_act.source_activity = activity
    brain_act.values_snapshot = values_used
    brain_act.values_full = values
    brain_act.time = time_vector
    brain_act.montage = montage
    brain_act.sfreq = getattr(activity, "sfreq", None)
    brain_act.frequency = getattr(activity, "frequency", None)
    
    if show:
        brain_act.show()
        return brain_act
    
    return brain_act

def activity_to_dipole(activity,
                       montage=None,
                       estimate_orientation=False,
                       estimate_position=False,
                       estimate_amplitude=False,
                       estimate_frequency=False):
    """
    Infer dipole parameters from scalp activity.

    The function executes only the minimal required steps, but always returns
    a dipole dictionary in the same format as `dipole_in_brain`. Fields not
    explicitly requested remain `None` or `NaN`. Additional metadata
    (`methods`, `brain_activity`, `sfreq`) is included for traceability.

    Parameters
    ----------
    activity : array-like, dict, or BrainActivity
        Electrode potentials, possibly with metadata. May be:
        - BrainActivity instance → used directly.
        - ElectrodeActivity / ndarray / EEG dict → internally normalized with `activity_brain`.
    montage : str, optional
        Electrode montage name (e.g., "biosemi32", "biosemi64").
        Required if activity lacks montage metadata.
    estimate_orientation : bool
        If True, estimate dipole orientation.
    estimate_position : bool
        If True, estimate dipole location.
    estimate_amplitude : bool
        If True, estimate dipole amplitude (requires orientation and position).
    estimate_frequency : bool
        If True, estimate dominant frequency via FFT.

    Returns
    -------
    dict
        Dipole dictionary with the same structure as `dipole_in_brain`, containing:
        - 'time' : ndarray or None
        - 'amplitude' : ndarray, float, or None
        - 'location' : ndarray(3,) or None
        - 'orientation' : ndarray(3,) or None
        - 'frequency' : float or None
        - 'phase' : float (default 0.0)
        - 'size' : float (default 0.20 m)
        - 'methods' : dict, processing traceability
        - 'brain_activity' : BrainActivity or None
        - 'sfreq' : float or None
    """
    import numpy as np

    # --- Local imports to avoid circular dependencies ---
    from sreeg.interface import activity_brain
    from onedipoleprediction import (
        dipole_orientation_by_nodal_plane,
        dipole_position_by_planes,
        dipole_orientation_by_fast_grid,
        dipole_position_by_fast_grid,
        dipole_amplitude_by_fit,
    )

    # ------------------------------------------------------------
    # 0) Resolve requested outputs
    # ------------------------------------------------------------
    want_ori  = bool(estimate_orientation)
    want_pos  = bool(estimate_position)
    want_amp  = bool(estimate_amplitude)
    want_freq = bool(estimate_frequency)

    need_spatial_3 = (want_ori and not (want_pos or want_amp))
    need_spatial_4 = (want_pos or want_amp)

    # ------------------------------------------------------------
    # 1) Extract montage and sampling frequency metadata
    # ------------------------------------------------------------
    montage_in = getattr(activity, "montage", None)
    if montage is None:
        montage = montage_in
    if montage is None and isinstance(activity, dict):
        montage = activity.get("montage", None)

    sfreq = getattr(activity, "sfreq", None)
    time_vec = getattr(activity, "time", None)
    if sfreq is None and isinstance(activity, dict):
        sfreq = activity.get("sfreq", None)
        if time_vec is None:
            time_vec = activity.get("time", None)

    methods = {"pipeline": {}, "sfreq_source": None}

    # ------------------------------------------------------------
    # 2) Sampling frequency policy
    # ------------------------------------------------------------
    def _derive_sfreq_if_possible(tv):
        """Attempt to derive sfreq from uniform time vector."""
        try:
            tv = np.asarray(tv, float)
            if tv.ndim != 1 or tv.size < 2:
                return None
            dt = np.diff(tv)
            if not np.all(np.isfinite(dt)):
                return None
            mu, sd = float(np.mean(dt)), float(np.std(dt))
            if mu <= 0 or (sd / (mu + 1e-12)) > 1e-3:  # tolerance 0.1%
                return None
            return float(1.0 / mu)
        except Exception:
            return None

    used_sfreq = None
    if sfreq is not None:
        used_sfreq = float(sfreq)
        methods["sfreq_source"] = "given"
    elif time_vec is not None:
        deriv = _derive_sfreq_if_possible(time_vec)
        if deriv is not None:
            used_sfreq = deriv
            methods["sfreq_source"] = "derived"
        else:
            methods["sfreq_source"] = "sfreq-required"
    else:
        methods["sfreq_source"] = "sfreq-required"

    # ------------------------------------------------------------
    # 3) Normalize to BrainActivity (only if spatial estimation needed)
    # ------------------------------------------------------------
    brain_act = None
    if need_spatial_3 or need_spatial_4:
        if montage is not None:
            if hasattr(activity, "mesh") and hasattr(activity, "values"):
                brain_act = activity
            else:
                # --- Automatic representative time selection based on global maximum ---
                act_arr = np.asarray(activity, float)
                if act_arr.ndim == 2:
                    ch_idx, t_idx = np.unravel_index(np.argmax(np.abs(act_arr)), act_arr.shape)
                else:
                    ch_idx, t_idx = 0, 0
                
                brain_act = activity_brain(activity, montage=montage, when=int(t_idx))
                
                # Store channel/time indices for later frequency estimation
                methods["time_selection"] = {"channel_index": int(ch_idx), "time_index": int(t_idx)}
            if used_sfreq is not None:
                try:
                    setattr(brain_act, "sfreq", used_sfreq)
                except Exception:
                    pass

    # ------------------------------------------------------------
    # 4) Spatial pipeline (S1–S4)
    # ------------------------------------------------------------
    orientation = None
    position = None
    amplitude = None

    if brain_act is not None:
        mesh = brain_act.mesh
        positions = brain_act.positions
        values = np.asarray(brain_act.values, float)

        # Step S1: Orientation via nodal plane
        res_ori_nodal = dipole_orientation_by_nodal_plane(mesh=mesh)
        methods["pipeline"]["S1"] = res_ori_nodal.get("method", "dipole_orientation_by_nodal_plane")
        u_nodal = np.asarray(res_ori_nodal.get("direction", [np.nan, np.nan, np.nan]), float)

        # --- Sign disambiguation from electrode data ---
        if np.all(np.isfinite(u_nodal)) and values is not None:
            half_proj = positions @ u_nodal
            mask_pos = half_proj >= 0
            if mask_pos.sum() > 0 and (~mask_pos).sum() > 0:
                mean_pos = values[mask_pos].mean()
                mean_neg = values[~mask_pos].mean()
                # enforce convention: positive side must carry higher mean potential
                if mean_pos < mean_neg:
                    u_nodal = -u_nodal

        # Step S2: Position via planes
        res_pos_planes = dipole_position_by_planes(
            positions=positions,
            values=values,
            orientation=u_nodal,
            n_planes=7,
            band_halfwidth=0.01,
            loss="zMSE",
            sigma=0.33
        )
        methods["pipeline"]["S2"] = res_pos_planes.get("method", "dipole_position_by_planes")
        p_planes = np.asarray(res_pos_planes.get("position", [np.nan, np.nan, np.nan]), float)

        # Step S3: Orientation refinement via fast grid
        res_ori_grid = dipole_orientation_by_fast_grid(
            positions=positions,
            values=values,
            pos_est=p_planes,
            n_orient=128,
            sigma=0.33,
            loss="zMSE"
        )
        methods["pipeline"]["S3"] = res_ori_grid.get("method", "dipole_orientation_by_fast_grid")
        u_grid = np.asarray(res_ori_grid.get("direction", [np.nan, np.nan, np.nan]), float)

        # --- Sign disambiguation (S1 → S3) ---
        # The fast grid search (S3) is invariant to dipole sign, because the scaling
        # factor α can absorb polarity. To ensure orientation consistency, we inherit
        # the sign from the nodal-plane estimate (S1).
        if np.dot(u_grid, u_nodal) < 0:
            u_grid = -u_grid

        if need_spatial_4:
            # Step S4: Position refinement via fast grid
            res_pos_grid = dipole_position_by_fast_grid(
                positions=positions,
                values=values,
                orientation=u_grid,
                pos_est=p_planes,
                axis_dir=None,
                coarse_line_span=0.07,
                coarse_n_line=9,
                line_span=0.02,
                n_line=11,
                radial_offsets=(0.0, 0.003, -0.003),
                two_pass=True,
                shrink=0.5,
                sigma=0.33,
                loss="zMSE"
            )
            methods["pipeline"]["S4"] = res_pos_grid.get("method", "dipole_position_by_fast_grid")

            pos_s4 = np.asarray(res_pos_grid.get("position", p_planes), float)
            loss_s4 = float(res_pos_grid.get("loss", np.inf))
            loss_s2 = float(res_pos_planes.get("loss", np.inf))
            if np.isfinite(loss_s4) and (loss_s4 < 0.90 * loss_s2):
                position = pos_s4
                methods["pipeline"]["S4_accept"] = True
            else:
                position = p_planes
                methods["pipeline"]["S4_accept"] = False

            orientation = u_grid

            if want_amp:
                res_amp = dipole_amplitude_by_fit(
                    positions=positions,
                    values=values,
                    location=position,
                    orientation=orientation,
                    sigma=0.33,
                    loss="zMSE"
                )
                amplitude = float(res_amp.get("alpha", np.nan))
                methods["pipeline"]["AMP"] = res_amp.get("method", "dipole_amplitude_by_fit")
            

        else:
            if want_ori:
                orientation = u_grid

    # ------------------------------------------------------------
    # 5) Frequency estimation (FFT, independent of spatial steps)
    # ------------------------------------------------------------
    frequency = None
    if want_freq:
        # Helper always available
        def _extract_matrix(values_like):
            if isinstance(values_like, np.ndarray):
                return values_like
            if isinstance(values_like, dict) and "activity" in values_like:
                return np.asarray(values_like["activity"], float)
            return None  # BrainActivity does not retain temporal structure
    
        # Priority: reuse frequency metadata if available
        if hasattr(activity, "frequency") and activity.frequency is not None:
            frequency = float(activity.frequency)
            methods["frequency"] = "from-metadata"
        else:
            mat = _extract_matrix(activity)
            if used_sfreq is None:
                frequency = float("nan")
                methods["frequency"] = "sfreq-required"
            elif mat is None or (mat.ndim != 2 or mat.shape[1] < 2):
                frequency = float("nan")
                methods["frequency"] = "no-temporal"
            else:
                ch_sel = methods.get("time_selection", {}).get("channel_index", 0)
                sig = np.asarray(mat[ch_sel], float)
                T = sig.shape[0]
                freqs = np.fft.rfftfreq(T, d=1.0 / used_sfreq)
                psd = np.abs(np.fft.rfft(sig))**2
                if psd[1:].size == 0:
                    frequency = float("nan")
                    methods["frequency"] = f"no-peak-found@ch{ch_sel}"
                else:
                    k = int(np.argmax(psd[1:])) + 1
                    frequency = float(freqs[k])
                    methods["frequency"] = f"from-fft@ch{ch_sel}"

    # ------------------------------------------------------------
    # 6) Build dipole return (stable format)
    # ------------------------------------------------------------
    dipole_est = {
        "time": time_vec if (want_amp or want_freq) else None,
        "amplitude": amplitude if want_amp else None,
        "location": position if want_pos else None,
        "orientation": orientation if want_ori else None,
        "frequency": frequency if want_freq else None,
        "phase": 0.0,
        "size": 0.20,
        # extras
        "methods": methods,
        "brain_activity": brain_act,
        "sfreq": used_sfreq
    }

    return dipole_est

def dipole_contrast(ref_dipole, *dipoles):
    """
    Compare one or more dipoles against a reference dipole and return
    a formatted report of differences.

    Parameters
    ----------
    ref_dipole : dict
        Reference dipole dictionary (from `dipole_in_brain` or
        `activity_to_dipole`).
    *dipoles : dict
        Dipole dictionaries to compare against the reference. Each dipole
        may contain:
        - 'location' : ndarray(3,), dipole position [m]
        - 'orientation' : ndarray(3,), dipole orientation
        - 'amplitude' : float, array, or dict with key "signal"/"amplitude"
        - 'frequency' : float, dominant frequency [Hz]

    Returns
    -------
    str
        Multiline string report. For each dipole compared:
            Dipole N vs Ref
              Δloc  = xx.xx mm
              Δori  = xx.xx°
              Δamp  = xx.xx %
              Δfreq = xx.xx %
    """
    import numpy as np

    def _as_peak_scalar(d):
        if d is None:
            return None
        if isinstance(d, dict):
            if "signal" in d:
                arr = np.asarray(d["signal"], float)
            elif "amplitude" in d:
                arr = np.asarray(d["amplitude"], float)
            else:
                return None
        else:
            arr = np.asarray(d, float)
        if arr.ndim == 0:
            return float(arr)
        return float(np.max(np.abs(arr)))

    lines = []
    loc_ref = np.asarray(ref_dipole.get("location"), float) if ref_dipole.get("location") is not None else None
    ori_ref = np.asarray(ref_dipole.get("orientation"), float) if ref_dipole.get("orientation") is not None else None
    amp_ref = _as_peak_scalar(ref_dipole.get("amplitude"))
    freq_ref = ref_dipole.get("frequency")

    for idx, dip in enumerate(dipoles, start=2):
        # Δ location
        loc = dip.get("location")
        if loc_ref is not None and loc is not None:
            dloc = float(np.linalg.norm(np.asarray(loc) - loc_ref))
            dloc_str = f"{dloc*1000:.2f} mm"
        else:
            dloc_str = "n/a"

        # Δ orientation
        ori = dip.get("orientation")
        if ori_ref is not None and ori is not None:
            v1 = ori_ref / (np.linalg.norm(ori_ref) + 1e-12)
            v2 = np.asarray(ori, float) / (np.linalg.norm(ori) + 1e-12)
            cosang = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cosang))
            dori_str = f"{angle:.2f}°"
        else:
            dori_str = "n/a"

        # Δ amplitude
        amp = _as_peak_scalar(dip.get("amplitude"))
        if amp_ref is not None and amp is not None and amp_ref > 1e-12:
            damp = 100.0 * (amp / amp_ref - 1.0)
            damp_str = f"{damp:.2f} %"
        else:
            damp_str = "n/a"

        # Δ frequency
        freq = dip.get("frequency")
        if freq_ref is not None and freq is not None and np.isfinite(freq) and freq_ref > 1e-12:
            dfreq = 100.0 * (freq / freq_ref - 1.0)
            dfreq_str = f"{dfreq:.2f} %"
        else:
            dfreq_str = "n/a"

        lines.append(
            f"Dipole {idx} vs Ref\n"
            f"  Δloc  = {dloc_str}\n"
            f"  Δori  = {dori_str}\n"
            f"  Δamp  = {damp_str}\n"
            f"  Δfreq = {dfreq_str}"
        )

    return "\n".join(lines)
