# sreeg/electrodes.py

import numpy as np
import mne

# --- Internal cache for per-montage electrode positions ---
_positions_cache = {}

def get_electrode_positions(montage_name="biosemi32"):
    """
    Loads standard EEG electrode positions using MNE.

    Parameters
    ----------
    montage_name : str
        Name of the electrode system to load. Examples:
        'biosemi32', 'biosemi64', 'standard_1020', etc.

    Returns
    -------
    positions : ndarray, shape (N, 3)
        3D positions of electrodes in meters.
    names : list of str
        Names of the electrodes.
    """
    if montage_name in _positions_cache:
        return _positions_cache[montage_name]

    montage = mne.channels.make_standard_montage(montage_name)
    pos_dict = montage.get_positions()["ch_pos"]
    names = list(pos_dict.keys())
    positions = np.array([pos_dict[name] for name in names])

    _positions_cache[montage_name] = (positions, names)
    return positions, names


def evaluate_dipole_at_electrodes(dipole, electrode_positions, sigma=0.33, ax=None):
    """
    Computes the electric potential induced by a dipole at the positions of EEG electrodes,
    and visualizes the result as a topographic map (topoplot) using MNE if available.

    Parameters
    ----------
    dipole : dict
        Dictionary containing 'location' and 'orientation', both in meters.
    electrode_positions : array-like, shape (N, 3)
        3D coordinates of electrode positions (in meters).
    sigma : float, optional
        Electrical conductivity of the medium [S/m]. Default is 0.33.
    ax : matplotlib.axes._subplots.AxesSubplot or None
        Axis to plot into. If None, a new figure will be created.

    Returns
    -------
    potential : ndarray, shape (N,)
        Electric potential at each electrode [V].
    """

    # --- Potential computation ---
    r0 = np.array(dipole["location"])
    p = np.array(dipole["orientation"])
    pos = np.array(electrode_positions)

    diff = pos - r0
    norm = np.linalg.norm(diff, axis=1)
    norm[norm < 1e-6] = np.inf # Avoid division by zero
    dot = np.dot(diff, p)
    potential = (1 / (4 * np.pi * sigma)) * (dot / norm**3)

    # --- Visualization with MNE ---
    if mne is not None:
        import matplotlib.pyplot as plt

        # Use provided names if available, otherwise fallback to E0, E1...
        names = dipole.get("names", [f"E{i}" for i in range(len(pos))])

        # Create a montage directly from the electrode positions
        montage = mne.channels.make_dig_montage(
            ch_pos={names[i]: tuple(pos[i]) for i in range(len(pos))},
            coord_frame='head'
        )

        # Extract 2D coordinates aligned with the order of 'potential'
        pos_2d = np.array([
            montage.get_positions()["ch_pos"][name][:2]
            for name in names
        ])

        # Create MNE Info object
        info = mne.create_info(ch_names=names, sfreq=1000, ch_types="eeg")
        evoked = mne.EvokedArray(potential[:, np.newaxis], info)
        evoked.set_montage(montage)

        # Plot topomap
        if ax is None:
            fig, ax_out = plt.subplots(figsize=(4, 4))
        else:
            ax_out = ax

        mne.viz.plot_topomap(
            potential, pos_2d,
            cmap='RdBu_r',
            vlim=(-np.max(np.abs(potential)), np.max(np.abs(potential))),
            show=False,
            contours=6,
            size=3.5,
            axes=ax_out
        )

        if 'fig' in locals():
            plt.close(fig)

    else:
        import warnings
        warnings.warn("MNE is not installed. Only the potential values are returned.",
                      RuntimeWarning, stacklevel=2)

    return potential
