import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sreeg.brain import get_sigma_at, sigma_to_rgb

def simulate_dipole(
    time,
    location,
    orientation,
    *,
    amplitude_fn=None,
    frequency: float | None = None,
    amplitude: float = 1.0,
    phase: float = 0.0
):
    """
    Simulate a dipole source inside the brain.

    Parameters
    ----------
    time : array-like
        Time vector [s].
    location : array-like, shape (3,)
        Dipole location in meters.
    orientation : array-like, shape (3,)
        Dipole orientation vector.
    amplitude_fn : callable, optional
        Custom function f(time) -> signal. If provided, overrides frequency/amplitude/phase.
    frequency : float, optional
        Frequency of sinusoidal source [Hz]. Used if amplitude_fn is None.
    amplitude : float, default=1.0
        Amplitude of sinusoidal source.
    phase : float, default=0.0
        Phase offset of sinusoidal source [radians].

    Returns
    -------
    dict
        {
          "time": array,
          "signal": array,
          "location": array(3,),
          "orientation": array(3,)
        }
    """
    time = np.asarray(time, dtype=float)

    if amplitude_fn is not None:
        signal = amplitude_fn(time)
    elif frequency is not None:
        signal = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    else:
        signal = np.full_like(time, fill_value=amplitude, dtype=float)

    orientation = np.asarray(orientation, dtype=float)
    norm = np.linalg.norm(orientation)
    if norm == 0:
        raise ValueError("Dipole orientation vector must be non-zero.")
    orientation = orientation / norm

    return {
        "time": time,
        "signal": signal,
        "location": np.asarray(location, dtype=float),
        "orientation": orientation
    }

def plot_dipole_3d(
    dipoles,
    background: bool = True,
    realism: str = "vacuum",
    size: float = 0.20,
    figsize: tuple = (6, 6),
    show: bool = True,
    ax=None
):
    """
    Visualize one or more dipoles inside a cubic brain box.

    Parameters
    ----------
    dipoles : dict or list of dict
        Dipole(s) with keys {"location": array(3,), "orientation": array(3,)}.
    background : bool, default=True
        If True, render anatomical projections on the box walls.
    realism : {"vacuum", "cross"}, default="vacuum"
        Background style: static anatomical images ("vacuum") or conductivity map ("cross").
    size : float, default=0.20
        Physical cube size [m] representing the head.
    figsize : tuple, default=(6, 6)
        Matplotlib figure size.
    show : bool, default=True
        If True, display the figure immediately.
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        External 3D axis to plot into.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the 3D visualization.
    """

    if not isinstance(dipoles, list):
        dipoles = [dipoles]

    # Adapt box size to the 'size' argument (meters)
    internal_L = size / 2.0
    visual_L = size / 2.0

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        created_fig = True
    else:
        fig = ax.figure

    ax.set_xlim([-visual_L, visual_L])
    ax.set_ylim([-visual_L, visual_L])
    ax.set_zlim([-visual_L, visual_L])
    ax.set_box_aspect([1, 1, 1])

    # --- Background handling ---
    if background and realism == 'cross':
        ref_pos = np.array(dipoles[0]["location"])
        x_d, y_d, z_d = ref_pos
        res = 200
        step = (2 * internal_L) / res
        axis = np.linspace(-internal_L + step/2, internal_L - step/2, res)
        X, Y = np.meshgrid(axis, axis)

        vec_sigma = np.vectorize(lambda x, y, z: get_sigma_at(x, y, z))
        vec_rgb = np.vectorize(lambda sigma: sigma_to_rgb(sigma), otypes=[float, float, float])

        sigma_xy = vec_sigma(X, Y, z_d)
        rgb_xy = np.stack(vec_rgb(sigma_xy), axis=-1)
        Z_xy = np.full_like(X, -internal_L)
        ax.plot_surface(X, Y, Z_xy, facecolors=rgb_xy, rstride=1, cstride=1, shade=False, alpha=0.6)

        sigma_yz = vec_sigma(-X, Y, z_d)
        rgb_yz = np.stack(vec_rgb(sigma_yz), axis=-1)
        X_yz = np.full_like(X, -internal_L)
        ax.plot_surface(X_yz, Y, X, facecolors=rgb_yz, rstride=1, cstride=1, shade=False, alpha=0.6)

        sigma_xz = vec_sigma(X, y_d, -Y)
        rgb_xz = np.stack(vec_rgb(sigma_xz), axis=-1)
        Y_xz = np.full_like(X, internal_L)
        ax.plot_surface(X, Y_xz, Y, facecolors=rgb_xz, rstride=1, cstride=1, shade=False, alpha=0.6)

    elif background:
        try:
            asset_dir = os.path.join(os.path.dirname(__file__), 'assets')
            projections = [
                {"filename": "brain_Lateral_48.png",  "plane": "x", "fixed_val": -internal_L, "rotate": lambda img: np.rot90(img, 2)},
                {"filename": "brain_Frontal_48.png",  "plane": "y", "fixed_val": -internal_L, "rotate": lambda img: np.rot90(img, 2)},
                {"filename": "brain_Superior_48.png", "plane": "z", "fixed_val": -internal_L, "rotate": lambda img: np.rot90(img, 0)},
            ]

            for proj in projections:
                path = os.path.join(asset_dir, proj["filename"])
                if not os.path.exists(path):
                    continue

                img = mpimg.imread(path)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                elif img.shape[-1] == 4:
                    img = img[..., :3]

                h, w = img.shape[:2]
                step_x = (2 * internal_L) / w
                step_y = (2 * internal_L) / h
                extent_x = np.linspace(-internal_L + step_x/2, internal_L - step_x/2, w)
                extent_y = np.linspace(-internal_L + step_y/2, internal_L - step_y/2, h)
                A, B = np.meshgrid(extent_x, extent_y)

                fixed_val = np.full_like(A, proj["fixed_val"])

                if proj["plane"] == "z":
                    X, Y, Z = A, B, fixed_val
                elif proj["plane"] == "y":
                    X, Z, Y = A, B, fixed_val
                elif proj["plane"] == "x":
                    Y, Z, X = A, B, fixed_val

                ax.plot_surface(
                    X, Y, Z,
                    facecolors=proj["rotate"](img),
                    rstride=1, cstride=1,
                    shade=False, alpha=0.30,
                    linewidth=0, antialiased=False, zorder=0.1
                )
        except (FileNotFoundError, OSError):
            pass

    # --- Dipoles plotting ---
    base_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    for idx, dipole in enumerate(dipoles):
        pos = np.array(dipole["location"])
        ori = np.array(dipole["orientation"])
        color = base_colors[idx % len(base_colors)]
        label = dipole.get("label", f"Dipole {idx+1}")

        ax.scatter(*pos, color=color, s=60, label=label, zorder=10)
        ax.quiver(*pos, *ori, length=0.02, normalize=True, color=color, linewidth=2, zorder=10)

        # XY projection
        Lz = -internal_L
        ax.plot([pos[0]], [pos[1]], [Lz], marker='o', color=color, alpha=0.8)
        ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [pos[2], Lz], color=color, linestyle='--', linewidth=1)
        ax.quiver(pos[0], pos[1], Lz, ori[0], ori[1], 0, length=0.02, normalize=True, color=color, alpha=0.6)

        # YZ projection
        Ly = -internal_L
        ax.plot([pos[0]], [Ly], [pos[2]], marker='o', color=color, alpha=0.8)
        ax.plot([pos[0], pos[0]], [pos[1], Ly], [pos[2], pos[2]], color=color, linestyle='--', linewidth=1)
        ax.quiver(pos[0], Ly, pos[2], ori[0], 0, ori[2], length=0.02, normalize=True, color=color, alpha=0.6)

        # XZ projection
        Lx = -internal_L
        ax.plot([Lx], [pos[1]], [pos[2]], marker='o', color=color, alpha=0.8)
        ax.plot([pos[0], Lx], [pos[1], pos[1]], [pos[2], pos[2]], color=color, linestyle='--', linewidth=1)
        ax.quiver(Lx, pos[1], pos[2], 0, ori[1], ori[2], length=0.02, normalize=True, color=color, alpha=0.6)

    ax.view_init(elev=22, azim=30)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Dipole in 3D Space")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    if created_fig:
        if show:
            plt.tight_layout()
            plt.show()
        else:
            import matplotlib
            matplotlib.pyplot.ioff()
            plt.close(fig)
            matplotlib.pyplot.ion()
        return fig
