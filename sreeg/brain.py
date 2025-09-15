# sreeg/brain.py
import numpy as np

# -------------------------------
# Geometrical and Physical Constants
# -------------------------------

BRAIN_RADIUS = 0.09    # meters (approx. 9 cm radius)
SCALP_RADIUS = 0.10    # meters (outer head surface)
VOXEL_SIZE = 0.005     # meters (5 mm resolution)

# -------------------------------
# Electrical Conductivity (S/m)
# -------------------------------

CONDUCTIVITY = {
    'scalp': 0.43,
    'skull': 0.01,
    'csf':   1.79,
    'brain': 0.33
}

# Optional: anatomical layer thicknesses from outside inward
LAYER_DEPTHS = {
    'scalp': 0.005,
    'skull': 0.007,
    'csf':   0.003,
    'brain': BRAIN_RADIUS  # remainder
}

# -------------------------------
# Volumetric Brain Mask (voxel grid)
# -------------------------------

GRID_RANGE = np.arange(-BRAIN_RADIUS, BRAIN_RADIUS + VOXEL_SIZE, VOXEL_SIZE)
X, Y, Z = np.meshgrid(GRID_RANGE, GRID_RANGE, GRID_RANGE, indexing='ij')

# Boolean mask: True for voxels inside brain radius
BRAIN_MASK = (X**2 + Y**2 + Z**2 <= BRAIN_RADIUS**2)

# Centered coordinate list of all valid voxel positions
BRAIN_POINTS = np.column_stack((X[BRAIN_MASK], Y[BRAIN_MASK], Z[BRAIN_MASK]))

# -------------------------------
# Spatial Query Utilities
# -------------------------------

def is_inside_brain(position):
    """
    Returns True if the point is inside the spherical brain volume.
    Parameters
    ----------
    position : array-like, shape (3,)
        Cartesian 3D coordinates [x, y, z] in meters.
    """
    return np.linalg.norm(position) <= BRAIN_RADIUS


def get_conductivity_at(position):
    """
    Returns the local electrical conductivity (S/m) at a 3D position
    based on concentric spherical model of scalp, skull, CSF, brain.

    Parameters
    ----------
    position : array-like, shape (3,)
        Cartesian 3D coordinates in meters.

    Returns
    -------
    sigma : float
        Electrical conductivity [S/m] at given position.
    """
    r = np.linalg.norm(position)
    depth = SCALP_RADIUS - r
    if depth < 0:
        return 0.0
    elif depth < LAYER_DEPTHS['scalp']:
        return CONDUCTIVITY['scalp']
    elif depth < LAYER_DEPTHS['scalp'] + LAYER_DEPTHS['skull']:
        return CONDUCTIVITY['skull']
    elif depth < (LAYER_DEPTHS['scalp'] + LAYER_DEPTHS['skull'] + LAYER_DEPTHS['csf']):
        # CSF layer
        return CONDUCTIVITY['csf']
    else:
        return CONDUCTIVITY['brain']


def get_sigma_at(x, y, z):
    """
    Alias of get_conductivity_at, with separate arguments for iteration purposes.
    """
    return get_conductivity_at([x, y, z])


def sigma_to_rgb(sigma):
    """
    Map conductivity value (S/m) to an approximate RGB color for visualization.

    Parameters
    ----------
    sigma : float
        Electrical conductivity [S/m].

    Returns
    -------
    rgb : tuple of float
        RGB color values in range [0, 1].
    """
    if sigma == 0.0:
        return (1.0, 1.0, 1.0)
    elif sigma < 0.02:
        return (0.85, 0.81, 0.77)
    elif sigma < 0.37:
        return (0.89, 0.63, 0.45)
    elif sigma < 0.8:
        return (0.95, 0.70, 0.70)
    else:
        return (0.20, 0.60, 0.86)
        

def get_conductor_model():
    """
    Returns a single-layer spherical conductor model for MNE dipole fitting.

    Returns
    -------
    model : mne.ConductorModel
        Spherical conductor model centered at origin with brain conductivity.
    """
    import mne
    return mne.make_sphere_model(
        r0=(0., 0., 0.),
        relative_radii=(1.0,),
        sigmas=(CONDUCTIVITY['brain'],)
    )


