"""
SREEG: Simulation and Reconstruction of EEG Signals
===================================================

A research-oriented library for modeling dipole sources,
projecting their activity onto EEG electrodes, and solving
inverse problems (orientation, position, amplitude, frequency).

Modules
-------
- simulation: dipole source generation
- brain: spherical conductor model and conductivity profiles
- electrodes: standard electrode positions and forward model
- equipotential: topographic maps and nodal analysis
- onedipoleprediction: inverse methods for single dipole estimation
"""

__version__ = "0.1.0"

# --- Core API: Forward Modeling ---
from .electrodes import (
    get_electrode_positions,
    evaluate_dipole_at_electrodes,
)

# --- Core API: Equipotential Mapping ---
from .equipotential import (
    plot_equipotential_map,
    equipotentials,
)

# --- Core API: Inverse Prediction ---
from .onedipoleprediction import (
    dipole_orientation_by_nodal_plane,
    dipole_position_by_planes,
    dipole_orientation_by_fast_grid,
    dipole_position_by_fast_grid,
    dipole_amplitude_by_fit,
    onedipole_frequency_by_fft,
)

__all__ = [
    # Forward
    "get_electrode_positions",
    "evaluate_dipole_at_electrodes",
    # Equipotential
    "plot_equipotential_map",
    "equipotentials",
    # Inverse
    "dipole_orientation_by_nodal_plane",
    "dipole_position_by_planes",
    "dipole_orientation_by_fast_grid",
    "dipole_position_by_fast_grid",
    "dipole_amplitude_by_fit",
    "onedipole_frequency_by_fft",
]
