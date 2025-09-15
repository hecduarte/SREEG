# SREEG: Simulation and Reconstruction of EEG Signals

SREEG is a Python library for **dipole simulation, projection, and inverse reconstruction** of EEG activity.  
It provides tools to simulate brain dipoles, project them onto EEG electrodes, reconstruct their properties, and compare results through quantitative contrasts and visualizations.

## Features

- Dipole simulation in a simplified spherical brain model.  
- Forward projection onto standard EEG montages (e.g. BioSemi32/64/128).  
- Inverse reconstruction of dipole orientation, position, amplitude, and frequency.  
- Equipotential maps and nodal curve analysis.  
- 3D visualization of dipoles inside a brain cube with anatomical projections.  
- Compatible with MNE-Python.

## Installation

Clone this repository and install locally:

    git clone https://github.com/hecduarte/SREEG.git
    cd SREEG
    pip install -e .

Requirements:
- numpy
- scipy
- matplotlib
- mne

## Quick Example

    import numpy as np
    from sreeg.interface import (
        dipole_in_brain,
        dipole_to_electrodes,
        activity_brain,
        activity_to_dipole,
        brain_box,
        dipole_contrast
    )

    # 1) Define a known dipole (reproducible)
    dip1 = dipole_in_brain(
        location=[-0.03, 0.03, 0.05],
        orientation=[-3.0, -3.0, 1.0]
    )

    # 2) Project onto electrodes (BioSemi128)
    EEG = dipole_to_electrodes(dip1, montage="biosemi128")

    # 3) Encapsulate into an Activity object
    Activity = activity_brain(EEG, show=True)

    # 4) Reconstruct dipole from EEG activity
    dip2 = activity_to_dipole(
        Activity,
        estimate_orientation=True,
        estimate_position=True,
        estimate_amplitude=True,
        estimate_frequency=True
    )

    # 5) Compare original vs reconstructed dipoles
    contrasts = dipole_contrast(dip1, dip2)
    print(contrasts)

    # 6) Visualize both dipoles in a 3D brain box
    brain_box(dipoles=[(dip1, "red"), (dip2, "blue")], show=True)

## License

This project is released under the MIT License.

## Citation

If you use this library in academic work, please cite:

    @misc{duarte2025sreeg,
      author       = {Héctor A. Duarte-Portilla},
      title        = {Simulation and Reconstruction of EEG Signals (SREEG)},
      year         = {2025},
      publisher    = {GitHub},
      howpublished = {\url{https://github.com/hecduarte/SREEG}}
    }
