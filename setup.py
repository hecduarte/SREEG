from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sreeg",
    version="0.1.0",
    author="Héctor Duarte",
    author_email="hector.duarte@usm.cl",  # cámbialo si prefieres no usar correo institucional
    description="Simulation and Reconstruction of EEG Signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hecduarte/SREEG",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "mne",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.9",
)
