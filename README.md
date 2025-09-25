# corefinder

**corefinder** is a Python package designed to identify and trace dense cores in 3D magnetohydrodynamics (MHD) astrophysical simulations, with a focus on Eulerian grid data and periodic boundary conditions. The package implements a moving isosurface method to robustly track the evolution of cores across simulation snapshots.

## Features

- **Core Identification:** Find dense cores in 3D simulation data using isosurface-based segmentation.
- **Periodic Boundary Support:** Handles periodic boundary conditions, ideal for Eulerian grid simulations.
- **Core Evolution Tracking:** Trace the evolution of individual cores across time steps.
- **Flexible Input:** Works with standard NumPy arrays and supports common astrophysical data formats.
- **Extensible:** Easily integrates with other analysis pipelines.

## Installation

```sh
pip install corefinder
```

**Note**: I have not made the PyPI repo ready yet, please use `pip install xxx.whl`, where the `xxx.whl` can be obtained by `python -m build --wheel`.

Or, for development:

```sh
pip install -e .
```

## Usage

```python
import corefinder as cf
import pickle
import numpy as np


# Load the core
with open("./tests/core_snap042id001.pickle", "rb") as f:
    core: cf.CoreCube = pickle.load(f)


print(core)  # oneline summary

core.info()  # detailed summary

print(core.phyinfo)  # physical information

# example of the mass
core_array = core.data(-2, dataset_name="density", return_data_type="masked")
core_mass = core_array.sum() * core.phyinfo["pixel_size"] ** 3
print(f"Mass of the core: {core_mass:.2e} Msun")

# example of the vx
vx, roi, mask = core.data(-2, dataset_name="Vx", return_data_type="subcube_roi_mask")
# currently, ROI (region of interest) is not used, simply being np.ones_like(data)
print("The std of vx: ", np.std(vx[mask]))
print("The mean of vx: ", np.mean(vx[mask]))

```

See the [documentation](https://github.com/yourusername/corefinder) (in prep.) for more details and advanced usage.

## Requirements

- Python >= 3.9
- numpy >= 2.0.0
- scikit-image >= 0.24.0
- connected-components-3d >= 3.18.0
- h5py >= 3.10.0
- scipy >= 1.14.0

## License

MIT License

## Citation

If you use **corefinder** in your research, please contact the author for citation information (because it is still in a developing status, though usable)

~~please cite:~~

> ~~Yuan, S. (2025). CoreFinder: A Python package for core identification in 3D MHD simulations (in perp.). [GitHub repository](https://github.com/S-Yuan137/CoreFinder).~~

---
For questions or contributions, please open an issue or pull request on GitHub.

## Acknowledge

This package was developed within the vibrant research environment of CUHK-SFG. Its development draws heavily on the foundational codebase kindly shared by Dr. Cao, an inspiring collaborator whose expertise in astrophysics and AI has profoundly influenced the project's trajectory. Special thanks are also due to Mr. Chen and Mr. Sun for their substantial support in the technical development of this package, especially in debugging, data processing, and system optimization.
