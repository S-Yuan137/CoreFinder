import corefinder as cf
import pickle
import numpy as np


# Load the core
with open("./core_snap042id001.pickle", "rb") as f:
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
