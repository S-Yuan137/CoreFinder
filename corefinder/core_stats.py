import numpy as np


def convert_box_from_downpixel_to_real(
    downpixeled_boxes: dict,
    block_size: tuple
) -> dict:
    """
    Convert the bounding boxes from downpixel to real size.

    Parameters
    ----------
    downpixeled_boxes: dict
        The bounding boxes in downpixel.
    block_size: tuple
        The block size for downpixel.

    Returns
    -------
    box_from_downpixel_to_real: dict
        The bounding boxes in real size.

    """
    box_real = {}
    for tag, (lower_indices, size) in downpixeled_boxes.items():
        lower_indices_real = [lower_indices[i]*block_size[i] for i in range(3)]
        size_real = [size[i] * block_size[i] for i in range(3)]
        box_real[tag] = (lower_indices_real, size_real)
    return box_real


def uppixel(binary_cube, block_size: tuple):
    """
    Up pixel (upsample) the binary cube by a factor.

    Parameters:
    - binary_cube: numpy.ndarray
        The binary cube to be upsampled.
    - block_size: tuple
        The block size for upsample.

    Returns:
    - upsampled_cube: numpy.ndarray
        The upsampled binary cube.
    """
    assert len(binary_cube.shape) == 3, "Only 3D data cube is supported."
    uppixel_cube = np.repeat(
        np.repeat(
            np.repeat(binary_cube, block_size[0], axis=0), block_size[1], axis=1),
        block_size[2],
        axis=2,
    )
    return uppixel_cube


def geometry_info(masked_density: np.ndarray) -> tuple[float, float, float]:
    # one ref https://www.sciencedirect.com/science/article/pii/S003132030700324X
    # ! Some CAS parameters for 3D data
    return masked_density.shape
    # pass


def discrete_compactness(mask: np.ndarray[bool]) -> float:
    """
    Compute the discrete compactness for the (binary) mask. The discrete compactness is
    in the range of [0, 1], where 0 means the mask is not compact at all and 1 means
    the mask is the most compact.

    Parameters
    ----------
    mask : np.ndarray[bool]
        The binary mask of volumetric data.

    Returns
    -------
    discrete_compactness : float
        The discrete compactness of the mask, defined as $A_c / A_{c, max}$,
        where $A_c$ is the contact area and $A_{c, max}$ is the maximum contact area.
        See [Bribiesca E. 2008](https://www.sciencedirect.com/science/article/pii/S003132030700324X)
    """
    # ! TODO: Implement 26-connectivity support
    front_contact = np.zeros_like(mask, dtype=bool)
    front_contact[1:, :, :] = mask[:-1, :, :] * mask[1:, :, :]
    top_contact = np.zeros_like(mask, dtype=bool)
    top_contact[:, :, 1:] = mask[:, :, :-1] * mask[:, :, 1:]
    left_contact = np.zeros_like(mask, dtype=bool)
    left_contact[:, 1:, :] = mask[:, :-1, :] * mask[:, 1:, :]
    area = np.sum(front_contact) + np.sum(top_contact) + np.sum(left_contact)
    max_contact = 3 * (mask.sum() - mask.sum() ** (2 / 3))
    return area / max_contact
    
    

def energy_info():
    # ! Some energy parameters for 3D data
    pass


def self_gravity_potential():
    # ! Compute the gravity potential for the 3D data
    pass


def virial_analysis():
    # ! Perform the virial analysis for the 3D data
    pass

def forces_analysis():
    # ! Perform the forces analysis for the 3D data
    pass
