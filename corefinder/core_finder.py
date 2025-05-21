import numpy as np
import cc3d
import h5py
import pickle
from skimage.measure import block_reduce
from skimage.morphology import remove_small_holes
from . import core_stats


class DataCube:
    def __init__(self, data, mask=None, phyinfo=None):
        self._data = data
        self._mask = mask
        self.phyinfo = self._get_phyinfo(phyinfo)
        self.shape = data.shape
        self.ndim = data.ndim
        self.size = data.size
        self.dtype = data.dtype
        self._check_mask()

    def _check_mask(self):
        if self._mask is None:
            self._mask = np.ones(self.shape, dtype=bool)
        else:
            if self._mask.shape != self.shape:
                raise ValueError("Mask shape does not match data shape")
            if self._mask.dtype != bool:
                raise ValueError("Mask must be boolean")

    def _get_phyinfo(self, phyinfo):
        if phyinfo is None:
            return {
                "pixel_size": None,
                "boundary": None,
                "time": None,
                "length_unit": None,
                "time_unit": None,
                "value_unit": None,
            }
        else:
            if not isinstance(phyinfo, dict):
                raise ValueError("Physical information must be a dictionary")
            pixel_size = phyinfo.get("pixel_size", None)
            boundary = phyinfo.get("boundary", None)
            time = phyinfo.get("time", None)
            length_unit = phyinfo.get("length_unit", None)
            time_unit = phyinfo.get("time_unit", None)
            value_unit = phyinfo.get("value_unit", None)
            return {
                "pixel_size": pixel_size,
                "boundary": boundary,
                "time": time,
                "length_unit": length_unit,
                "time_unit": time_unit,
                "value_unit": value_unit,
            }

    def __repr__(self):
        return (
            f"datacube in shape of {self.shape} in pixels, with pixel size "
            f"of {self.phyinfo['pixel_size']} {self.phyinfo['length_unit']}. "
            f"Details see {self.__class__.__name__}.info()"
        )

    def __str__(self):
        return (
            f"datacube in shape of {self.shape} in pixels, with pixel size "
            f"of {self.phyinfo['pixel_size']} {self.phyinfo['length_unit']}. "
            f"Details see {self.__class__.__name__}.info()"
        )

    def info(self, *args):
        if len(args) == 0:
            # print out the information of the datacube
            print(f"Data shape: {self.shape}")
            print(f"Data size: {self.size}")
            print(f"Data type: {self.dtype}")
            print(f"Data ratio in ROI: {np.sum(self._mask)/self.size:.2f}")
            print(f"Physical information: {self.phyinfo}")
        else:
            for arg in args:
                if arg in self.phyinfo:
                    print(f"{arg}: {self.phyinfo[arg]}")
                elif arg in ["shape", "dimension", "size", "type"]:
                    print(f"Data {arg}: {getattr(self, arg)}")
                else:
                    pass  # other keywords are not valid until in derived class


class MaskCube(DataCube):
    def __init__(
        self,
        data: np.ndarray,
        ROI: np.ndarray,
        masks: dict[float, np.ndarray],
        refpoints: dict[float, tuple],
        internal_id: int,
        snapshot: int = None,
        phyinfo: dict = None,
        **kwargs,
    ):
        """
        The MaskCube class is used to store the clump and core data.

        Parameters
        ----------
        data: numpy.ndarray
            The data cube.
        ROI: numpy.ndarray
            The region of interest (ROI) mask.
        masks: dict
            The masks for different thresholds.
        refpoints: dict
            The reference points for different thresholds.
        internal_id: int
            The internal ID of the mask cube.
        snapshot: int, optional
            The snapshot number. Default is None.
        phyinfo: dict, optional
            The physical information. Default is None.
        **kwargs:
            - file_load_path: str
                The file path to load the data.
            - original_shape: tuple
                The original shape of the raw data cube. Default is
                (960,960,960).

        Notes
        -----
        - The mask cube is used to store the clump and core data.
        - The masks and refpoints are stored in a dictionary with the
        thresholds as the keys.
            - thresholds: float, positive number defining the threshold for
            the clump mask. The negative number defining the threshold for the
            core mask. For example, -2.0 is used for the core mask meaning the
            mass of the core is 2.0 Msun. 20.0 is used for the clump mask
            meaning the density threshold is 20.0 Msun/pc^3. The units are
            defined in the phyinfo.
        - The internal ID is used to identify different cores within the same
        snapshot.
            - when the internal ID is positive, it is from positive evolution tracking
            by the predicted-spatial-overlap method.
            - when the internal ID is negative, it is from the reverse tracking by
            the particle tracing to the previous snapshot, due to the amorphous core
            ancestor.

        """
        if data is None:
            raise ValueError("Data must be provided")
        if masks is None:
            raise ValueError("masks must be provided")
        if not isinstance(masks, dict):
            raise ValueError("masks must be a dictionary")
        if not isinstance(refpoints, dict):
            raise ValueError("refpoints must be provided as a dictionary")
        self.masks = masks
        self.thresholds = np.array(list(masks.keys()))
        self.refpoints = refpoints
        self.internal_id = internal_id
        self.snapshot = snapshot
        self.file_load_path = kwargs.get("file_load_path", None)
        self.original_shape = kwargs.get("original_shape", (960, 960, 960))
        if self.file_load_path is None:
            raise ValueError("file_load_path must be provided")
        if not all(key in self.refpoints for key in self.masks):
            raise ValueError(
                "masks and refpoints must have the same keys as thresholds"
            )
        super().__init__(data, ROI, phyinfo)
        self._check_data()

    def info(self, *args):
        super().info(*args)
        if len(args) == 0:
            print(f"Snapshot: {self.snapshot}")
            print(f"Keys of masks: {self.masks.keys()}")
            print(f"Sizes of masks: {[mask.shape for mask in self.masks.values()]}")
            print(f"Refpoints: {self.refpoints}")
            print(f"Internal ID: {self.internal_id}")
            print(f"Original shape: {self.original_shape}")
            print(f"File load path: {self.file_load_path}")
        else:
            for arg in args:
                if arg == "snapshot":
                    print(f"{arg}: {self.snapshot}")
                elif arg == "masks":
                    print(f"{arg} keys: {self.masks.keys()}")
                    print(
                        f"{arg} sizes: {[mask.shape for mask in self.masks.values()]}"
                    )
                elif arg == "refpoints":
                    print(f"{arg}: {self.refpoints}")
                elif arg == "internal_id":
                    print(f"{arg}: {self.internal_id}")
                elif arg == "original_shape":
                    print(f"{arg}: {self.original_shape}")
                elif arg == "file_load_path":
                    print(f"{arg}: {self.file_load_path}")
                else:
                    print(f"{arg} is not a valid keyword")

    def __repr__(self):
        return (
            f"maskcube in shape of {self.shape} in pixels, with pixel size "
            f"of {self.phyinfo['pixel_size']} {self.phyinfo['length_unit']}. "
            f"Details see {self.__class__.__name__}.info()"
        )

    def __str__(self):
        return (
            f"maskcube in shape of {self.shape} in pixels, with pixel size "
            f"of {self.phyinfo['pixel_size']} {self.phyinfo['length_unit']}. "
            f"Details see {self.__class__.__name__}.info()"
        )
        
    def __eq__(self, other):
        if not isinstance(other, MaskCube):
            return False
        if not np.allclose(self._data, other._data):
            return False
        if not np.allclose(self._mask, other._mask):
            return False
        if not np.allclose(self.thresholds, other.thresholds):
            return False
        if not all(
            np.allclose(self.masks[key], other.masks[key])
            for key in self.masks.keys()
        ):
            return False
        if not all(
            np.allclose(self.refpoints[key], other.refpoints[key])
            for key in self.refpoints.keys()
        ):
            return False
        if self.internal_id != other.internal_id:
            return False
        if self.snapshot != other.snapshot:
            return False
        if self.file_load_path != other.file_load_path:
            return False
        if self.original_shape != other.original_shape:
            return False
        return True

    def _check_ROI(self):
        if self._mask is None:
            self._mask = np.ones(self.shape, dtype=bool)
        else:
            if self._mask.shape != self.shape:
                raise ValueError("Mask shape does not match instance shape")
            if self._mask.dtype != bool:
                raise ValueError("Mask must be boolean")

    def _check_data(self):
        # this is assumption of Nested Cube for masks, that is,
        # (A, B, C) > (a, b, c) when A > a, B > b, C > c
        largest_mask_shape = max([mask.shape for mask in self.masks.values()])
        if self._data.shape != largest_mask_shape:
            raise ValueError("Data shape does not match the largest mask shape")
        if self.shape != self._data.shape:
            raise ValueError("Instance shape does not match data shape")

    def _get_threshold_of_largest_subcube(self) -> float:
        """
        Get the threshold of the largest subcube (mask), i.e., the self._data.
        """
        return max(self.masks, key=lambda x: self.masks[x].shape)

    def _in_largest_subcube(self, coord: tuple) -> bool:
        """
        Check if the pixel coordinate is in the largest subcube.

        Note: using numpy array indexing rules, like
              a[1:2] does not include a[2].
        """
        if len(coord) != 3:
            raise ValueError("The coordinate must be a tuple of 3 elements.")
        if not all(isinstance(i, (int, np.int64)) for i in coord):
            raise ValueError("The coordinate must be integers.")
        if not isinstance(coord, tuple):
            raise ValueError("The coordinate must be a tuple to avoid indexing error.")
        subcube_lower = self.refpoints[self._get_threshold_of_largest_subcube()]
        shape = self.masks[self._get_threshold_of_largest_subcube()].shape
        return MaskCube._in_subcube(coord, subcube_lower, shape, self.original_shape)

    @staticmethod
    def _in_subcube(
        coord: tuple, refpoint: tuple, shape: tuple, original_cube_shape: tuple
    ) -> bool:
        """
        Check if the pixel coordinate is in the subcube defined by refpoint
        and size. The subcube is in the original cube with periodic boundary.

        Note: using numpy array indexing rules, like
              a[1:2] does not include a[2].
        """
        if len(coord) != 3:
            raise ValueError("The coordinate must be a tuple of 3 elements.")
        if not all(isinstance(i, (int, np.int64)) for i in coord):
            raise ValueError("The coordinate must be integers.")
        if not isinstance(coord, tuple):
            raise ValueError("The coordinate must be a tuple to avoid indexing error.")
        subcube_lower = np.array(refpoint)
        subcube_upper = subcube_lower + np.array(shape)
        subcube_over_bound = [
            subcube_upper[i] > original_cube_shape[i] for i in range(3)
        ]
        in_subcube = [False, False, False]
        for i in range(3):
            if subcube_over_bound[i]:
                if (0 <= coord[i] < subcube_upper[i] - original_cube_shape[i]) or (
                    subcube_lower[i] <= coord[i] < original_cube_shape[i]
                ):
                    in_subcube[i] = True
            else:
                if subcube_lower[i] <= coord[i] < subcube_upper[i]:
                    in_subcube[i] = True
        return all(in_subcube)

    def _covered_by_new_subcube(self, new_refpoint: tuple, new_shape: tuple) -> bool:
        """
        Check if the new subcube covers the current largest one, including
        the same region.
        """
        original_cube_shape = self.original_shape
        current_refpoint = self.refpoints[self._get_threshold_of_largest_subcube()]
        if not MaskCube._in_subcube(
            current_refpoint, new_refpoint, new_shape, original_cube_shape
        ):
            return False
        current_size = self.masks[self._get_threshold_of_largest_subcube()].shape
        current_upper = current_refpoint + np.array(current_size)
        new_upper = np.array(new_refpoint) + np.array(new_shape)
        cover = [False, False, False]
        for i in range(3):
            if (
                new_refpoint[i] <= current_refpoint[i]
                and new_upper[i] >= current_upper[i]
            ):

                cover[i] = True
            elif (
                new_refpoint[i] <= current_refpoint[i] + original_cube_shape[i]
                and new_upper[i] >= current_upper[i] + original_cube_shape[i]
            ):
                cover[i] = True
        return all(cover)

    def _cover_new_subcube(self, new_refpoint: tuple, new_shape: tuple) -> bool:
        """
        Check if the new subcube is covered by the current largest one.
        """
        original_cube_shape = self.original_shape
        current_refpoint = self.refpoints[self._get_threshold_of_largest_subcube()]
        if not self._in_largest_subcube(new_refpoint):
            return False
        current_shape = self.masks[self._get_threshold_of_largest_subcube()].shape
        current_upper = current_refpoint + np.array(current_shape)
        new_upper = np.array(new_refpoint) + np.array(new_shape)
        cover = [False, False, False]
        for i in range(3):
            if (
                new_refpoint[i] >= current_refpoint[i]
                and new_upper[i] <= current_upper[i]
            ):
                cover[i] = True
            elif (
                new_refpoint[i] >= current_refpoint[i] - original_cube_shape[i]
                and new_upper[i] <= current_upper[i] - original_cube_shape[i]
            ):
                cover[i] = True
        return all(cover)

    def _pixel_coordinate_in_subcube(self, coord: tuple) -> tuple:
        """
        Convert the pixel coordinate in the original cube to relative
        one in the self._data (largest subcube).
        """
        if self._in_largest_subcube(coord):
            relative_coord = MaskCube._compute_relative_coord(
                self.refpoints[self._get_threshold_of_largest_subcube()],
                coord,
                self.original_shape,
            )
        else:
            raise ValueError("The coordinate is not in the largest subcube.")
        return relative_coord

    @staticmethod
    def _compute_relative_coord(
        coord1: tuple, coord2: tuple, original_cube_shape: tuple
    ) -> tuple:
        """
        Compute the relative coordinate of coord2 to coord1 (coord1 as new
        origin).
        """
        # ! avoid using this directly, it should be in core_stats.py infuture.
        if len(coord1) != 3 or len(coord2) != 3:
            raise ValueError("The coordinate must be a tuple of 3 elements.")
        if not all(isinstance(i, (int, np.int64)) for i in coord1 + coord2):
            raise ValueError("The coordinate must be integers.")
        if not all(0 <= i < original_cube_shape[j] for j, i in enumerate(coord1)):
            raise ValueError("The coordinate is not in the original cube.")
        if not all(0 <= i < original_cube_shape[j] for j, i in enumerate(coord2)):
            raise ValueError("The coordinate is not in the original cube.")
        new_coord = np.array(coord2) - np.array(coord1)
        for i in range(3):
            if new_coord[i] < 0:
                new_coord[i] += original_cube_shape[i]
        return tuple(new_coord)

    def load_one_h5_data(self, dataset_name, file_load_path=None):
        # warnings.warn(
        #     "This method may involve multiple loading of the same file. ",
        #     "Consider using load_snap_h5_data method.",
        #     "If have to use, be cautious.",
        # )
        if file_load_path is None:
            file_load_path = self.file_load_path
        with h5py.File(file_load_path, "r") as f:
            dataset = f[dataset_name][...].T
        return dataset

    def _update_data(self, new_data: np.ndarray, new_ROI: np.ndarray):
        if new_data.shape != new_ROI.shape:
            raise ValueError("Data shape does not match ROI shape")
        else:
            self._data = new_data
            self._mask = new_ROI
            self.shape = new_data.shape
            self.ndim = new_data.ndim
            self.size = new_data.size

    def _update_mask(
        self, new_threshold: float, new_mask: np.ndarray, new_refpoint: tuple
    ):
        # check ref point
        if new_threshold in self.thresholds:
            print(
                f"Warning: Threshold {new_threshold} already exists. "
                "The old values will be replaced"
            )
        else:
            self.thresholds = np.append(self.thresholds, new_threshold)
        self.masks[new_threshold] = new_mask
        self.refpoints[new_threshold] = new_refpoint

    def update_data_mask(
        self,
        new_data: np.ndarray,
        new_ROI: np.ndarray,
        new_threshold: float,
        new_mask: np.ndarray,
        new_refpoint: tuple,
    ) -> None:
        if self._cover_new_subcube(new_refpoint, new_mask.shape):
            self._update_mask(new_threshold, new_mask, new_refpoint)
            self._check_ROI()
            self._check_data()
        elif self._covered_by_new_subcube(new_refpoint, new_mask.shape):
            self._update_data(new_data, new_ROI)
            self._update_mask(new_threshold, new_mask, new_refpoint)
            self._check_ROI()
            self._check_data()
        else:
            # raise ValueError(
            #     "The new input subcube does not cover or "
            #     "is covered by the current largest one."
            # )
            print(
                "Warning: The new input subcube does not cover or "
                "is covered by the current largest one."
            )
            print("Warning: Not updating the MaskCube.")
            # self._update_mask(new_threshold, new_mask, new_refpoint)

    def compute_new_mask(
        self, new_threshold: float, density_cube: np.ndarray, **kargs
    ) -> np.ndarray[bool]:
        """
        Compute the new mask based on the existed data

        Parameters
        ----------
        new_threshold : float
            the new value of density threshold to extract the isolated strucure
        density_cube : np.ndarray
            the original data cube

        Returns
        -------
        new_mask: np.ndarray[bool]
            the new mask corresponding to the new threhsold
        """

        tag_cube = cc3d.connected_components(
            density_cube >= new_threshold,
            out_dtype=np.uint64,
            connectivity=6,
            periodic_boundary=True,
        )
        locate_method = kargs.get("locate_method", "peak density")
        if locate_method == "center of mass":
            pass
        elif locate_method == "peak density":
            peak_coord_in_subcube = np.array(
                np.unravel_index(np.argmax(self._data), self._data.shape)
            )
            peak_coord = peak_coord_in_subcube + np.array(
                self.refpoints[self._get_threshold_of_largest_subcube()]
            )
            for i in range(3):
                if peak_coord[i] >= self.original_shape[i]:
                    peak_coord[i] -= self.original_shape[i]
            peak_coord = tuple(peak_coord)
            seed_tag = tag_cube[peak_coord]
        new_mask = tag_cube == seed_tag
        return new_mask

    def dump(self, file_path: str):
        """
        Dump the MaskCube instance to a pickle file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def get_subcube_from_rawcube(
        refpoint: tuple[int | np.int64],
        shape: tuple[int | np.int64],
        original_cube: np.ndarray,
        periodic_boundary: bool = True,
    ) -> np.ndarray:
        """get the subcube data from the original cube (raw cube).

        Parameters
        ----------
        refpoint : tuple
            the reference point of the subcube
        shape : tuple
            the shape of the subcube
        original_cube : np.ndarray
            the original cube
        periodic_boundary : bool, optional
            whether to consider the periodic boundary, by default True

        Returns
        -------
        subcube_data : np.ndarray
            the subcube data from the original cube
        """
        subcube_lower = np.array(refpoint)
        subcube_upper = subcube_lower + np.array(shape)
        over_bound_axis = [subcube_upper[i] > original_cube.shape[i] for i in range(3)]
        if not periodic_boundary and np.any(over_bound_axis):
            raise ValueError("The bounding box is out of the raw cube.")
        for axis, over_bound in enumerate(over_bound_axis):
            if over_bound:
                original_cube = np.roll(original_cube, -subcube_lower[axis], axis=axis)
                subcube_upper[axis] = np.array(shape)[axis]
                subcube_lower[axis] = 0
        subcube_data = original_cube[
            subcube_lower[0] : subcube_upper[0],
            subcube_lower[1] : subcube_upper[1],
            subcube_lower[2] : subcube_upper[2],
        ]
        return subcube_data

    def get_specific_subcube(
        self,
        threshold: float,
        original_cube: np.ndarray = None,
        dataset_name: str = None,
    ) -> np.ndarray:
        """
        get the specific subcube data by the threshold.

        Parameters
        ----------
        threshold : float
            the threshold to get the subcube
        original_cube : np.ndarray, optional
            the raw datacube, by default None
        dataset_name : str, optional
            dataset name in original hdf5 file, by default None

        Returns
        -------
        subcube
            the subcube data by the threshold

        Notes
        -----
        This function is expected to be used to get the the data besides the
        density, like the velocity, B-field, etc. Hence,
        - The original_cube is preferred to be provided.
        - Either original_cube or dataset_name must be provided.
        - If both original_cube and dataset_name are provided, the
        original_cube will be used.

        """
        if threshold not in self.thresholds:
            raise ValueError("The threshold is not in the thresholds.")
        refpoint = self.refpoints[threshold]
        shape = self.masks[threshold].shape
        if original_cube is None and dataset_name is None:
            raise ValueError("Either original_cube or dataset_name must be provided.")
        elif original_cube is None and dataset_name is not None:
            print("Warning: Try to find the original cube, which may be slow.")
            original_cube = self.load_one_h5_data(
                dataset_name, file_load_path=self.file_load_path
            )
        elif original_cube is not None and dataset_name is not None:
            print(
                "Warning: Both original_cube and dataset_name are provided. "
                "The original_cube will be used."
            )
        elif original_cube is not None and dataset_name is None:
            pass
        return MaskCube.get_subcube_from_rawcube(
            refpoint, shape, original_cube, periodic_boundary=True
        )

    def get_previous_structure(
        self,
        threshold: float = -2,
        time_step: float = 4,
    ):
        """
        Get the previous structure of the subcube by the threshold, using passive tracer.


        Notes
        -----
        - The passive tracer is used, i.e., the tracer is not affected by the each other.
        And tracers do not change the velocity field.
        - The time_step is in the unit of pixel_size. It is 4 in default resolution of
        original data (960^3). More details see the `predict_next_position` method in
        `SimCube` class.
        """
        if threshold not in self.thresholds:
            raise ValueError("The threshold is not in the thresholds.")

        # load previous data cube
        def _previous_file_path(file_path):
            # file_path = self.file_load_path in format of "/path/to/hdfaa.012"
            # return "/path/to/hdfaa.011", note last is hdfaa.{snap:03d}
            file_path = file_path.split(".")
            file_path[-1] = f"{int(file_path[-1])-1:03d}"
            return ".".join(file_path)

        file_load_path = _previous_file_path(self.file_load_path)

        # ! here the dataset name is hard coded, need to be changed

        with h5py.File(file_load_path, "r") as f:
            prev_den = f["gas_density"][...].T
            prev_vx = f["i_velocity"][...].T
            prev_vy = f["j_velocity"][...].T
            prev_vz = f["k_velocity"][...].T

        assert prev_den.shape == prev_vx.shape == prev_vy.shape == prev_vz.shape
        assert prev_den.shape == self.original_shape
        mask = self.masks[threshold]
        refpoint = self.refpoints[threshold]
        # get coordinates of masked region ( true values in mask)
        coord = np.argwhere(mask) + np.array(refpoint)
        # here is a trick to avoid float -> int conversion by zeros_like
        previous_coord = np.zeros_like(coord)
        for i, c in enumerate(coord):
            # deal with periodic boundary
            cx = c[0] if c[0] < self.original_shape[0] else c[0] - self.original_shape[0]
            cy = c[1] if c[1] < self.original_shape[1] else c[1] - self.original_shape[1]
            cz = c[2] if c[2] < self.original_shape[2] else c[2] - self.original_shape[2]
            previous_coord[i] = coord[i] - time_step * np.array(
                [
                    prev_vx[cx, cy, cz],
                    prev_vy[cx, cy, cz],
                    prev_vz[cx, cy, cz],
                ]
            )

        prev_refpoint = (
            np.min(previous_coord[:, 0]),
            np.min(previous_coord[:, 1]),
            np.min(previous_coord[:, 2]),
        )
        prev_shape = (
            np.max(previous_coord[:, 0]) - prev_refpoint[0] + 1,
            np.max(previous_coord[:, 1]) - prev_refpoint[1] + 1,
            np.max(previous_coord[:, 2]) - prev_refpoint[2] + 1,
        )
        prev_mask = np.zeros(prev_shape, dtype=bool)
        for c in previous_coord:
            prev_mask[tuple(c - np.array(prev_refpoint))] = True
        # increase the robustness by removing small holes
        prev_mask = remove_small_holes(prev_mask, area_threshold=64, out=prev_mask)
        prev_data = MaskCube.get_subcube_from_rawcube(
            prev_refpoint, prev_shape, prev_den, periodic_boundary=True
        )
        # ! Future work: add ROI
        output = MaskCube(
            prev_data,
            np.ones_like(prev_mask, dtype=bool),
            {threshold: prev_mask},
            {threshold: prev_refpoint},
            internal_id=-int(
                abs(self.internal_id)
            ),  # negative internal_id for previous structure
            snapshot=self.snapshot - 1,  # previous snapshot
            phyinfo=self.phyinfo,
            file_load_path=file_load_path,
            original_shape=self.original_shape,
        )
        return output

    def find_clump(self, threshold, density_cube: np.ndarray, **kargs):
        """
        Find the clump by the threshold in the density cube. The clump will be stored in
        the mask cube automatically.

        Parameters
        ----------
        threshold : float
            The threshold of the clump
        density_cube : np.ndarray
            The density cube to find the clump, original cube from the hdf5 file.

        Raises
        ------
        ValueError
            The threshold must be positive for clump, different from the core.
        """
        if threshold < 0:
            raise ValueError("Threshold must be positive for clump")
        new_mask = self.compute_new_mask(threshold, density_cube, **kargs)
        new_boxes = SimCube.get_box_of_tags(new_mask, periodic_boundary=True)
        new_refpoint, new_size = new_boxes[np.True_]
        new_data = MaskCube.get_subcube_from_rawcube(
            new_refpoint, new_size, density_cube, periodic_boundary=True
        )
        new_mask = MaskCube.get_subcube_from_rawcube(
            new_refpoint, new_size, new_mask, periodic_boundary=True
        )
        new_ROI = np.ones_like(new_data, dtype=bool)  # ! Future work: add ROI
        self.update_data_mask(new_data, new_ROI, threshold, new_mask, new_refpoint)

    def find_core(
        self,
        target_mass: float,
        clump_threshold: float = None,
        **kargs,
    ):
        """
        Find the core by the target mass in the clump. The core will be stored in the
        mask cube automatically.

        Parameters
        ----------
        target_mass : float
            The target mass of the core. It must be negative. For example, -2.0 is used
            for 2.0 Msun, where `-` is used to distinguish from the clump threshold.
        clump_threshold : float, optional
            the threshold of the clump where to find the core, by default the most
            compact clump will be used (the largest positive threshold).

        Notes
        -----
        For details of the core finding, see the `MaskCube.get_fixed_mass_core` method
        as below:
        ```
        Parameters in get_fixed_mass_core
        ----------------------------------
        masked_data: numpy.ndarray
            The masked data. Note it also can be unmasked data.
        target_mass: float
            The target mass of the core.
        kwargs: dict
            The keyword arguments for the function, including:
            tolerance: float, optional
                The relative tolerance of the mass. Default is 0.05.
            max_iteration: int, optional
                The maximum iteration. Default is 50.
            num_sequence: int, optional
                The number of the search sequence. Default is 50.
            refine: bool, optional
                Whether to refine the core mass. Default is True.
        ```
        """

        if target_mass >= 0:
            raise ValueError(
                "Threshold (target_mass) must be negative for core, like -2"
                " for 2 Msun"
            )
        if clump_threshold is None:
            # will use the largest positive threshold (smallest clump) as the
            # density cube
            clump_threshold = max(self.thresholds[self.thresholds > 0])
        else:
            if clump_threshold <= 0:
                raise ValueError("Clump threshold must be positive.")
            elif clump_threshold not in self.thresholds:
                raise ValueError(
                    "Clump threshold is not in the thresholds. If it has to "
                    "be a new threshold, please use find_clump() method first."
                )
        clump_mask = self.masks[clump_threshold]
        clump_refpoint = self.refpoints[clump_threshold]
        # data_refpoint = self.refpoints[self._get_threshold_of_largest_subcube()]
        # relative_coord = MaskCube._compute_relative_coord(
        #     data_refpoint, clump_refpoint, original_shape
        # )
        relative_coord = self._pixel_coordinate_in_subcube(clump_refpoint)
        clump = self._data[
            relative_coord[0] : relative_coord[0] + clump_mask.shape[0],
            relative_coord[1] : relative_coord[1] + clump_mask.shape[1],
            relative_coord[2] : relative_coord[2] + clump_mask.shape[2],
        ]
        clump = clump * clump_mask
        # core will has the same shape as clump which is used to find the core
        if "pixel_length" not in kargs:
            pixel_length = self.phyinfo["pixel_size"]
        else:
            pixel_length = kargs["pixel_length"]
            kargs.pop("pixel_length")  # avoid the repeated keyword
        core_mask = MaskCube.get_fixed_mass_core(
            clump, -target_mass, pixel_length=pixel_length, **kargs
        )
        if isinstance(core_mask, str):
            print("Warning: The core mask is not found. Reason:")
            print(core_mask)
        elif isinstance(core_mask, np.ndarray):
            self._update_mask(target_mass, core_mask, clump_refpoint)
        else:
            raise ValueError("How is it possible?")

    def data(self, threshold: float = None, **kwargs) -> np.ndarray:
        """
        Get the data (density) of the mask cube by the threshold. For other datasets,
        use get_specific_subcube() method.

        Parameters
        ----------
        threshold: float, optional
            The threshold of the mask cube. Default is None.
        kwargs: dict
            The keyword arguments for the function, including:
            return_data_type: str, optional
                The type of the data to return. Default is "masked".
                - "subcube": the density data of the subcube.
                - "masked": the density data masked by the ROI and the
                  mask.
                - "subcube_roi_mask": the density data, the ROI, and the mask.

        Returns
        -------
        data: numpy.ndarray or tuple
            The data of the mask cube by the threshold.
            Or the tuple of the data, the ROI, and the mask.

        """

        valid_args = ["return_data_type"]
        for arg in kwargs:
            if arg not in valid_args:
                print(f"{arg} is not a valid keyword")
                kwargs.pop(arg)

        return_data_type = kwargs.get("return_data_type", "masked")
        if (threshold is not None) and (threshold not in self.thresholds):
            raise ValueError("The threshold is not in the thresholds.")
        if threshold is None:
            if return_data_type == "subcube":
                return self._data
            elif return_data_type == "masked":
                return (
                    self._data
                    * self._mask
                    * self.masks[self._get_threshold_of_largest_subcube()]
                )
            elif return_data_type == "subcube_roi_mask":
                return (
                    self._data,
                    self._mask,
                    self.masks[self._get_threshold_of_largest_subcube()],
                )
        else:
            refpoint = self._pixel_coordinate_in_subcube(self.refpoints[threshold])
            shape = self.masks[threshold].shape
            density = self._data[
                refpoint[0] : refpoint[0] + shape[0],
                refpoint[1] : refpoint[1] + shape[1],
                refpoint[2] : refpoint[2] + shape[2],
            ]
            roi = self._mask[
                refpoint[0] : refpoint[0] + shape[0],
                refpoint[1] : refpoint[1] + shape[1],
                refpoint[2] : refpoint[2] + shape[2],
            ]
            if return_data_type == "subcube":
                return density
            elif return_data_type == "masked":
                return density * roi * self.masks[threshold]
            elif return_data_type == "subcube_roi_mask":
                return density, roi, self.masks[threshold]
            else:
                raise ValueError("The return_data_type is not valid.")

    def geometry_info(self, threshold: float = None) -> tuple:
        density = self.data(threshold=threshold, return_data_type="masked_density")
        return core_stats.geometry_info(density)

    @staticmethod
    def get_fixed_mass_core(
        masked_data: np.ndarray, target_mass: float, **kwargs
    ) -> np.ndarray | str:
        """
        Get the fixed mass core from the masked data.

        Parameters
        ----------
        masked_data: numpy.ndarray
            The masked data. Note it also can be unmasked data.
        target_mass: float
            The target mass of the core.
        kwargs: dict
            The keyword arguments for the function, including:
            tolerance: float, optional
                The relative tolerance of the mass. Default is 0.05.
            max_iteration: int, optional
                The maximum iteration. Default is 50.
            num_sequence: int, optional
                The number of the search sequence. Default is 50.
            refine: bool, optional
                Whether to refine the core mass. Default is True.

        Returns
        -------
        core_mask: numpy.ndarray or string message
            The fixed mass core mask ndarray if any. Otherwise, the string
            message will be returned.

        """
        tolerance = kwargs.get("tolerance", 0.05)
        max_iteration = kwargs.get("max_iteration", 20)
        num_sequence = kwargs.get("num_sequence", 50)
        pixel_length = kwargs.get("pixel_length", 0.005)  # in pc
        refine = kwargs.get("refine", True)
        total_mass = np.sum(masked_data * pixel_length**3)
        # mass in the unit of Msun, 1 pixel = 0.005 pc
        if total_mass < target_mass * (1 - tolerance):
            return "The target mass should be less than the parental mass."

        if target_mass * (1 - tolerance) < total_mass < target_mass * (1 + tolerance):
            return masked_data > 0  # no negative density

        def single_mask(masked_data, contour, seed_point):
            temp_mask1 = masked_data >= contour
            temp_mask2 = cc3d.connected_components(
                temp_mask1, out_dtype=np.uint64, connectivity=26
            )  # disable the periodic boundary
            # select the highest density point as the seed point
            # ! Future work: add the option to select the seed point
            seed_label = temp_mask2[seed_point]
            return temp_mask2 == seed_label

        initial_contour_sequence = np.linspace(
            masked_data.max(), masked_data.min(), num_sequence
        )
        rough_contour = np.zeros(2)
        seed_point = np.unravel_index(np.argmax(masked_data), masked_data.shape)
        for i_contour, contour in enumerate(initial_contour_sequence):
            core_mask = single_mask(masked_data, contour, seed_point)
            mass_in_contour = np.sum(masked_data[core_mask] * pixel_length**3)
            if mass_in_contour >= target_mass:
                rough_contour[1] = contour
                rough_contour[0] = initial_contour_sequence[i_contour - 1]
                break
        else:
            return "The target mass is not reached."
        if abs(mass_in_contour - target_mass) < target_mass * tolerance:
            return core_mask
        elif refine:
            for _ in range(max_iteration):
                contour = np.mean(rough_contour)
                core_mask = single_mask(masked_data, contour, seed_point)
                mass_in_contour = np.sum(masked_data[core_mask] * pixel_length**3)
                if abs(mass_in_contour - target_mass) < target_mass * tolerance:
                    break
                elif mass_in_contour > target_mass:
                    rough_contour[1] = contour
                else:
                    rough_contour[0] = contour
            else:
                return "The target mass is not reached with refinement."
            return core_mask
        else:
            return "The target mass is not reached without refinement."

    @staticmethod
    def load_snap_h5_data(
        maskcubes: dict[int, "MaskCube"],
        dataset_name: str,
        file_load_paths: dict[int, str] = None,
    ):
        # aviod loading the same snapshot file multiple times
        if file_load_paths is None:
            file_load_paths = {
                maskcube.snapshot: maskcube.file_load_path
                for maskcube in maskcubes.values()
            }
        datasets = {}
        for snapshot, file_load_path in file_load_paths.items():
            with h5py.File(file_load_path, "r") as f:
                dataset = f[dataset_name][...].T
            datasets[snapshot] = dataset
        return datasets


class SimCube(DataCube):
    def __init__(self, data, mask, velocity_tuple, phyinfo=None, **kwargs):
        if data is None:
            raise ValueError("Data must be provided")
        if velocity_tuple is None:
            raise ValueError("Velocity tuple must be provided")
        elif not isinstance(velocity_tuple, tuple):
            raise ValueError("Velocity tuple must be a tuple")
        elif len(velocity_tuple) != 3:
            raise ValueError("Velocity tuple must have 3 elements")
        super().__init__(data, mask, phyinfo)
        self.snapshot = kwargs.get("snapshot", None)
        self.vx = velocity_tuple[0]
        self.vy = velocity_tuple[1]
        self.vz = velocity_tuple[2]
        self.compute_speedup = kwargs.get("compute_speedup", True)
        self.file_load_path = kwargs.get("file_load_path", None)

    def info(self, *args):
        super().info(*args)
        if len(args) == 0:
            print(f"Snapshot: {self.snapshot}")
        else:
            for arg in args:
                if arg == "snapshot":
                    print(f"{arg}: {self.snapshot}")
                else:
                    print(f"{arg} is not a valid keyword")

    def __repr__(self):
        return (
            f"simcube in shape of {self.shape} in pixels, with pixel size"
            f"of {self.phyinfo['pixel_size']} {self.phyinfo['length_unit']}. "
            f"Details see {self.__class__.__name__}.info()"
        )

    def __str__(self):
        return (
            f"simcube in shape of {self.shape} in pixels, with pixel size"
            f"of {self.phyinfo['pixel_size']} {self.phyinfo['length_unit']}. "
            f"Details see {self.__class__.__name__}.info()"
        )

    def tag_connected_volume(
        self,
        threshold: float,
        periodic: bool = True,
        sort_volume: bool = True,
        count_pixel_no_less_than: int = 37,
    ) -> np.ndarray:
        """
        Tags connected volume in a binary cube.

        Parameters
        ----------
        threshold: float
            The threshold to binarize the data cube.
        periodic: bool, optional
            Whether to consider the periodic boundary. Default is True.
        sort_volume: bool, optional
            Whether to sort the volume of the tagged volumes. Default is True.
        count_pixel_no_less_than: int, optional
            The count of pixels no less than this value will be tagged.
            Default is 37.

        Returns
        -------
        tagged_volume: numpy.ndarray
            The tagged volume. The data type is np.uint64.

        Notes
        -----
        - The first tag usually is the background of the data cube, and it is
        tagged as 0. The other tags are sorted by their volumes in descending
        if `sort_volume` is True.
        - The `count_pixel_no_less_than` is only used when `sort_volume` is
        True.

        """
        binary_cube = self._data > threshold
        dimensions = len(binary_cube.shape)
        assert dimensions == 3, "Only 3D data cube is supported."
        connectivity = 6
        labels_out = cc3d.connected_components(
            binary_cube,
            out_dtype=np.uint64,
            connectivity=connectivity,
            periodic_boundary=periodic,
        )

        if sort_volume:
            unique_labels, counts = np.unique(labels_out, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            sorted_labels = unique_labels[sorted_indices]
            sorted_counts = counts[sorted_indices]
            # Create a mapping from old labels to new labels
            # New labels are the indices in the sorted_labels array,
            # except those with counts < `count_pixel_no_less_than` being 0
            label_mapping = {}
            new_label = 0  # Start new labels from 0
            for old_label, count in zip(sorted_labels, sorted_counts):
                if count < count_pixel_no_less_than:
                    label_mapping[old_label] = 0  # Neglect small volumes
                else:
                    label_mapping[old_label] = new_label
                    new_label += 1
            # Apply the mapping to the original array, np.uint64
            sorted_array = np.vectorize(label_mapping.get)(labels_out)
            return sorted_array.astype(np.uint64)
        else:
            return labels_out

    def predict_next_position(
        self,
        tagged_volume: np.ndarray,
        time_step: float,
        periodic_boundary: bool = True,
        **kwargs,
    ):
        """
        predict the spatial positions of the tagged volumes in the next time
        step.

        Parameters
        ----------
        tagged_volume: numpy.ndarray
            The tagged volume to be predicted.
        time_step: float
            The time step.
        periodic_boundary: bool, optional
            Whether to consider the periodic boundary. Default is True.
        method: str, optional
            The method to compute the next position. Default is
            `center of mass`. Available methods are `center of mass`,
            `peak density`. `peak density` method refers to [Offner et al.
            (2022).](https://doi.org/10.1093/mnras/stac2734)

        Returns
        -------
        tagged_volume_new: numpy.ndarray
            The predicted tagged volume.

        Notes
        -----
        This function is dimensional dependent, that is, involving units.
        Default velocity unit is pc/Myr.
        Raw time snap is 0.02 Myr. 1 pixel = 0.005 pc. So the default time
        step is 4 in this function. If downsampled/upsampled in pixels (I call
        them downpixel/uppixel), the time step should be adjusted accordingly,
        e.g., if downsampled by 3 (960^3 --> 320^3), the time step should be
        4/3.
        The `periodic_boundary` is forced to be `True` in this function so far.

        """
        assert len(tagged_volume.shape) == 3, "Only 3D data cube is supported."
        assert (
            tagged_volume.shape
            == self._data.shape
            == self.vx.shape
            == self.vy.shape
            == self.vz.shape
        ), "The shapes of the input data should be the same."
        assert periodic_boundary, "The periodic boundary is forced to be True."
        method = kwargs.get("method", "center of mass")
        preidcted_volume = np.zeros_like(tagged_volume)
        for tag in np.unique(tagged_volume):
            if tag == 0:
                continue
            indices = np.where(tagged_volume == tag)
            indices_ = [0, 0, 0]
            if method == "center of mass":
                indices_[0] = (
                    indices[0]
                    + round(
                        np.average(self.vx[indices], weights=self._data[indices])
                        * time_step
                    )
                ) % tagged_volume.shape[0]
                indices_[1] = (
                    indices[1]
                    + round(
                        np.average(self.vy[indices], weights=self._data[indices])
                        * time_step
                    )
                ) % tagged_volume.shape[1]
                indices_[2] = (
                    indices[2]
                    + round(
                        np.average(self.vz[indices], weights=self._data[indices])
                        * time_step
                    )
                ) % tagged_volume.shape[2]
            elif method == "peak density":
                peak_density_velocity = (
                    self.vx[indices][np.argmax(self._data[indices])],
                    self.vy[indices][np.argmax(self._data[indices])],
                    self.vz[indices][np.argmax(self._data[indices])],
                )
                for i in range(3):
                    indices_[i] = (
                        indices[i] + round(peak_density_velocity[i] * time_step)
                    ) % tagged_volume.shape[i]
            else:
                raise ValueError("The method is not supported yet.")
            indices_ = tuple(indices_)
            preidcted_volume[indices_] = tag
        return preidcted_volume

    @staticmethod
    def get_box_of_tags(
        tagged_volume: np.ndarray, periodic_boundary: bool = True
    ) -> dict:
        """
        Compute the axis-aligned bounding box (AABB) of each tag
        in a tagged volume.

        Parameters
        ----------
        tagged_volume: np.ndarray
            The tagged volume.
        periodic_boundary: bool, optional
            Whether to consider periodic boundary conditions. Defaults to True.

        Returns
        -------
        box_of_tags: dict
            A dictionary containing the lower indices and size of the bounding
            box for each tag. The dictionary is structured as follows:
            {
                tag1: ((lower_x, lower_y, lower_z), (size_x, size_y, size_z)),
                tag2: ((lower_x, lower_y, lower_z), (size_x, size_y, size_z)),
                ...
            }

        The lower indices represent the starting indices of the bounding box in
        the original data cube. The size represents the size of the bounding
        box in each dimension.

        Example
        -------
        >>> tagged_volume = np.array(
        [
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 2, 0, 0], [2, 2, 0, 2], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=np.uint64,
        )
        >>> bounding_boxes = get_box_of_tags(tagged_volume)
        >>> print(bounding_boxes)
        {0: ([0, 0, 0], [5, 4, 4]), 1: ([4, 1, 2], [2, 1, 1]),
        2: ([0, 1, 3], [2, 2, 3]), 3: ([3, 2, 2], [1, 1, 1])}

        """
        tags = np.unique(tagged_volume)
        bounding_boxes = {}
        for tag in tags:
            indices = np.where(tagged_volume == tag)
            lower_indices = [np.min(indices[i]) for i in range(3)]
            upper_indices = [np.max(indices[i]) for i in range(3)]
            if periodic_boundary:
                for axis, indices_ in enumerate(zip(lower_indices, upper_indices)):
                    # this if statement is redundant, keep it for clarity
                    if indices_[0] == 0 and (
                        indices_[1] == tagged_volume.shape[axis] - 1
                    ):
                        lost_index = np.setdiff1d(
                            np.arange(tagged_volume.shape[axis]), indices[axis]
                        )
                        if len(lost_index) == 0:
                            continue
                        else:
                            lower_indices[axis] = np.max(lost_index) + 1
                            upper_indices[axis] = (
                                tagged_volume.shape[axis] + np.min(lost_index) - 1
                            )
            size = [upper_indices[i] - lower_indices[i] + 1 for i in range(3)]
            bounding_boxes[tag] = (tuple(lower_indices), tuple(size))
            
        # ! consider improve the function here by combining `convert_box_from_downpixel_to_real`
        # function in the `core_stats` module
        return bounding_boxes

    @staticmethod
    def get_spatial_overlap(
        tagged_volume1: np.ndarray, tagged_volume2: np.ndarray
    ) -> dict:
        """
        Compute the spatial overlap between two volumes.

        Parameters
        ----------
        tagged_volume1: ndarray
            The first volume. Tagged volumes as 0, 1, 2, ...
        tagged_volume2: ndarray
            The second volume. Tagged volumes as 0, 1, 2, ...

        Returns
        -------
        spatial_overlap: dict
            The spatial overlap between the two volumes.

        Notes
        -----
        Only 3D data cubes are supported.

        """
        dimensions = len(tagged_volume1.shape)
        assert dimensions == 3, "Only 3D data cube is supported."
        assert (
            tagged_volume1.shape == tagged_volume2.shape
        ), "The two volumes should have the same shape."
        result = {}
        for tag in np.unique(tagged_volume1):
            if tag == 0:
                continue
            volume1_in_volume2 = tagged_volume2[tagged_volume1 == tag]
            overlap_components_in_v2 = np.unique(volume1_in_volume2)
            overlap_ratio_over_v1 = np.zeros_like(
                overlap_components_in_v2, dtype=np.float64
            )
            overlap_ratio_over_v2 = np.zeros_like(
                overlap_components_in_v2, dtype=np.float64
            )
            for i, component in enumerate(overlap_components_in_v2):
                overlap_ratio_over_v1[i] = np.sum(
                    volume1_in_volume2 == component, dtype=np.float64
                ) / np.sum(tagged_volume1 == tag, dtype=np.float64)
                overlap_ratio_over_v2[i] = np.sum(
                    volume1_in_volume2 == component, dtype=np.float64
                ) / np.sum(tagged_volume2 == component, dtype=np.float64)
            result[tag] = (
                overlap_components_in_v2,
                overlap_ratio_over_v1,
                overlap_ratio_over_v2,
            )
        return result

    # this static method is used to load the snapshot from the zeus snapshot
    # return a SimCube object
    @staticmethod
    def load_zeus_snapshot(file_load_path=None, **kwargs) -> "SimCube":
        """
        Load a snapshot from the zeus snapshot.

        Parameters
        ----------
        **kwargs:
            compute_speedup: bool, optional
                Whether to compute the speedup. Default is True.
            snapshot: int, optional
                The snapshot number. Default is 0.
            time: float, optional
                The time of the snapshot. Default is accessed from the file.

        Returns
        -------
        simcube: SimCube
            The simulation cube object.

        """

        if file_load_path is None:
            raise ValueError("The file path must be provided.")
        with h5py.File(file_load_path, "r") as f:
            density = f["gas_density"][...].T
            vx = f["i_velocity"][...].T
            vy = f["j_velocity"][...].T
            vz = f["k_velocity"][...].T
            time = f["time"][()][0]
        snapshot = kwargs.get("snapshot", None)
        if not isinstance(time, float | np.float32 | np.float64):
            # zeus uses float32
            time = None
        mask = kwargs.get("mask", None)
        phyinfo = {
            "pixel_size": 0.005,
            "boundary": "perodic",
            "time": time,
            "length_unit": "pc",
            "time_unit": "Myr",
            "value_unit": "Msun/pc^3",
        }
        phyinfo_ = kwargs.get("phyinfo", {})
        # use phyinfo from kwargs if provided to update phyinfo_
        phyinfo.update(phyinfo_)
        speedup = kwargs.get("compute_speedup", True)
        print(f"Snapshot {snapshot} loaded from {file_load_path}")
        if speedup:
            # velocity average weighted by density
            vx = block_reduce(vx * density, (3, 3, 3), np.sum) / block_reduce(
                density, (3, 3, 3), np.sum
            )
            vy = block_reduce(vy * density, (3, 3, 3), np.sum) / block_reduce(
                density, (3, 3, 3), np.sum
            )
            vz = block_reduce(vz * density, (3, 3, 3), np.sum) / block_reduce(
                density, (3, 3, 3), np.sum
            )
            density = block_reduce(density, (3, 3, 3), np.mean)
            if mask is not None:
                mask = block_reduce(mask, (3, 3, 3), np.sum) > 0
        kwargs["file_load_path"] = file_load_path
        # delete phyinfo from kwargs
        kwargs.pop("phyinfo", None)
        return SimCube(density, mask, (vx, vy, vz), phyinfo=phyinfo, **kwargs)


class CoreCube(MaskCube):
    """
    The CoreCube class is used to store the clump and core data, with velocity and B-field
    information compared to the MaskCube class.

    Parameters
    ----------
    data: numpy.ndarray
        The data cube of density.
    extra_data: dict[str, numpy.ndarray]
        The extra data for the core cube.
    ROI: numpy.ndarray
        The region of interest (ROI) mask.
    masks: dict
        The masks for different thresholds.
    refpoints: dict
        The reference points for different thresholds.
    internal_id: int
        The internal ID of the mask cube.
    snapshot: int, optional
        The snapshot number. Default is None.
    phyinfo: dict, optional
        The physical information. Default is None.
    **kwargs:
        - file_load_path: str
            The file path to load the data.
        - original_shape: tuple
            The original shape of the raw data cube. Default is
            (960,960,960).

    Notes
    -----
    - The mask cube is used to store the clump and core data.
    - The masks and refpoints are stored in a dictionary with the
    thresholds as the keys.
        - thresholds: float, positive number defining the threshold for
        the clump mask. The negative number defining the threshold for the
        core mask. For example, -2.0 is used for the core mask meaning the
        mass of the core is 2.0 Msun. 20.0 is used for the clump mask
        meaning the density threshold is 20.0 Msun/pc^3. The units are
        defined in the phyinfo.
    - The internal ID is used to identify different cores within the same
    snapshot.
        - when the internal ID is positive, it is from positive evolution tracking
        by the predicted-spatial-overlap method.
        - when the internal ID is negative, it is from the reverse tracking by
        the particle tracing to the previous snapshot, due to the amorphous core
        ancestor.

    """

    def __init__(
        self,
        data: np.ndarray,
        extra_data: dict[str, np.ndarray],
        ROI: np.ndarray,
        masks: dict,
        refpoints: dict,
        internal_id: int,
        snapshot: int = None,
        phyinfo: dict = None,
        **kwargs,
    ):  
        for key in extra_data.keys():
            if extra_data[key].shape != data.shape:
                raise ValueError(
                    f"Shape of new data and new extra data should be the same. {key} is not the same."
                )
        super().__init__(
            data, ROI, masks, refpoints, internal_id, snapshot, phyinfo, **kwargs
        )
        self._extra_data = extra_data

    def info(self, *args):
        super().info(*args)
        if len(args) == 0:
            print(f"Extra data: {list(self._extra_data.keys())}")
        else:
            for arg in args:
                if arg == "_":
                    print(f"{arg}: {list(self._extra_data.keys())}")
                else:
                    print(f"{arg} is not a valid keyword")

    def __repr__(self):
        return (
            f"corecube in shape of {self.shape} in pixels, with pixel size"
            f"of {self.phyinfo['pixel_size']} {self.phyinfo['length_unit']}. "
            f"Details see {self.__class__.__name__}.info()"
        )

    def __str__(self):
        return (
            f"corecube in shape of {self.shape} in pixels, with pixel size"
            f"of {self.phyinfo['pixel_size']} {self.phyinfo['length_unit']}. "
            f"Details see {self.__class__.__name__}.info()"
        )
        
    def __eq__(self, value):
        if not isinstance(value, CoreCube):
            return False
        if not super().__eq__(value):
            return False
        # extra data is a dictionary, so we need to compare the keys and values
        if self._extra_data.keys() != value._extra_data.keys():
            return False
        for key in self._extra_data.keys():
            if not np.allclose(self._extra_data[key], value._extra_data[key]):
                return False
        return True

    def data(self, threshold: float = None, dataset_name = "density", return_data_type = "masked") -> np.ndarray:
        """
        Get the data (density, vx, Bz, ...) of the mask cube by the threshold.

        Parameters
        ----------
        threshold: float, optional
            The threshold of the mask cube. Default is None.
        kwargs: dict
            The keyword arguments for the function, including:
            dataset_name: str, optional
                The name of the dataset. Default is "density". The available
                datasets are: "density", "Vx", "Vy", "Vz", "Bx", "By", "Bz"
                and "gravity_potential".
            return_data_type: str, optional
                The type of the data to return. Default is "masked".
                - "subcube": the subcube data.
                - "masked": the subcube data masked by the ROI and the
                mask.
                - "subcube_roi_mask": the subcube data, the ROI, and the mask.

        Returns
        -------
        data: numpy.ndarray or tuple
            The data of the mask cube by the threshold.
            Or the tuple of the data, the ROI, and the mask.

        """

        if dataset_name not in self._extra_data.keys() and dataset_name != "density":
            raise ValueError(f"The dataset name {dataset_name} is not in the extra data.")
        
        if (threshold is not None) and (threshold not in self.thresholds):
            raise ValueError("The threshold is not in the thresholds.")

        if dataset_name == "density":
            return super().data(threshold=threshold, return_data_type=return_data_type)
        else:
            if threshold is None:
                if return_data_type == "subcube":
                    return self._extra_data[dataset_name]
                elif return_data_type == "masked":
                    return (
                        self._extra_data[dataset_name]
                        * self._mask
                        * self.masks[self._get_threshold_of_largest_subcube()]
                    )
                elif return_data_type == "subcube_roi_mask":
                    return (
                        self._extra_data[dataset_name],
                        self._mask,
                        self.masks[self._get_threshold_of_largest_subcube()],
                    )
            else:
                refpoint = self._pixel_coordinate_in_subcube(self.refpoints[threshold])
                shape = self.masks[threshold].shape
                data = self._extra_data[dataset_name][
                    refpoint[0] : refpoint[0] + shape[0],
                    refpoint[1] : refpoint[1] + shape[1],
                    refpoint[2] : refpoint[2] + shape[2],
                ]
                roi = self._mask[
                    refpoint[0] : refpoint[0] + shape[0],
                    refpoint[1] : refpoint[1] + shape[1],
                    refpoint[2] : refpoint[2] + shape[2],
                ]
                if return_data_type == "subcube":
                    return data
                elif return_data_type == "masked":
                    return data * roi * self.masks[threshold]
                elif return_data_type == "subcube_roi_mask":
                    return data, roi, self.masks[threshold]
                else:
                    raise ValueError("The return_data_type is not valid.")
    
    def find_core(
        self,
        target_mass: float,
        parental_threshold: float = None,
        **kargs,
    ):
        """
        Find the core by the target mass in the parental structure. The core will be stored in the
        mask cube automatically. Enhance the MaskCube.find_core() method.

        Parameters
        ----------
        target_mass : float
            The target mass of the core. It must be negative. For example, -2.0 is used
            for 2.0 Msun, where `-` is used to distinguish from the clump threshold.
        parental_threshold : float, optional
            the threshold of the parental structure. The default is None, which means
            the largest volume core (mast massive) will be used. The threshold can be
            negative.
        Notes
        -----
        For details of the core finding, see the `MaskCube.get_fixed_mass_core` method
        as below:
        ```
        Parameters in get_fixed_mass_core
        ----------------------------------
        masked_data: numpy.ndarray
            The masked data. Note it also can be unmasked data.
        target_mass: float
            The target mass of the core.
        kwargs: dict
            The keyword arguments for the function, including:
            tolerance: float, optional
                The relative tolerance of the mass. Default is 0.05.
            max_iteration: int, optional
                The maximum iteration. Default is 50.
            num_sequence: int, optional
                The number of the search sequence. Default is 50.
            refine: bool, optional
                Whether to refine the core mass. Default is True.
        ```
        """

        if target_mass >= 0:
            raise ValueError(
                "Threshold (target_mass) must be negative for core, like -2"
                " for 2 Msun"
            )
        if parental_threshold is None:
            # will use the most massive core as the parental structure
            parental_threshold = min(self.thresholds[self.thresholds < 0])
        else:
            if parental_threshold not in self.thresholds:
                raise ValueError(
                    "Clump threshold is not in the thresholds. If it has to "
                    "be a new threshold, please use find_clump() method first."
                )
        clump_mask = self.masks[parental_threshold]
        clump_refpoint = self.refpoints[parental_threshold]
        # data_refpoint = self.refpoints[self._get_threshold_of_largest_subcube()]
        # relative_coord = MaskCube._compute_relative_coord(
        #     data_refpoint, clump_refpoint, original_shape
        # )
        relative_coord = self._pixel_coordinate_in_subcube(clump_refpoint)
        clump = self._data[
            relative_coord[0] : relative_coord[0] + clump_mask.shape[0],
            relative_coord[1] : relative_coord[1] + clump_mask.shape[1],
            relative_coord[2] : relative_coord[2] + clump_mask.shape[2],
        ]
        clump = clump * clump_mask
        # core will has the same shape as clump which is used to find the core
        if "pixel_length" not in kargs:
            pixel_length = self.phyinfo["pixel_size"]
        else:
            pixel_length = kargs["pixel_length"]
            kargs.pop("pixel_length")  # avoid the repeated keyword
        core_mask = MaskCube.get_fixed_mass_core(
            clump, -target_mass, pixel_length=pixel_length, **kargs
        )
        if isinstance(core_mask, str):
            print("Warning: The core mask is not found. Reason:")
            print(core_mask)
        elif isinstance(core_mask, np.ndarray):
            self._update_mask(target_mass, core_mask, clump_refpoint)
        else:
            raise ValueError("How is it possible?")
    
    # ! unfinished TODO
    def get_previous_structure(self, threshold = -2, time_step = 4):
        maskcube = super().get_previous_structure(threshold, time_step)
        maskcube.find_core(target_mass=threshold)  #! need to think about the low mass clump
        data, ROI, mask = maskcube.data(threshold=threshold, return_data_type="subcube_roi_mask")
        extra_data = {}
        corecube = CoreCube(data, extra_data,ROI, 
                            {threshold: mask}, 
                            {threshold: maskcube.refpoints[threshold]},
                            internal_id=maskcube.internal_id,
                            snapshot=maskcube.snapshot,
                            phyinfo=maskcube.phyinfo,
                            file_load_path=maskcube.file_load_path,
                            original_shape=maskcube.original_shape)
        return corecube

    def update_data_mask(
        self, new_data, new_extra_data, new_ROI, new_threshold, new_mask, new_refpoint
    ):
        for key in new_extra_data.keys():
            if new_extra_data[key].shape != new_data.shape:
                raise ValueError(
                    f"Shape of new data and new extra data should be the same. {key} is not the same."
                )
        self._extra_data = new_extra_data

        super().update_data_mask(
            new_data, new_ROI, new_threshold, new_mask, new_refpoint
        )

    def dump(self, file_path):
        """
        Dump the CoreCube instance to a pickle file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

# test the class
if __name__ == "__main__":

    # data = np.random.rand(3, 3, 4)
    # mask = np.ones((3, 3, 4), dtype=bool)
    # cube = DataCube(data, mask, phyinfo={"pixel_size": 0.005, "boundary": "periodic"})
    # print("==== Test datacube ====")
    # print(cube)
    # cube.info()
    # # since mask is mutable, the class will be changed if we change the mask element
    # print("==== Test datacube after modify an element of mask ====")
    # mask[0, 0, 0] = False
    # cube.info()
    # # but if we create a same named variable, the class will not be changed
    # print("==== Test datacube after asign new value for mask ====")
    # mask = np.zeros((3, 3, 4), dtype=bool)
    # cube.info()
    # # above point is a little bit dangerous, we should avoid this kind of operation
    # in the future
    # # we can use the class to change the mask
    # print("==== Test datacube after change mask using class method ====")
    # # or cube._mask = np.zeros((3, 3, 4), dtype=bool)
    # cube._mask[0, 0, 0] = True
    # cube.info()
    # print("==== Test maskcube ====")
    # cube2 = MaskCube(
    #     data,
    #     mask,
    #     {1: mask},
    #     {1: [0, 0, 0]},
    #     0,
    #     file_load_path="test.h5",
    #     phyinfo={"pixel_size": 0.005, "boundary": "periodic", "length_unit": "pc"},
    # )
    # print(cube2)
    # cube2.info()

    # # --- test the method of MaskCube ---
    # original = np.arange(5 * 5 * 5).reshape(5, 5, 5)
    # print(original)
    # original_ = np.roll(original, 0, axis=0)
    # original_ = np.roll(original_, 0, axis=1)
    # original_ = np.roll(original_, -1, axis=2)
    # data = original_[0:4, 0:2, 0:3]
    # ROI = np.ones_like(data, dtype=bool)
    # masks = {1: np.ones_like(data, dtype=bool)}
    # refpoints = {1: (0, 0, 1)}
    # maskcube = MaskCube(
    #     data,
    #     ROI,
    #     masks,
    #     refpoints,
    #     internal_id=0,
    #     snapshot=40,
    #     file_load_path="test.h5",
    #     phyinfo={"pixel_size": 0.005,
    #              "boundary": "periodic", "length_unit": "pc"},
    #     original_shape=(5, 5, 5),
    # )
    # maskcube.info()
    # ## -------- the pixel coordinate in the subcube --------
    # print(maskcube._data)
    # print(maskcube._in_largest_subcube((3, 0, 3)))
    # print(original[(3, 0, 3)])
    # # generate a series of random coordinates
    # np.random.seed(0)
    # coords = np.random.randint(0, 5, (1000, 3))
    # # print(coords)
    # for coord in coords:
    #     coord = tuple(coord)
    #     print(
    #         "Two methods equal: ",
    #         maskcube._in_largest_subcube(coord)
    #         == MaskCube._in_subcube(coord, (0, 0, 1), data.shape, (5, 5, 5)),
    #     )
    #     if maskcube._in_largest_subcube(coord):
    #         new_coord = maskcube._pixel_coordinate_in_subcube(coord)
    #         # print(new_coord)
    #         # print(maskcube._data[new_coord])
    #         print(original[coord] == maskcube._data[new_coord])

    # -------- the new subcube update method--------
    # np.random.seed(10)
    # coords = np.random.randint(0, 5, (1000, 3))
    # sizes = np.random.randint(1, 6, (1000, 3))
    # print(
    #     "Current largest subcube: ",
    #     maskcube.refpoints[maskcube._get_threshold_of_largest_subcube()],
    #     maskcube.masks[maskcube._get_threshold_of_largest_subcube()].shape,
    # )
    # for coord, size in zip(coords, sizes):
    #     # coord = (0,0,1)
    #     # size = (1,1,2)
    #     coord = tuple(coord)
    #     size = tuple(size)
    #     if maskcube._covered_by_new_subcube(coord, size):
    #         print("Covered by new subcube: ", np.array(coord), np.array(size))
    #         temp = MaskCube.get_subcube_from_rawcube(coord, size, original)
    #         relative_coord = MaskCube._compute_relative_coord(
    #             coord,
    #             maskcube.refpoints[maskcube._get_threshold_of_largest_subcube()],
    #             (5, 5, 5),
    #         )
    #         temp_ = temp[
    #             relative_coord[0]: relative_coord[0] + maskcube.shape[0],
    #             relative_coord[1]: relative_coord[1] + maskcube.shape[1],
    #             relative_coord[2]: relative_coord[2] + maskcube.shape[2],
    #         ]
    #         print((maskcube._data == temp_).all())
    #     if maskcube._cover_new_subcube(coord, size):
    #         print("Cover new subcube: ", np.array(coord), np.array(size))
    #         temp = MaskCube.get_subcube_from_rawcube(coord, size, original)
    #         relative_coord = maskcube._pixel_coordinate_in_subcube(
    #             coord)
    #         temp_ = maskcube._data[
    #             relative_coord[0]: relative_coord[0] + size[0],
    #             relative_coord[1]: relative_coord[1] + size[1],
    #             relative_coord[2]: relative_coord[2] + size[2],
    #         ]
    #         print((temp == temp_).all())
    #         # pass
    #     else:
    #         # print("Third: ", np.array(coord), np.array(size))
    #         pass

    # ------- dump and load -------
    # maskcube.dump("test.pkl")
    with open(
        "/data/shibo/CoresProject/seed1234/clump_core_data/clump_core_snap045_id001.pickle",
        "rb",
    ) as f:
        maskcube_: MaskCube = pickle.load(f)
    print(maskcube_)
    maskcube_.info()
    # print(maskcube_.data(1, return_data_type="density_roi_mask"))
    print(maskcube_.geometry_info())
    with h5py.File(f"{maskcube_.file_load_path}", "r") as f:
        den = f["gas_density"][...].T
        vx = f["i_velocity"][...].T
        vy = f["j_velocity"][...].T
        vz = f["k_velocity"][...].T

    tempa = maskcube_.get_previous_structure(-2, vx, vy, vz)
    # tempa = maskcube_.get_previous_structure2(-2)

    tempa.dump("testa.pkl")

    with open("testa.pkl", "rb") as f:
        maskcube_a: MaskCube = pickle.load(f)
    print(maskcube_a)
    maskcube_a.info()
