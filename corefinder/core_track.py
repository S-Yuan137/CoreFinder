import pickle
import os
import numpy as np
from collections import deque, defaultdict
from .core_finder import MaskCube, CoreCube


def is_moving(next_overlap_tuple: tuple) -> int | None:
    """
    Gives the next core ID if the core is moving to another position.

    - displace/translocate:
    ```plaintext
        overlap_components_in_v2: [0, tag_next] or [tag_next]
        overlap_ratio_over_v1: [f0, f1] or [f1], where f1 > 0.5
        overlap_ratio_over_v2: [f0, f1] or [f1], where f1 > 0.5
        other tags in `overlap_components_in_v2` must be negligible
        overlap except tag_next.
    ```

    Parameters
    ----------
    next_overlap_tuple : tuple
        (new_coreID, ratio_1, ratio_2)

    Returns
    -------
    next_index : int | None
        The next core ID if the core is moving to another position, otherwise None.
    """
    # overlap tuple is (next_indices, ratio_1, ratio_2)
    next_index = next_overlap_tuple[0]
    ratio_1 = next_overlap_tuple[1]
    ratio_2 = next_overlap_tuple[2]

    if ratio_1.max() > 0.5:
        if ratio_2[ratio_1.argmax()] > 0.5:
            return next_index[ratio_1.argmax()]
        else:
            return None
    else:
        return None


def is_disappearing(next_overlap_tuple: tuple) -> int | None:
    """
    Gives the next core ID if the core is disappearing (to background).

    - disappear/dissipate/dissolve:
    ```plaintext
        overlap_components_in_v2: [0] or [0, tag_next]
        overlap_ratio_over_v1: [f0] or [f0, f1], where f0 ~= 1, f1 < 0.2
        overlap_ratio_over_v2: [f0] or [f0, f1], where f0 ~= 0, f1 < 0.2
        other tags in `overlap_components_in_v2` must be negligible
        overlap besides tag_next.
        That is, all the tags in `overlap_components_in_v2` are negligible
        overlap except tag 0.
    ```
    Parameters
    ----------
    next_overlap_tuple : tuple
        (new_coreID, ratio_1, ratio_2)

    Returns
    -------
    next_index : int | None
        The core ID 0 if the core is disappearing, otherwise None.
    """
    # overlap tuple is (next_indices, ratio_1, ratio_2)
    next_index = next_overlap_tuple[0]
    ratio_1 = next_overlap_tuple[1]
    ratio_2 = next_overlap_tuple[2]
    # background is 0 index and only appears once
    bg = np.where(next_index == 0)[0]
    if bg.size == 0:
        return None
    else:
        if ratio_1[bg] > 0.8:
            # all components in ratio_2 should be less than 0.2
            if np.all(ratio_2 < 0.2):
                return 0
            else:
                return None
        else:
            return None


def is_expand(next_overlap_tuple: tuple) -> int | None:
    """
    Gives the next core ID if the core is expanding.

    - expand:
    ```plaintext
        overlap_components_in_v2: [0, tag_next] or [tag_next]
        overlap_ratio_over_v1: [f0, f1] or [f1], where f1 > 0.5
        overlap_ratio_over_v2: [f0, f1] or [f1], where f1 < 0.5
        There might be other tags in `overlap_components_in_v2` with
        negligible overlap.
    ```
    Parameters
    ----------
    next_overlap_tuple : tuple
        (new_coreID, ratio_1, ratio_2)

    Returns
    -------
    next_index : int | None
        The next core ID if the core is expanding, otherwise None.
    """

    # overlap tuple is (next_indices, ratio_1, ratio_2)
    next_index = next_overlap_tuple[0]
    ratio_1 = next_overlap_tuple[1]
    ratio_2 = next_overlap_tuple[2]

    if ratio_1.max() > 0.5 and ratio_1.max() < 0.8:
        # if ratio_1.max() > 0.5 and ratio_1.max() < 0.8:
        if ratio_2[ratio_1.argmax()] < 0.5 and ratio_2[ratio_1.argmax()] > 0.2:
            return next_index[ratio_1.argmax()]
        else:
            return None
    else:
        return None


def is_collapse(next_overlap_tuple: tuple) -> int | None:
    """
    Gives the next core ID if the core is collapsing.

    - collapse/shrink/contract:
    ```plaintext
        overlap_components_in_v2: [0, tag_next] or [tag_next]
        overlap_ratio_over_v1: [f0, f1] or [f1], where f1 < 0.5
        overlap_ratio_over_v2: [f0, f1] or [f1], where f1 > 0.5
        There might be other tags in `overlap_components_in_v2` with
        negligible overlap.
    ```

    Parameters
    ----------
    next_overlap_tuple : tuple
        (new_coreID, ratio_1, ratio_2)

    Returns
    -------
    next_index : int | None
        The next core ID if the core is collapsing, otherwise None.
    """

    # overlap tuple is (next_indices, ratio_1, ratio_2)
    next_index = next_overlap_tuple[0]
    ratio_1 = next_overlap_tuple[1]
    ratio_2 = next_overlap_tuple[2]

    if ratio_1.max() < 0.5:
        if ratio_2[ratio_1.argmax()] > 0.5:
            return next_index[ratio_1.argmax()]
        else:
            return None
    else:
        return None


def is_merge(next_overlap_tuple: tuple) -> int | None:
    """
    Gives the next core ID if the core is merging.

    - merge:
    ```plaintext
        candidate 1 is expanded, candidate 2 is expanded
        where two candidates' tag_next must be the same. Candidates can be
        more than 2.
    ```
    Parameters
    ----------
    next_overlap_tuple : tuple
        (new_coreID, ratio_1, ratio_2)

    Returns
    -------
    next_index : int | None
        The next core ID if the core is merging, otherwise None.
    """
    pass


def is_split(next_overlap_tuple: tuple) -> int | None:
    """
    Gives the next core ID if the core is splitting.

    - split/segment/fragement:
    ```plaintext
        overlap_components_in_v2: [0, tag_next1, tag_next2] or [tag_next1,
        tag_next2]
        overlap_ratio_over_v1: [f0, f1, f2] or [f1, f2], where f0 < 0.3,
        f1 > 0.2, f2 > 0.2
        overlap_ratio_over_v2: [f0, f1, f2] or [f1, f2], where f0 < 0.3,
        f1 > 0.2, f2 > 0.2
        where tag_next1 != tag_next2 and they can be more than 2.
    ```

    Parameters
    ----------
    next_overlap_tuple : tuple
        (new_coreID, ratio_1, ratio_2)

    Returns
    -------
    next_index : int | None
        The next core ID if the core is splitting, otherwise None.
    """
    pass



def periodic_coord_set(point1: np.ndarray, point2: np.ndarray,
                       original_size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    def get_short_array(arr1, arr2):
        if len(arr1) <= len(arr2):
            return arr1
        else:
            return arr2
        
    # convert to periodic coordinates first
    point1 = point1 % original_size
    point2 = point2 % original_size
    lower = np.minimum(point1, point2)
    upper = np.maximum(point1, point2)
    direct_x = np.arange(lower[0], upper[0]+1)
    direct_y = np.arange(lower[1], upper[1]+1)
    direct_z = np.arange(lower[2], upper[2]+1)
    cross = []
    for i in range(3):
        cross_axis = []
        cross_axis.extend(list(np.arange(0, lower[i]+1)))
        cross_axis.extend(list(np.arange(upper[i], original_size[i])))
        cross.append(np.array(cross_axis))
    cross_x, cross_y, cross_z = cross 
    out_x = get_short_array(direct_x, cross_x)
    out_y = get_short_array(direct_y, cross_y)
    out_z = get_short_array(direct_z, cross_z)
    return out_x, out_y, out_z


def compute_pixel_range(lower_x: np.ndarray, lower_y: np.ndarray, lower_z: np.ndarray,
                        upper_x: np.ndarray, upper_y: np.ndarray, upper_z: np.ndarray,
                        original_size: tuple[int, int, int]):
    """
    Compute the bounded box of multiple boxes in periodic 3D, which is defined by the lower-left and
    upper-right coordinates.
    """
    # build all the possible point pairs from the lower-left and upper-right coordinates
    point_pairs = []
    for i in range(len(lower_x)):
        for j in range(len(upper_x)):
            point_pairs.append((np.array([lower_x[i], lower_y[i], lower_z[i]]), 
                                np.array([upper_x[j], upper_y[j], upper_z[j]])))
    out_x, out_y, out_z = set(), set(), set()
    for point1, point2 in point_pairs:
        temp_x, temp_y, temp_z = periodic_coord_set(point1, point2, original_size)
        out_x.update(temp_x)
        out_y.update(temp_y)
        out_z.update(temp_z)
    out_x = np.array(sorted(out_x))
    out_y = np.array(sorted(out_y))
    out_z = np.array(sorted(out_z))
    out_lower_x = np.min(out_x)
    out_upper_x = np.max(out_x)
    out_lower_y = np.min(out_y)
    out_upper_y = np.max(out_y)
    out_lower_z = np.min(out_z)
    out_upper_z = np.max(out_z)
    if out_lower_x == 0 and out_upper_x == original_size[0] - 1:
        lost_x = np.setdiff1d(np.arange(original_size[0]), out_x)
        if len(lost_x) > 0:
            out_lower_x = np.max(lost_x)+1
            out_upper_x = np.min(lost_x)-1 + original_size[0]
    if out_lower_y == 0 and out_upper_y == original_size[1] - 1:
        lost_y = np.setdiff1d(np.arange(original_size[1]), out_y)
        if len(lost_y) > 0:
            out_lower_y = np.max(lost_y)+1
            out_upper_y = np.min(lost_y)-1 + original_size[1]
    if out_lower_z == 0 and out_upper_z == original_size[2] - 1:
        lost_z = np.setdiff1d(np.arange(original_size[2]), out_z)
        if len(lost_z) > 0:
            out_lower_z = np.max(lost_z)+1
            out_upper_z = np.min(lost_z)-1 + original_size[2]
    return (out_lower_x, out_lower_y, out_lower_z,
            out_upper_x, out_upper_y, out_upper_z)
    


def get_bound_box_per_snap(
    corelist: list["MaskCube"] | list["CoreCube"], threshold: float = 17.682717 * 30,
    original_size: tuple[int, int, int] = (960, 960, 960)
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Get the motion coordinates of Cubes lower-left and upper-right coordinates in 3D.
    
    Parameters
    ----------
    corelist : list[MaskCube] | list[CoreCube]
        The list of MaskCube or CoreCube objects.
    threshold : float, optional
        The threshold value, by default 17.682717 * 30
    
    Returns
    -------
    tuple[list[int], list[int], list[int], list[int], list[int], list[int]]
        The motion coordinates of lower-left and upper-right coordinates in 3D.
        (lower_x, lower_y, lower_z, upper_x, upper_y, upper_z)
    """
    # sort the corelist by snapshot
    corelist.sort(key=lambda x: x.snapshot)
    unique_snaps = np.unique([core.snapshot for core in corelist])
    
    
    list_len = len(unique_snaps)
    lower_x = np.zeros(list_len)
    lower_y = np.zeros(list_len)
    lower_z = np.zeros(list_len)
    upper_x = np.zeros(list_len)
    upper_y = np.zeros(list_len)
    upper_z = np.zeros(list_len)
    for i, snap in enumerate(unique_snaps):
        cores_in_snap = [core for core in corelist if core.snapshot == snap]
        if len(cores_in_snap) == 0:
            raise ValueError(f"No cores found in snapshot {snap}.")
        elif len(cores_in_snap) == 1:
            core = cores_in_snap[0]
            lower_x[i], lower_y[i], lower_z[i] = core.refpoints[threshold]
            size_x, size_y, size_z = core.masks[threshold].shape
            upper_x[i] = lower_x[i] + size_x 
            upper_y[i] = lower_y[i] + size_y
            upper_z[i] = lower_z[i] + size_z 
        elif len(cores_in_snap) > 1:
            temp_lower_x = np.array([core.refpoints[threshold][0] for core in cores_in_snap])
            temp_lower_y = np.array([core.refpoints[threshold][1] for core in cores_in_snap])
            temp_lower_z = np.array([core.refpoints[threshold][2] for core in cores_in_snap])
            temp_upper_x = np.array([core.refpoints[threshold][0] + core.masks[threshold].shape[0] for core in cores_in_snap])
            temp_upper_y = np.array([core.refpoints[threshold][1] + core.masks[threshold].shape[1] for core in cores_in_snap])
            temp_upper_z = np.array([core.refpoints[threshold][2] + core.masks[threshold].shape[2] for core in cores_in_snap])
            # here is a complex logic to find the lower and upper coordinates
            lower_x[i], lower_y[i], lower_z[i], \
            upper_x[i], upper_y[i], upper_z[i] = compute_pixel_range(
                temp_lower_x, temp_lower_y, temp_lower_z,
                temp_upper_x, temp_upper_y, temp_upper_z,
                original_size=original_size
            )
    size_x = upper_x - lower_x
    size_y = upper_y - lower_y
    size_z = upper_z - lower_z
    return (size_x, size_y, size_z), (lower_x, lower_y, lower_z), unique_snaps


class CoreTrack:
    def __init__(self, track: list[tuple[int, int]]) -> None:
        # sort the track by the snap and id
        # [(snap, coreID), ...]
        self.track = sorted(track, key=lambda x: (x[0], x[1]))

    def __str__(self):
        return f"CoreTrack (snap, ID): {self.track}"

    def __repr__(self):
        return f"CoreTrack (snap, ID): {self.track}"

    def __contains__(self, item):
        return item in self.track

    def __iter__(self):
        return iter(self.track)

    def get_file_list(self, directory: str, file_name_format: str) -> list[str]:
        """get a list of file names based on the track

        Parameters
        ----------
        directory : str
            The directory where the files are stored.
        file_name_format : str
            The format of the file name. It should contain the format string
            for snap and coreID, e.g., "core_snap{snap:03d}_id{coreID:03d}.pickle"

        Returns
        -------
        list[str]
            The list of file names.

        Raises
        ------
        FileNotFoundError
            If the file is not found.
        """
        
        # this is based on the naming convention of the files
        # clump_core_snap{snap:03d}_id{coreID:03d}.pickle
        #
        file_list = []
        for snap, coreID in self.track:
            if isinstance(coreID, int):
                if coreID == 0:  # background, skip, normally it should be the last one in a track
                    break
                formatted_file_name = file_name_format.format(snap=snap, coreID=coreID)
                file_list.append(
                    f"{directory}/{formatted_file_name}"
                )
        # verify that all files exist
        for file in file_list:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} not found")
        return file_list

    def get_cores(self, directory: str, file_name_format: str) -> list["MaskCube"] | list["CoreCube"]:
        """
        load the cores from the directory

        Returns
        -------
        cores : list[MaskCube]
            The list of MaskCube objects
        """
        cores = []
        file_list = self.get_file_list(directory, file_name_format)
        for file in file_list:
            with open(file, "rb") as f:
                core = pickle.load(f)
            cores.append(core)
        return cores

    def get_filled_canvas3d_list_float_position(
        self, coreslist: list["MaskCube"] | list["CoreCube"], threshold: float = 17.682717 * 30
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
        """
        Fill in the canvas with the data (masked_density in MaskCube list), where the
        positions of canvas in each snap are float (not fixed).

        Parameters
        ----------
        coreslist : list[MaskCube], optional
            The list of MaskCube objects.
        threshold : float, optional
            The threshold value, by default 17.682717 * 30

        Returns
        -------
        canvas3d_list: list[np.ndarray]
            The list of filled canvas in 3D.
        refpoints: list[tuple[int, int, int]]
            The most lower-left point cooridinate of the canvas in 3D.
        """
        original_size = coreslist[0].original_shape
        canvas_sizes, canvas_refs, unique_snaps = get_bound_box_per_snap(
            coreslist, threshold=threshold, original_size=original_size
        )
        canvas3d_list = []
        canvas3d_refs = []
        for i, snap in enumerate(unique_snaps):
            canvas_ref = (canvas_refs[0][i], canvas_refs[1][i], canvas_refs[2][i])
            canvas_size = (canvas_sizes[0][i], canvas_sizes[1][i], canvas_sizes[2][i])
            canvas = np.zeros(canvas_size, dtype=np.float32)
            for core in coreslist:
                if core.snapshot == snap:
                    # get the mask and refpoint for this core
                    data = core.data(threshold=threshold,return_data_type="masked")
                    refpoint = core.refpoints[threshold]
                    # fill the canvas with the masked density
                    canvas[
                        refpoint[0] - canvas_ref[0]:refpoint[0] - canvas_ref[0] + data.shape[0],
                        refpoint[1] - canvas_ref[1]:refpoint[1] - canvas_ref[1] + data.shape[1],
                        refpoint[2] - canvas_ref[2]:refpoint[2] - canvas_ref[2] + data.shape[2]
                    ] += data
            canvas3d_list.append(canvas)
            canvas3d_refs.append(canvas_ref)
        return canvas3d_list, canvas3d_refs
        

    def get_filled_canvas3d_list(
        self, coreslist: list["MaskCube"] = None, threshold: float = 17.682717 * 30
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
        """
        Fill in the canvas with the data (masked_density in MaskCube list), where the
        positions of canvas in snaps have correct relative distances.

        Parameters
        ----------
        coreslist : list[MaskCube], optional
            The list of MaskCube objects, by default None. However, this should be
            provided if the MaskCube objects are not loaded from the directory.
            And for computation efficiency, it is recommended to provide the MaskCube.
        threshold : float, optional
            The threshold value, by default 17.682717 * 30

        Returns
        -------
        canvas3d_list: list[np.ndarray]
            The list of filled canvas in 3D.
        refpoints: list[tuple[int, int, int]]
            The most lower-left point cooridinate of the canvas in 3D.
        """
        canvs3d_list, refpoints = self.get_filled_canvas3d_list_float_position(
            coreslist, threshold
        )
        
        def get_bounding_canvas_size(
            canvases: list[np.ndarray], refpoints: list[tuple[int, int, int]],
            original_size: tuple[int, int, int] = (960, 960, 960)
        ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
            """
            Get the size of the bounding canvas for plotting
            """
            # get the minimum x, y, z
            min_xs = np.array([ref[0] for ref in refpoints])
            min_ys = np.array([ref[1] for ref in refpoints])
            min_zs = np.array([ref[2] for ref in refpoints])

            # check canvas size not exceed the original size
            temp_x_sizes = np.array([canvas.shape[0] for canvas in canvases])
            temp_y_sizes = np.array([canvas.shape[1] for canvas in canvases])
            temp_z_sizes = np.array([canvas.shape[2] for canvas in canvases])
            
            if (temp_x_sizes >= original_size[0]).any():
                print("Warning: canvas size exceeds the original size in x")
            if (temp_y_sizes >= original_size[1]).any():
                print("Warning: canvas size exceeds the original size in y")
            if (temp_z_sizes >= original_size[2]).any():
                print("Warning: canvas size exceeds the original size in z")
            
            # make sure min_x, min_y, min_z are within the original size
            min_xs[min_xs >= original_size[0]] -= original_size[0]
            min_ys[min_ys >= original_size[1]] -= original_size[1]
            min_zs[min_zs >= original_size[2]] -= original_size[2]
            max_xs = min_xs + np.array([canvas.shape[0] for canvas in canvases])
            max_ys = min_ys + np.array([canvas.shape[1] for canvas in canvases])
            max_zs = min_zs + np.array([canvas.shape[2] for canvas in canvases])
            
            temp_min_x = min_xs.min()
            temp_min_y = min_ys.min()
            temp_min_z = min_zs.min()
            temp_max_x = max_xs.max()
            temp_max_y = max_ys.max()
            temp_max_z = max_zs.max()
            
            
            # refpoints of the most lower-left point
            # find the element >=960  and subtract 960 for them
            min_x =  np.array([ref[0] for ref in refpoints])[min_xs.argmin()]
            min_y =  np.array([ref[1] for ref in refpoints])[min_ys.argmin()]
            min_z =  np.array([ref[2] for ref in refpoints])[min_zs.argmin()]
            bounding_x = temp_max_x - temp_min_x
            bounding_y = temp_max_y - temp_min_y
            bounding_z = temp_max_z - temp_min_z
            print(f"min_x: {min_x}, min_y: {min_y}, min_z: {min_z}")
            print(f"bounding_x: {bounding_x}, bounding_y: {bounding_y}, bounding_z: {bounding_z}")
            return (bounding_x, bounding_y, bounding_z), (min_x, min_y, min_z)

        bc_size, min_ref_bc = get_bounding_canvas_size(canvs3d_list, refpoints)
        fixed_position_canvs3d_list = []
        for i, canvas in enumerate(canvs3d_list):
            temp_canvas = np.zeros(bc_size)
            temp_canvas[
                refpoints[i][0]
                - min_ref_bc[0] : refpoints[i][0]
                - min_ref_bc[0]
                + canvas.shape[0],
                refpoints[i][1]
                - min_ref_bc[1] : refpoints[i][1]
                - min_ref_bc[1]
                + canvas.shape[1],
                refpoints[i][2]
                - min_ref_bc[2] : refpoints[i][2]
                - min_ref_bc[2]
                + canvas.shape[2],
            ] = canvas
            fixed_position_canvs3d_list.append(temp_canvas)

        return fixed_position_canvs3d_list

    def get_filled_canvas2d_list(
        self,
        coreslist: list["MaskCube"] = None,
        threshold: float = 17.682717 * 30,
        LOS_direction=(1, 0, 0),
    ) -> list[np.ndarray]:
        """
        Fill in the canvas with the data (masked_density in MaskCube list) for 2D

        Parameters
        ----------
        coreslist : list[&quot;MaskCube&quot;], optional
            The list of MaskCube objects, by default None. However, this should be
            provided if the MaskCube objects are not loaded from the directory.
            And for computation efficiency, it is recommended to provide the MaskCube.
        threshold : float, optional
            The threshold value, by default 17.682717 * 30
        LOS_direction : tuple[int, int, int], optional
            The line of sight direction, by default (1, 0, 0).

        Returns
        -------
        canvas2d: list[np.ndarray]
            The filled canvas in 2D.
        """
        pass

    def add_core(self, snap_coreID: tuple[int, int]) -> None:
        self.track.append(snap_coreID)

    def get_random_core(self) -> "MaskCube":
        pass

    def dump(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(self, f)


class OverLap:
    def __init__(self, snap: int, files_path: str) -> None:
        self.snap = snap
        self.files_path = files_path
        self.overlap = self.load_overlap()

    def load_overlap(
        self,
    ) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        with open(self.files_path, "rb") as f:
            overlap = pickle.load(f)
        return overlap

    def filter_overlap(self, negligible_ratio: float = 0.01) -> None:
        filtered_overlap = {}
        for CoreID, overlap_tuple in self.overlap.items():
            next_ID = overlap_tuple[0]
            ratio_1 = overlap_tuple[1]
            ratio_2 = overlap_tuple[2]
            ratio_1[ratio_1 <= negligible_ratio] = 0
            ratio_2[ratio_2 <= negligible_ratio] = 0
            negligible_indices = np.where((ratio_1 == 0) & (ratio_2 == 0))[0]
            next_ID = np.delete(next_ID, negligible_indices)
            ratio_1 = np.delete(ratio_1, negligible_indices)
            ratio_2 = np.delete(ratio_2, negligible_indices)
            filtered_overlap[CoreID] = (next_ID, ratio_1, ratio_2)
        self.overlap = filtered_overlap

    def get_next_core(self, coreID_window: int = 20) -> dict[int, tuple[int]]:
        next_core = {}
        for CoreID, overlap_tuple in self.overlap.items():
            if CoreID <= coreID_window:
                next_index = overlap_tuple[0]
                ratio_1 = overlap_tuple[1]
                if len(next_index) == 1 and next_index <= coreID_window:
                    next_core[CoreID] = tuple(next_index.tolist())
                elif len(next_index) > 1:
                    # find the index of 0 in the next core ID
                    bg = np.where(next_index == 0)[0]
                    if ratio_1[bg] > 0.8:
                        next_core[CoreID] = 0
                    else:
                        temp_less = next_index <= coreID_window
                        if np.all(temp_less):
                            next_core[CoreID] = tuple(
                                np.delete(next_index, bg).tolist()
                            )
                        elif (ratio_1[temp_less]).sum() > 0.8:
                            # delete the over window core ID
                            next_core[CoreID] = tuple(
                                np.delete(next_index[temp_less], bg).tolist()
                            )
                        else:
                            continue
        return next_core

    def get_previous_core(self) -> dict[int, int | tuple[int]]:
        pass


def overlaps2tracks(
    overlaps: list["OverLap"], passing_node: tuple[int, int] = None
) -> list["CoreTrack"]:
    """
    Convert a list of OverLap objects to a list of CoreTrack objects.
    BFS algorithm is used to find the tracks.

    Parameters
    ----------
    overlaps : list[OverLap]
        A list of OverLap objects.

    passing_node : tuple[int, int], optional
        The node that the track must pass through, by default None.

    Returns
    -------
    track : list[CoreTrack]
        A list of CoreTrack objects. Each CoreTrack object represents a track.
        If passing_node is not None, then the list contains all the tracks that
        pass through the passing_node.
    """
    mappings = {}
    for overlap in overlaps:
        overlap.filter_overlap(0.01)
        snap = overlap.snap
        for key, value in overlap.get_next_core().items():
            if len(value) == 1:
                mappings[(snap, int(key))] = [(snap + 1, value[0])]
            elif len(value) > 1:
                mappings[(snap, int(key))] = [(snap + 1, int(v)) for v in value]
    # construct the graph
    graph = defaultdict(list)
    for key, value in mappings.items():
        for v in value:
            graph[key].append(v)
            graph[v].append(key)

    evolution_paths = []
    visited = set()
    for key in graph.keys():
        if key not in visited:
            path = []
            queue = deque([key])
            visited.add(key)
            while queue:
                node = queue.popleft()
                path.append(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)
            evolution_paths.append(CoreTrack(path))
    if passing_node is None:
        return evolution_paths
    else:
        passing_paths = []
        for path in evolution_paths:
            if passing_node in path:
                passing_paths.append(path)
        return passing_paths


def tracks_branch(
    overlaps: list["OverLap"], passing_node: tuple[int, int] = None
) -> dict[str, list["CoreTrack"]]:
    """
    Branch the tracks into different branches.

    Parameters
    ----------
    overlaps : list[OverLap]
        A list of OverLap objects.

    passing_node : tuple[int, int], optional
        The node that the track must pass through, by default None.

    Returns
    -------
    branches : dict[str, list[CoreTrack]]
        A dictionary of branches. The key is the branch name, and the value is
        a list of CoreTrack objects.
    """
    mappings = {}
    for overlap in overlaps:
        overlap.filter_overlap(0.01)
        snap = overlap.snap
        for key, value in overlap.get_next_core().items():
            if len(value) == 1:
                mappings[(snap, int(key))] = [(snap + 1, value[0])]
            elif len(value) > 1:
                mappings[(snap, int(key))] = [(snap + 1, int(v)) for v in value]

    clusters = overlaps2tracks(overlaps, passing_node)

    def is_continous_subset(a, b):
        # if longer one contains shorter one, return True
        # note the elements is ordered, ie, [1,2,3,4] not contain [1,3,4]
        if len(a) <= len(b):
            for i in range(len(b) - len(a) + 1):
                if a == b[i:i+len(a)]:
                    return True
        else:
            for i in range(len(a) - len(b) + 1):
                if b == a[i:i+len(b)]:
                    return True
        return False

    def dfs_chains(graph, node, visited, path, cluster_paths, recorded_paths):
        visited.add(node)
        path.append(node)

        # if the current node has no out-edges, it is a terminal
        if not graph[node]:
            current_path = tuple(path)
            if current_path not in recorded_paths:
                cluster_paths.append(list(path))
                recorded_paths.add(current_path)

        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs_chains(
                    graph, neighbor, visited, path, cluster_paths, recorded_paths
                )

        path.pop()
        visited.remove(node)

    # construct directed graph
    graph = defaultdict(list)
    for key, value in mappings.items():
        for v in value:
            graph[key].append(v)

    result = {}
    for i, cluster in enumerate(clusters):
        cluster_paths = []
        visited = set()
        recorded_paths = set()

        for node in cluster:
            if node not in visited:
                path = []
                dfs_chains(graph, node, visited, path, cluster_paths, recorded_paths)
        # filter out the redundant paths
        # sort lists by length, from longest to shortest
        cluster_paths = sorted(cluster_paths, key=lambda x: len(x), reverse=True)
        unique_branches = []
        for branch in cluster_paths:
            if not any([is_continous_subset(branch, unique_branch) for unique_branch in unique_branches]):
                unique_branches.append(branch)
        # ! Track the unique paths?
        # currently, [A, B, C, D] and [A, B, C, E] are considered as two different paths
        # this is because their terminal nodes are different

        result[f"cluster{i}"] = unique_branches

    return result


def get_clusters_branches(overlaps: list["OverLap"]) -> dict[str, list["CoreTrack"]]:
    """
    Get the clusters and branches from the overlaps.

    Parameters
    ----------
    overlaps : list[OverLap]
        A list of OverLap objects.

    Returns
    -------
    branches : dict[str, list[CoreTrack]]
        A dictionary of branches. The key is the branch name, and the value is
        a list of CoreTrack objects.
    """
    def analyze_mappings(mappings):
        # Step 1: Process mappings into children and parents
        children = defaultdict(list)
        parents = defaultdict(list)
        for key, value in mappings.items():
            if isinstance(value, list):
                vals = value
            else:
                vals = [value]
            children[key].extend(vals)
            for v in vals:
                parents[v].append(key)
        
        # Step 2: Collect all nodes
        nodes = set()
        nodes.update(children.keys())
        for vs in children.values():
            nodes.update(vs)
        nodes.update(parents.keys())
        for ps in parents.values():
            nodes.update(ps)
        nodes = list(nodes)
        
        # Step 3: Build undirected graph adjacency list
        undirected = defaultdict(set)
        for node in nodes:
            # Add children as neighbors
            for child in children.get(node, []):
                undirected[node].add(child)
                undirected[child].add(node)
            # Add parents as neighbors
            for parent in parents.get(node, []):
                undirected[node].add(parent)
                undirected[parent].add(node)
        
        # Step 4: Find connected components (clusters) using BFS
        visited = set()
        clusters = []
        for node in nodes:
            if node not in visited:
                queue = [node]
                visited.add(node)
                cluster = []
                while queue:
                    current = queue.pop(0)
                    cluster.append(current)
                    for neighbor in undirected[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                clusters.append(cluster)
        
        # Step 5 & 6: Process each cluster to find roots and paths
        result_clusters = {}
        result_branches_in_cluster = {}
        for i, cluster in enumerate(clusters):
            cluster_set = set(cluster)
            # Find root nodes in the cluster
            roots = []
            for node in cluster:
                is_root = True
                for parent in parents.get(node, []):
                    if parent in cluster_set:
                        is_root = False
                        break
                if is_root:
                    roots.append(node)
            # Generate paths from each root
            paths = []
            for root in roots:
                stack = [(root, [root])]
                while stack:
                    current_node, current_path = stack.pop()
                    # Get children in the cluster
                    children_in_cluster = []
                    for child in children.get(current_node, []):
                        if child in cluster_set:
                            children_in_cluster.append(child)
                    if not children_in_cluster:
                        paths.append(current_path)
                    else:
                        for child in reversed(children_in_cluster):  # To maintain order as per example
                            # Check if child is not already in path to avoid cycles
                            if child not in current_path:
                                stack.append((child, current_path + [child]))
            # Prepare cluster's nodes in sorted order for readability
            sorted_cluster = sorted(cluster)
            # Sort paths for consistent output
            sorted_paths = sorted(paths, key=lambda x: (len(x), x))
            # Format the result entry
            result_clusters[f"cluster{i}"] = sorted_cluster
            result_branches_in_cluster[f"branches_in_cluster{i}"] = sorted_paths
        
        return result_clusters, result_branches_in_cluster
    # Convert overlaps to mappings
    mappings = {}
    for overlap in overlaps:
        overlap.filter_overlap(0.01)
        snap = overlap.snap
        for key, value in overlap.get_next_core().items():
            if len(value) == 1:
                mappings[(snap, int(key))] = [(snap + 1, value[0])]
            elif len(value) > 1:
                mappings[(snap, int(key))] = [(snap + 1, int(v)) for v in value]
    # Analyze the mappings to get clusters and branches
    clusters, branches_in_cluster = analyze_mappings(mappings)
    return clusters, branches_in_cluster
    

if __name__ == "__main__":
    # ================= Test CoreTrack =================
    # track = [(20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 2)]
    # core_track = CoreTrack(track)
    # print(
    #     core_track.get_file_list("/data/shibo/CoresProject/seed1234/clump_core_data")
    # )

    # # ================= Test OverLap =================
    file_dir = "/data/shibo/CoresProject/seed1234/clump_core_data"
    overlaps = []
    for snap in range(20, 22):
        overlap = OverLap(
            snap,
            f"{file_dir}/thres30ini_overlap_result_downpixel_predict{snap}toreal{snap+1}.pickle",
        )
        # print(snap)
        overlap.filter_overlap(0.01)
        overlaps.append(overlap)

    a = overlaps2tracks(overlaps)
    print(a)
    # print(a.get_cores(file_dir))
    b = tracks_branch(overlaps)
    print(b)
