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
        self, coreslist: list["MaskCube"] | list["CoreCube"] = None, threshold: float = 17.682717 * 30
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
        """
        Fill in the canvas with the data (masked_density in MaskCube list), where the
        positions of canvas in each snap are float (not fixed).

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

        def get_canvas_size(
            refs: list[tuple[int, int, int]], sizes: list[tuple[int, int, int]],
            original_size: tuple[int, int, int] = (960, 960, 960)
        ) -> tuple[int, int, int]:
            """
            Get the size of the canvas for plotting
            """
            # get the maximum x and y
            max_x = max([ref[0] + size[0] for ref, size in zip(refs, sizes)])
            max_y = max([ref[1] + size[1] for ref, size in zip(refs, sizes)])
            max_z = max([ref[2] + size[2] for ref, size in zip(refs, sizes)])
            # get the minimum x, y, z
            min_x = min([ref[0] for ref in refs])
            min_y = min([ref[1] for ref in refs])
            min_z = min([ref[2] for ref in refs])

            # deal with the case where the canvas is larger than the original size
            if max_x - min_x >= original_size[0]:
                min_x += original_size[0]
            if max_y - min_y >= original_size[1]:
                min_y += original_size[1]
            if max_z - min_z >= original_size[2]:
                min_z += original_size[2]
            
            return max_x - min_x, max_y - min_y, max_z - min_z

        def fill_in_canvas(
            refs: list[tuple[int, int, int]],
            sizes: list[tuple[int, int, int]],
            datas: list[np.ndarray],
            canvas: np.ndarray,
            original_size: tuple[int, int, int] = (960, 960, 960),
        ):
            """
            Fill in the canvas with the data
            """
            # find the (0,0,0) point in the refs
            min_x = min([ref[0] for ref in refs])
            min_y = min([ref[1] for ref in refs])
            min_z = min([ref[2] for ref in refs])
            
            for ref, size, data in zip(refs, sizes, datas):
                x, y, z = ref
                
                start_x = x - min_x if x - min_x < original_size[0] else x - min_x - original_size[0]
                start_y = y - min_y if y - min_y < original_size[1] else y - min_y - original_size[1]
                start_z = z - min_z if z - min_z < original_size[2] else z - min_z - original_size[2]
                
                canvas[
                    start_x : start_x + size[0],
                    start_y : start_y + size[1],
                    start_z : start_z + size[2],
                ] = data
            return canvas, (min_x, min_y, min_z)

        snaps, ids = zip(*self.track)
        reduce_snaps, count = np.unique(snaps, return_counts=True)

        # the coretrack is sorted according to the snap and id by default
        # also sort the coreslist according to the snap and id
        # usually, the coreslist should have the same order as the track
        # but just in case, keep the following line
        coreslist = sorted(coreslist, key=lambda x: (x.snapshot, x.internal_id))

        ii = 0
        filled_canvas_list = []
        canvas_refpoints = []
        for idx, snap in enumerate(reduce_snaps):
            num_cross_clumps = count[idx]
            if num_cross_clumps == 1:
                filled_canvas_list.append(
                    coreslist[ii].data(threshold, return_data_type="masked")
                )
                canvas_refpoints.append(coreslist[ii].refpoints[threshold])
                ii += 1
            else:
                temp_refs = []
                temp_sizes = []
                temp_datas = []
                for _ in range(num_cross_clumps):
                    temp_refs.append(coreslist[ii].refpoints[threshold])
                    temp_sizes.append(coreslist[ii].masks[threshold].shape)
                    temp_datas.append(
                        coreslist[ii].data(threshold, return_data_type="masked")
                    )
                    ii += 1
                canvas = np.zeros(get_canvas_size(temp_refs, temp_sizes))
                canvas, new_ref = fill_in_canvas(
                    temp_refs, temp_sizes, temp_datas, canvas
                )
                filled_canvas_list.append(canvas)
                canvas_refpoints.append(new_ref)
        return filled_canvas_list, canvas_refpoints

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
            canvases: list[np.ndarray], refpoints: list[tuple[int, int, int]]
        ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
            """
            Get the size of the bounding canvas for plotting
            """
            # get the maximum x and y
            max_x = max(
                [ref[0] + canvas.shape[0] for ref, canvas in zip(refpoints, canvases)]
            )
            max_y = max(
                [ref[1] + canvas.shape[1] for ref, canvas in zip(refpoints, canvases)]
            )
            max_z = max(
                [ref[2] + canvas.shape[2] for ref, canvas in zip(refpoints, canvases)]
            )
            # get the minimum x, y, z
            min_x = min([ref[0] for ref in refpoints])
            min_y = min([ref[1] for ref in refpoints])
            min_z = min([ref[2] for ref in refpoints])

            return (max_x - min_x, max_y - min_y, max_z - min_z), (min_x, min_y, min_z)

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
