import pickle
import os
import numpy as np
from collections import deque, defaultdict
from .core_finder import MaskCube



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
        self.track = track  # [(snap, coreID), ...]

    def __str__(self):
        return f"CoreTrack (snap, ID): {self.track}"

    def __repr__(self):
        return f"CoreTrack (snap, ID): {self.track}"

    def __contains__(self, item):
        return item in self.track

    def __iter__(self):
        return iter(self.track)

    def get_file_list(self, directory: str) -> list[str]:
        # this is based on the naming convention of the files
        # clump_core_snap{snap:03d}_id{coreID:03d}.pickle
        #
        file_list = []
        for snap, coreID in self.track:
            if isinstance(coreID, int):
                file_list.append(
                    f"{directory}/clump_core_snap{snap:03d}_id{coreID:03d}.pickle"
                )
        # verify that all files exist
        for file in file_list:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} not found")
        return file_list

    def get_cores(self, directory: str) -> list["MaskCube"]:
        cores = []
        for snap, coreID in self.track:
            if coreID != 0 and isinstance(coreID, int):
                with open(
                    f"{directory}/clump_core_snap{snap:03d}_id{coreID:03d}.pickle", "rb"
                ) as f:
                    core = pickle.load(f)
                cores.append(core)
            else:
                cores.append(MaskCube(
                    np.zeros((1, 1, 1)),
                    np.ones((1, 1, 1), dtype=bool),
                    {0: np.ones((1, 1, 1), dtype=bool)},
                    {0: (0, 0, 0)},
                    internal_id=0,
                    snapshot=snap,
                    file_load_path="",
                ))
        return cores

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

    a = overlaps2tracks(overlaps, (20, 1))[0]
    print(a)
    print(a.get_cores(file_dir))
