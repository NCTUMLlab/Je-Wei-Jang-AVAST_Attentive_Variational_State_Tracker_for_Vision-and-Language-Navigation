from collections import namedtuple
import numpy as np


WorldState = namedtuple(
    "WorldState",
    ["scanId", "location", "viewIndex", "heading", "elevation"]
)


Location = namedtuple(
    "Location",
    ['viewpointId']
)


NavigableLocation = namedtuple(
    "NavigableLocation",
    ["viewpointId", "ix", "rel_heading", "rel_elevation", "rel_distance"]
)


class MatterEnvDictBase():
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__()
        self.config = config

        self.rad30, self.rad360 = np.deg2rad(30), np.deg2rad(30) * 12
        self._state_infos = []
        self._view_indices, self._headings, self._elevations = None, None, None
        self._scan_ids, self._vp_ids = [], []
        return

    @property
    def state_infos(
        self
    ) -> list:
        return self._state_infos

    def set_state_infos(
        self,
        new_state_infos: list
    ) -> None:
        self._state_infos = new_state_infos
        return

    @property
    def view_indices(
        self
    ) -> np.ndarray:
        return self._view_indices

    def set_pose_by_view_index(
        self,
        new_view_indices: np.ndarray
    ) -> None:
        self._view_indices = new_view_indices
        self.set_headings((new_view_indices % 12) * self.rad30)
        self.set_elevations((new_view_indices // 12 - 1) * self.rad30)
        return

    @property
    def scan_ids(
        self
    ) -> list:
        return self._scan_ids

    def set_scan_ids(
        self,
        new_scan_ids: list
    ) -> None:
        self._scan_ids = new_scan_ids
        return

    @property
    def vp_ids(
        self
    ) -> list:
        return self._vp_ids

    def set_vp_ids(
        self,
        new_vp_ids: list
    ) -> None:
        self._vp_ids = new_vp_ids
        return

    @property
    def headings(
        self
    ) -> np.ndarray:
        return self._headings

    def set_headings(
        self,
        new_headings: np.ndarray
    ) -> None:
        self._headings = new_headings
        return

    @property
    def elevations(
        self
    ) -> np.ndarray:
        return self._elevations

    def set_elevations(
        self,
        new_elevations: np.ndarray
    ) -> None:
        self._elevations = new_elevations
        return


class MatterEnvDict(MatterEnvDictBase):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)
        # rendering init
        self.verbose = config['args']['verbose']
        self.rendering_idx = config['args']['rendering_idx']
        return

    def get_states(
        self
    ) -> list:
        return self.state_infos

    def discretize_heading_rad(
        self,
        rad: float
    ) -> float:
        return (round(rad / self.rad30) * self.rad30) % self.rad360

    def pose_to_view_index(
        self,
        heading: float,
        elevation: float
    ) -> int:
        return int((12 * round(elevation / self.rad30 + 1) + round(heading / self.rad30) % 12))

    def update_world_states(
        self
    ) -> None:
        for idx, (scan_id, vp_id) in enumerate(zip(self.scan_ids, self.vp_ids)):
            view_index = self.view_indices[idx]
            heading = self.headings[idx]
            elevation = self.elevations[idx]
            # update state_infos
            self.state_infos[idx] = WorldState(
                scanId=scan_id,
                location=Location(viewpointId=vp_id),
                viewIndex=view_index,
                heading=heading,
                elevation=elevation
            )
        return

    def new_episodes(
        self,
        scan_ids: list,
        vp_ids: list,
        headings: list,
        gen_gif: bool
    ) -> None:
        assert not gen_gif
        view_indices = []
        for heading in headings:
            discrete_heading = self.discretize_heading_rad(heading)
            view_indices.append(self.pose_to_view_index(discrete_heading, 0))

        # store current scan_ids, vp_ids, view_indices
        self.set_scan_ids(scan_ids)
        self.set_vp_ids(vp_ids)
        self.set_pose_by_view_index(np.array(view_indices))
        self.set_state_infos([None] * len(scan_ids))
        # update world states
        self.update_world_states()
        return

    def make_actions(
        self,
        h_times: np.ndarray,
        e_times: np.ndarray,
        next_viewpoint_ids: list
    ) -> None:
        # pose_adapt
        e_masks = np.logical_or(
            np.logical_and(e_times == -1, self.view_indices < 12),
            np.logical_and(e_times == 1, self.view_indices > 23)
        )
        e_times[e_masks] = 0
        h_ticks = (self.view_indices + h_times) % 12        # 0~11
        e_ticks = (self.view_indices + e_times * 12) // 12  # 0~2
        self.set_pose_by_view_index(h_ticks + 12 * e_ticks)
        # update vp_ids
        self.set_vp_ids(next_viewpoint_ids)
        # update world states
        self.update_world_states()
        return


def main():
    return


if __name__ == '__main__':
    main()
