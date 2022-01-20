import MatterSim
import os
import json
from collections import namedtuple
import numpy as np
from tqdm import tqdm


Location = namedtuple(
    "Location",
    ["vp_id", "abs_heading", "abs_elevation", "view_index", "rel_heading", "rel_elevation", "forward"]
)
rad30 = np.deg2rad(30)
skybox_dir = '/root/mount/AVAST_R2R/data/v1/scans/'
connectivity_dir = '/root/mount/AVAST_R2R/connectivity/'


def absolutize_rad(
    rad: float
) -> float:
    """R -> (-pi,pi]"""
    return rad - 2 * np.pi * round(rad / (2 * np.pi))


def get_angular_distance(
    rel_heading: float,
    rel_elevation: float
) -> float:
    return np.sqrt(rel_heading ** 2 + rel_elevation ** 2)


def get_loc_navigable_key(
    scan_id: str,
    viewpoint_id: str,
    view_index: int
) -> str:
    return '%s_%s_%d' % (scan_id, viewpoint_id, view_index)


def new_episodes(
    scan_ids: list,
    vp_ids: list,
    headings: list
) -> None:
    sim = MatterSim.Simulator()
    sim.setDatasetPath(skybox_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setCameraResolution(800, 600)
    sim.setCameraVFOV(np.deg2rad(60))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(len(scan_ids))
    sim.setCacheSize(2 * len(scan_ids))
    sim.setDepthEnabled(False)
    sim.setRenderingEnabled(False)
    sim.newEpisode(scan_ids, vp_ids, headings, [0] * len(scan_ids))
    return sim


def main():
    # scan_ids = ['JF19kD82Mey']
    scan_ids = [tmp.split('_')[0] for tmp in os.listdir(connectivity_dir) if '.json' in tmp]

    connect = {}
    for scan_id in scan_ids:
        with open(connectivity_dir + '%s_connectivity.json' % scan_id) as file_name:
            connect_json = json.load(file_name)
        connect_scan_id = {}
        for loc in connect_json:
            if loc['included'] and any(loc['unobstructed']):
                connect_scan_id.update({loc['image_id']: loc})
        connect.update({scan_id: connect_scan_id})

    # get loc navigable
    loc_navigable = {}
    max_action_num = 0
    for scan_id in tqdm(scan_ids):
        # get possible navigate point at 0, 30, ..., 330 degree
        start_vp_ids = list(connect[scan_id].keys())
        num_start_vp_id = len(start_vp_ids)
        sim = new_episodes([scan_id] * num_start_vp_id, start_vp_ids, [0] * num_start_vp_id)
        navigable = {start_vp_id: {} for start_vp_id in start_vp_ids}

        # change perspective
        sim.makeAction([0] * num_start_vp_id, [0.0] * num_start_vp_id, [-1.0] * num_start_vp_id)
        for elevation_idx in range(3):
            for heading_idx in range(12):
                view_index = elevation_idx * 12 + heading_idx
                state_infos = sim.getState()
                for state_info in state_infos:
                    start_vp_id = state_info.location.viewpointId
                    for forward, loc_end_info in enumerate(state_info.navigableLocations[1:]):
                        end_vp_id = loc_end_info.viewpointId
                        distance = get_angular_distance(loc_end_info.rel_heading, loc_end_info.rel_elevation)

                        if end_vp_id not in navigable[start_vp_id] or distance < get_angular_distance(navigable[start_vp_id][end_vp_id].rel_heading, navigable[start_vp_id][end_vp_id].rel_elevation):
                            abs_heading = absolutize_rad(loc_end_info.rel_heading + state_info.heading)
                            abs_elevation = absolutize_rad(loc_end_info.rel_elevation + state_info.elevation)
                            navigable[start_vp_id][end_vp_id] = Location(
                                vp_id=loc_end_info.viewpointId,
                                abs_heading=abs_heading,
                                abs_elevation=abs_elevation,
                                view_index=view_index,
                                rel_heading=loc_end_info.rel_heading,
                                rel_elevation=loc_end_info.rel_elevation,
                                forward=forward + 1
                            )
                sim.makeAction([0] * num_start_vp_id, [1.0] * num_start_vp_id, [0.0] * num_start_vp_id)
            sim.makeAction([0] * num_start_vp_id, [0.0] * num_start_vp_id, [1.0] * num_start_vp_id)
        sim.makeAction([0] * num_start_vp_id, [0.0] * num_start_vp_id, [-1.0] * num_start_vp_id)

        navigable_sorted = {}
        for start_vp_id, loc_ends in navigable.items():
            loc_ends_sorted = sorted(loc_ends.values(), key=lambda x: abs(x.abs_heading))
            if len(loc_ends_sorted) > max_action_num:
                max_action_num = len(loc_ends_sorted)
            navigable_sorted[start_vp_id] = loc_ends_sorted

        for start_vp_id in start_vp_ids:
            for elevation_idx in range(3):
                for heading_idx in range(12):
                    view_index = elevation_idx * 12 + heading_idx
                    heading = (view_index % 12) * rad30
                    elevation = (view_index // 12 - 1) * rad30

                    loc_start = get_loc_navigable_key(scan_id, start_vp_id, view_index)
                    loc_navigable[loc_start] = [
                        {
                            'absViewIndex': view_index,
                            'nextViewpointId': start_vp_id,
                            'rel_heading': 0,
                            'rel_elevation': 0,
                            'distance': 0,
                            'forward': 0
                        }
                    ]

                    for loc_end in navigable_sorted[start_vp_id]:
                        rel_heading = absolutize_rad(loc_end.abs_heading - heading)
                        rel_elevation = absolutize_rad(loc_end.abs_elevation - elevation)

                        loc_navigable[loc_start].append(
                            {
                                'absViewIndex': loc_end.view_index,
                                'nextViewpointId': loc_end.vp_id,
                                'rel_heading': rel_heading,
                                'rel_elevation': rel_elevation,
                                'distance': get_angular_distance(rel_heading, rel_elevation),
                                'forward': loc_end.forward
                            }
                        )
                        if view_index == loc_end.view_index:
                            assert loc_end.rel_heading == loc_navigable[loc_start][-1]['rel_heading']
                            assert loc_end.rel_elevation == loc_navigable[loc_start][-1]['rel_elevation']
    print(json.dumps(loc_navigable, indent=4))
    # pipe into new json file
    return


if __name__ == '__main__':
    main()
