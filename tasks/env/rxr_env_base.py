import sys
import json
import multiprocessing
from collections import namedtuple
import numpy as np
import networkx as nx
from cv2 import cv2
from env.env_utils import load_datasets, ActionInfo, LocationInfo, CVutils
from env.mp_env_sim import MatterEnvSim
from env.mp_env_dict import MatterEnvDict


def get_lookup_table(
    graph: nx.classes.graph.Graph
) -> tuple:
    shortest_paths_map = dict(nx.all_pairs_dijkstra_path(graph))
    vp_ids_map = list(graph.nodes)
    distances_map = dict(nx.all_pairs_dijkstra_path_length(graph))
    return (shortest_paths_map, vp_ids_map, distances_map)


class RxREnvBase():
    def __init__(
        self,
        config: dict,
        scan_ids: list
    ) -> None:
        super().__init__()
        self.config = config
        self.rad30, self.rad180, self.rad360 = np.deg2rad(30), np.deg2rad(180), np.deg2rad(360)

        # load dataset
        self.path_set, self.datasets, self.nlp_utils = load_datasets(config, scan_ids)

        # cv_utils init
        self.cv_utils = CVutils(config)

        # action setting init
        self.action_space = config['r2r_env']['action_space']
        self.finish_action_idx = config['r2r_env']['finish_action_idx']
        self.skip_action_idx = config['r2r_env']['skip_action_idx']

        # init loc_navigable
        self.loc_navigable = self._init_loc_navigable()

        # init lookup table
        self.shortest_paths_map, self.vp_ids_map, self.distances_map = self._init_graphs()

        # mode init
        self._mode, self._env = '', None
        # instr mapping init
        self._path_instr_ids, self._instrs = [], []
        # mp_env parameters init
        self._scan_ids, self._start_vp_ids = [], []
        # environment recorder init
        self._goal_vp_ids, self._old_vp_ids, self._self_stop, self._iterations, self._dones = [], [], None, None, None
        # expert recorder init
        self._expert_paths, self._expert_path_lens = [], None
        # agent recorder init
        self._agent_paths, self._agent_path_lens = [], None

        # mp_env init
        self.set_env('dict')
        return

    def reset_init(
        self,
        data_num: int
    ) -> None:
        # instr mapping init
        self.set_path_instr_ids([])
        self.set_instrs([])
        # mp_env parameters init
        self.set_scan_ids([])
        self.set_start_vp_ids([])
        # environment recorder init
        self.set_goal_vp_ids([])
        self.set_old_vp_ids([])
        self.set_self_stop(np.zeros(data_num, dtype=np.float))
        self.set_iterations(np.zeros(data_num, dtype=np.int))
        self.set_dones(np.zeros(data_num, dtype=np.int))

        # expert recorder init
        self.set_expert_paths([])
        self.set_expert_path_lens(np.zeros(data_num, dtype=np.float))
        # agent recorder init
        self.set_agent_paths([])
        self.set_agent_path_lens(np.zeros(data_num, dtype=np.float))
        return

    def _init_loc_navigable(
        self
    ) -> dict:
        print('Loading navigable map from %s' % self.config['r2r_env']['adj_dict_file'])
        with open(self.config['r2r_env']['adj_dict_file'], 'r') as file_name:
            adj_dict = json.load(file_name)
        return adj_dict

    def _get_loc_ends(
        self,
        state_info: object or namedtuple
    ) -> list:
        loc_start = '%s_%s_%d' % (state_info.scanId, state_info.location.viewpointId, state_info.viewIndex)
        loc_ends = self.loc_navigable[loc_start]
        return loc_ends

    def _init_graphs(
        self
    ) -> (dict, dict, dict):
        def load_nav_graphs(
            scan_ids: set
        ) -> list:
            ''' Load connectivity graph for each scan '''
            def distance(
                pose1: dict,
                pose2: dict
            ) -> float:
                ''' Euclidean distance between two graph poses '''
                return ((pose1['pose'][3] - pose2['pose'][3]) ** 2 + (pose1['pose'][7] - pose2['pose'][7]) ** 2 + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

            graphs = []
            for scan_id in scan_ids:
                with open(self.config['r2r_env']['mp']['connectivity'] + '%s_connectivity.json' % scan_id) as file_name:
                    graph = nx.Graph()
                    positions = {}
                    data = json.load(file_name)
                    for i, item in enumerate(data):
                        if item['included']:
                            for j, conn in enumerate(item['unobstructed']):
                                if conn and data[j]['included']:
                                    positions[item['image_id']] = np.array(
                                        [
                                            item['pose'][3],
                                            item['pose'][7],
                                            item['pose'][11]
                                        ]
                                    )
                                    assert data[j]['unobstructed'][i], 'Graph should be undirected'
                                    graph.add_edge(
                                        item['image_id'],
                                        data[j]['image_id'],
                                        weight=distance(item, data[j])
                                    )
                    nx.set_node_attributes(graph, values=positions, name='position')
                    graphs.append(graph)
            return graphs

        # get scan_ids
        scan_ids = []
        for path_id, path_dict in self.path_set.items():
            if path_dict['scan'] not in scan_ids:
                scan_ids.append(path_dict['scan'])

        # load graph
        graphs = load_nav_graphs(scan_ids)

        # compute lookup table
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        lookup_tables = pool.map(get_lookup_table, graphs)
        shortest_paths_map, vp_ids_map, distances_map = {}, {}, {}
        for scan_id, lookup_table in zip(scan_ids, lookup_tables):
            shortest_paths_map[scan_id] = lookup_table[0]
            vp_ids_map[scan_id] = lookup_table[1]
            distances_map[scan_id] = lookup_table[2]
        return shortest_paths_map, vp_ids_map, distances_map

    def get_data_indices(
        self,
        split: str,
        it: int,
        parallel_size: int,
        evaluate: bool
    ) -> list:
        start_idx = (it % self.get_it_num(split, parallel_size)) * parallel_size
        data_indices = []
        for shift_idx in range(parallel_size):
            idx = start_idx + shift_idx
            if idx < len(self.datasets[split]):
                data_indices.append(idx)
            else:
                break
        if not evaluate and len(data_indices) < parallel_size:
            self.shuffle()
            data_indices += [i for i in range(parallel_size - len(data_indices))]
        return data_indices

    def get_it_num(
        self,
        split: str,
        parallel_size: int
    ) -> int:
        return np.ceil(len(self.datasets[split]) / parallel_size).astype(int)

    def get_expert_trajs(
        self,
        data_indices: list,
        split: str
    ) -> list:
        expert_trajs = []
        for idx in data_indices:
            path_id = self.datasets[split][idx]['path_id']
            expert_trajs.append(
                self.path_set[path_id]['expert']
            )
        return expert_trajs

    def get_location_infos(
        self,
        state_infos: list
    ) -> list:
        """
        get intent (view_index) for each action
            [legal action]: view_index (0~35)
            [other illegal action]: -1
        ---
        get rel_heading and rel_elevation for each action
            [legal action]: rel_heading, rel_elevation
            [finish action]: 0, 0
            [other illegal action]: 0, 0
        """
        location_infos = []
        for state_info in state_infos:
            # init non-finish action intent to -1
            intents = -np.ones(self.action_space, dtype=np.int)
            # init all action (rel_headings, rel_elevations) to (0, 0)
            rel_headings = np.zeros(self.action_space, dtype=np.float32)
            rel_elevations = np.zeros(self.action_space, dtype=np.float32)
            # get intents, rel_headings, rel_elevations
            for action_idx, loc_end in enumerate(self._get_loc_ends(state_info)):
                intents[action_idx] = loc_end['absViewIndex']
                rel_headings[action_idx] = loc_end['rel_heading']
                rel_elevations[action_idx] = loc_end['rel_elevation']

            # append location infomation
            location_infos.append(
                LocationInfo(
                    scan_id=state_info.scanId,
                    vp_id=state_info.location.viewpointId,
                    view_index=state_info.viewIndex,
                    action_info=ActionInfo(intents, rel_headings, rel_elevations)
                )
            )
        return location_infos

    def discretize_heading_rad(
        self,
        rad: float
    ) -> float:
        count = round(rad / self.rad30) % 12
        return count * self.rad30

    def discretize_elevation_rad(
        self,
        rad: float
    ) -> float:
        count = round(rad / self.rad30) % 12
        count -= 12 if count > 6 else 0
        if count == -2 or count == -1 or count == 0 or count == 1 or count == 2:
            pass
        elif count < -2:
            count = -2
        else:
            count = 2
        return count * self.rad30

    @property
    def mode(
        self
    ) -> str:
        return self._mode

    def set_mode(
        self,
        new_mode: str
    ) -> None:
        self._mode = new_mode
        return

    @property
    def env(
        self
    ) -> (MatterEnvSim, MatterEnvDict):
        return self._env

    def set_env(
        self,
        mode: str
    ) -> None:
        if mode == 'sim':
            self._env = MatterEnvSim(self.config)
            if self.env.rendering_idx > -1:
                cv2.namedWindow('Python RGB')
        elif mode == 'dict':
            self._env = MatterEnvDict(self.config)
            if self.env.rendering_idx > -1:
                cv2.destroyAllWindows()
        else:
            sys.exit('Invalid environment mode')
        self.set_mode(mode)
        return

    @property
    def path_instr_ids(
        self
    ) -> list:
        return self._path_instr_ids

    def set_path_instr_ids(
        self,
        new_path_instr_ids: list
    ) -> None:
        self._path_instr_ids = new_path_instr_ids
        return

    @property
    def instrs(
        self
    ) -> list:
        return self._instrs

    def set_instrs(
        self,
        new_instrs: list
    ) -> None:
        self._instrs = new_instrs
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
    def start_vp_ids(
        self
    ) -> list:
        return self._start_vp_ids

    def set_start_vp_ids(
        self,
        new_start_vp_ids: list
    ) -> None:
        self._start_vp_ids = new_start_vp_ids
        return

    @property
    def goal_vp_ids(
        self
    ) -> list:
        return self._goal_vp_ids

    def set_goal_vp_ids(
        self,
        new_goal_vp_ids: list
    ) -> None:
        self._goal_vp_ids = new_goal_vp_ids
        return

    @property
    def old_vp_ids(
        self
    ) -> list:
        return self._old_vp_ids

    def set_old_vp_ids(
        self,
        new_old_vp_ids: list
    ) -> None:
        self._old_vp_ids = new_old_vp_ids
        return

    @property
    def self_stop(
        self
    ) -> np.ndarray:
        return self._self_stop

    def set_self_stop(
        self,
        new_self_stop: np.ndarray
    ) -> None:
        self._self_stop = new_self_stop
        return

    @property
    def iterations(
        self
    ) -> np.ndarray:
        return self._iterations

    def set_iterations(
        self,
        new_iterations: np.ndarray
    ) -> None:
        self._iterations = new_iterations
        return

    @property
    def dones(
        self
    ) -> np.ndarray:
        return self._dones

    def set_dones(
        self,
        new_dones: np.ndarray
    ) -> None:
        self._dones = new_dones
        return

    @property
    def expert_paths(
        self
    ) -> list:
        return self._expert_paths

    def set_expert_paths(
        self,
        new_expert_paths: list
    ) -> None:
        self._expert_paths = new_expert_paths
        return

    @property
    def expert_path_lens(
        self
    ) -> np.ndarray:
        return self._expert_path_lens

    def set_expert_path_lens(
        self,
        new_expert_path_lens: np.ndarray
    ) -> None:
        self._expert_path_lens = new_expert_path_lens
        return

    @property
    def agent_paths(
        self
    ) -> list:
        return self._agent_paths

    def set_agent_paths(
        self,
        new_agent_paths: list
    ) -> None:
        self._agent_paths = new_agent_paths
        return

    @property
    def agent_path_lens(
        self
    ) -> np.ndarray:
        return self._agent_path_lens

    def set_agent_path_lens(
        self,
        new_agent_path_lens: np.ndarray
    ) -> None:
        self._agent_path_lens = new_agent_path_lens
        return


def main():
    return


if __name__ == '__main__':
    main()
