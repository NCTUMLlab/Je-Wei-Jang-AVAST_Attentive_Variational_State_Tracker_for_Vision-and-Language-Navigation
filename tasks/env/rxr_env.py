import sys
import random
import numpy as np
from cv2 import cv2
from env.env_utils import Statistic
from env.rxr_env_base import RxREnvBase


class RxREnv(RxREnvBase):
    def __init__(
        self,
        config: dict,
        scan_ids: list = []
    ) -> None:
        super().__init__(config, scan_ids)
        # env configuration
        self.max_iteration = config['r2r_env']['max_iteration']
        self.success_radius = config['r2r_env']['success_radius']
        if config['r2r_env']['reward_mode']['shaping'] in config['r2r_env']['reward_mode']['//shaping']:
            self.reward_shaping = config['r2r_env']['reward_mode']['shaping']
            self.reward_scale = config['r2r_env']['reward_mode']['scale']
        else:
            sys.exit('Invalid reward mode')
        return

    def shuffle(
        self
    ) -> None:
        random.shuffle(self.datasets['train'])
        return

    def reset(
        self,
        data_indices: list,
        split: str,
        gen_gif: bool = False
    ) -> (np.ndarray, list):
        headings = []
        self.reset_init(len(data_indices))

        for parallel_idx, data_idx in enumerate(data_indices):
            instr_data = self.datasets[split][data_idx]
            path_data = self.path_set[instr_data['path_id']]
            assert path_data['path_id'] == instr_data['path_id']

            # instr mapping init
            self.path_instr_ids.append('%d_%d' % (instr_data['path_id'], instr_data['idx']))
            self.instrs.append(instr_data['instr_tk_id'])
            # mp_env parameters init
            self.scan_ids.append(path_data['scan'])
            self.start_vp_ids.append(path_data['path'][0])
            headings.append(path_data['heading'])
            # environment recorder init
            self.goal_vp_ids.append(path_data['path'][-1] if len(path_data['path']) > 1 else '')
            self.old_vp_ids.append(path_data['path'][0])
            # expert recorder init
            self.expert_paths.append(path_data['path'])
            self.expert_path_lens[parallel_idx] = path_data['distance']
            # agent recorder init
            self.agent_paths.append([path_data['path'][0]])

        # start new episode from start location
        self.env.new_episodes(self.scan_ids, self.start_vp_ids, headings, gen_gif)
        state_infos = self.env.get_states()

        # render
        self.render()
        return self.dones, self.get_location_infos(state_infos)

    def _sim_make_action(
        self,
        h_times: np.ndarray,
        e_times: np.ndarray,
        forwards: np.ndarray
    ) -> None:
        def heading_adapt(
            h_times: np.ndarray
        ) -> None:
            seq_h_acts = []
            for i in range(np.max(np.abs(h_times))):
                h_acts = []
                for h_time in h_times:
                    h_acts.append(np.sign(h_time) if abs(h_time) > i else 0)
                seq_h_acts.append(h_acts)
            for h_acts in seq_h_acts:
                self.env.make_actions([0] * len(h_times), h_acts, [0.0] * len(h_times))
                self.render()
            return

        def elevation_adapt(
            e_times: np.ndarray
        ) -> None:
            seq_e_acts = []
            for i in range(np.max(np.abs(e_times))):
                e_acts = []
                for e_time in e_times:
                    e_acts.append(np.sign(e_time) if abs(e_time) > i else 0)
                seq_e_acts.append(e_acts)
            for e_acts in seq_e_acts:
                self.env.make_actions([0] * len(e_times), [0.0] * len(e_times), e_acts)
                self.render()
            return

        # heading, elevation adapt
        heading_adapt(h_times)
        elevation_adapt(e_times)

        # forward
        self.env.make_actions(forwards, [0.0] * len(forwards), [0.0] * len(forwards))
        self.render()
        return

    def _dict_make_action(
        self,
        h_times: np.ndarray,
        e_times: np.ndarray,
        next_viewpoint_ids: list
    ) -> None:
        self.env.make_actions(h_times, e_times, next_viewpoint_ids)
        self.render()
        return

    def _make_actions(
        self,
        action_indices: np.ndarray
    ) -> None:
        # get heading, elevation times, end_viewpoint_ids
        state_infos = self.env.get_states()
        h_times = np.zeros(len(action_indices), dtype=np.int)
        e_times = np.zeros(len(action_indices), dtype=np.int)
        forwards = np.zeros(len(action_indices), dtype=np.int)
        next_viewpoint_ids = []
        for idx, action_idx in enumerate(action_indices):
            state_info = state_infos[idx]
            loc_ends = self._get_loc_ends(state_info)
            if action_idx == self.skip_action_idx:
                # stay at same viewpoint when act skip-aciton or finish-action
                loc_end = loc_ends[0]
            else:
                loc_end = loc_ends[action_idx]
                h_times[idx] = loc_end['absViewIndex'] % 12 - state_info.viewIndex % 12
                e_times[idx] = loc_end['absViewIndex'] // 12 - state_info.viewIndex // 12
                forwards[idx] = loc_end['forward']
            next_viewpoint_ids.append(loc_end['nextViewpointId'])
            if state_info.location.viewpointId == loc_end['nextViewpointId']:
                assert forwards[idx] == h_times[idx] == e_times[idx] == 0
                assert action_idx == self.skip_action_idx or action_idx == self.finish_action_idx

        # short h_times
        h_times[h_times > 6] -= 12
        h_times[h_times < -6] += 12

        # make action on mp_env
        if self.mode == 'sim':
            self._sim_make_action(h_times, e_times, forwards)
        elif self.mode == 'dict':
            self._dict_make_action(h_times, e_times, next_viewpoint_ids)
        else:
            sys.exit('Invalid env mode')
        return

    def step(
        self,
        action_indices: np.ndarray
    ) -> (np.ndarray, np.ndarray, list):
        # act
        self._make_actions(action_indices)
        # get observation, direction and reward
        state_infos = self.env.get_states()
        for idx, state_info in enumerate(state_infos):
            if not self.dones[idx]:
                self.agent_paths[idx].append(state_info.location.viewpointId)
        rewards = self.reward_func(action_indices)
        self.set_iterations(self.iterations + (1 - np.sign(self.dones)))
        return rewards, self.dones, self.get_location_infos(state_infos)

    def reward_func(
        self,
        action_indices: np.ndarray
    ) -> np.ndarray:
        # get current vp_ids
        now_vp_ids = [str(state_info.location.viewpointId) for state_info in self.env.get_states()]

        # update agent path
        self.set_agent_path_lens(self.agent_path_lens + self.get_seq_distance(self.scan_ids, now_vp_ids, self.old_vp_ids))

        # calculate distance
        now_distance2goals = self.get_seq_distance(self.scan_ids, now_vp_ids, self.goal_vp_ids)

        # calculate improvement rewards
        if 'goal' in self.reward_shaping:
            old_distance2goals = self.get_seq_distance(self.scan_ids, self.old_vp_ids, self.goal_vp_ids)
            rewards = (old_distance2goals - now_distance2goals) / 10
        elif 'fidelity' in self.reward_shaping:
            rewards = np.zeros(len(action_indices))
        else:
            raise NotImplementedError

        # iterate each trajectory to check whether it is ending
        for idx, action_idx in enumerate(action_indices):
            if action_idx == self.skip_action_idx:
                self.dones[idx] += 1
                rewards[idx] = 0
                continue

            if (self.iterations[idx] == self.max_iteration - 1) or (action_idx == self.finish_action_idx):
                # check not done, and mark it as True
                assert not self.dones[idx]
                self.dones[idx] = 1

                # record self-stop rate
                if action_idx == self.finish_action_idx:
                    self.self_stop[idx] = 1

                # get success
                rewards[idx] = 1 if now_distance2goals[idx] <= self.success_radius else 0

                # get cls
                if 'fidelity' in self.reward_shaping:
                    rewards[idx] += self.get_cls_score(
                        scan_id=self.scan_ids[idx],
                        agent_path=self.agent_paths[idx],
                        agent_path_len=self.agent_path_lens[idx],
                        expert_path=self.expert_paths[idx],
                        expert_path_len=self.expert_path_lens[idx]
                    )

        # update old coordinate
        self.set_old_vp_ids(now_vp_ids)
        return rewards * self.reward_scale

    def get_cls_score(
        self,
        scan_id: str,
        agent_path: list,
        agent_path_len: float,
        expert_path: list,
        expert_path_len: float
    ) -> float:
        coverage = np.mean(
            [
                np.exp(
                    -np.min(
                        self.get_seq_distance(
                            [scan_id] * len(agent_path),
                            [expert_vp_id] * len(agent_path),
                            agent_path
                        )
                    ) / self.success_radius
                )
                for expert_vp_id in expert_path
            ]
        )
        expected = coverage * expert_path_len
        score = expected / (expected + np.abs(expected - agent_path_len))
        return coverage * score

    def get_seq_distance(
        self,
        scan_ids: list,
        vp_ids1: list,
        vp_ids2: list
    ) -> np.ndarray:
        return np.array([self.distances_map[scan_id][vp_id1][vp_id2] for scan_id, vp_id1, vp_id2 in zip(scan_ids, vp_ids1, vp_ids2)], dtype=np.float)

    def get_statistics(
        self
    ) -> Statistic:
        now_vp_ids = [str(state_info.location.viewpointId) for state_info in self.env.get_states()]
        nav_error = self.get_seq_distance(self.scan_ids, self.goal_vp_ids, now_vp_ids).tolist()
        succ_rate = [1 if ne <= 3 else 0 for ne in nav_error]
        shortest_path = np.array([self.distances_map[scan_id][start_vp_id][goal_vp_id]
                                 for scan_id, start_vp_id, goal_vp_id in zip(self.scan_ids, self.start_vp_ids, self.goal_vp_ids)])
        cov_w_len_score = [
            self.get_cls_score(scan_id, agent_path, agent_path_len, expert_path, expert_path_len)
            for scan_id, agent_path, agent_path_len, expert_path, expert_path_len
            in zip(self.scan_ids, self.agent_paths, self.agent_path_lens, self.expert_paths, self.expert_path_lens)
        ]
        # get stat
        stat = Statistic(
            path_len=self.agent_path_lens.tolist(),
            nav_error=nav_error,
            succ_rate=succ_rate,
            succ_w_path_len=(np.array(succ_rate) * ((shortest_path / np.max((shortest_path, self.agent_path_lens), axis=0)))).tolist(),
            self_stop_rate=self.self_stop.tolist(),
            cov_w_len_score=cov_w_len_score,
            path_instr_id=self.path_instr_ids
        )
        return stat

    def render(
        self
    ) -> None:
        state_info = self.env.get_states()[self.env.rendering_idx]
        if self.env.verbose:
            self.print_info(state_info, self.iterations[self.env.rendering_idx])
        if self.env.rendering_idx > -1 and self.mode == 'sim':
            rgb = np.array(state_info.rgb, copy=True)
            for loc in state_info.navigableLocations[1:]:
                # Draw actions on the screen
                font_scale = 3.0 / loc.rel_distance
                x_axis = int(self.env.width / 2 + loc.rel_heading / self.env.hfov * self.env.width)
                y_axis = int(self.env.height / 2 - loc.rel_elevation / self.env.vfov * self.env.height)
                if loc.viewpointId == self.goal_vp_ids[self.env.rendering_idx]:
                    cv2.putText(rgb, str(loc.view_index), (x_axis, y_axis), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, self.env.goal_text_color, thickness=3)
                else:
                    cv2.putText(rgb, str(loc.view_index), (x_axis, y_axis), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, self.env.nav_text_color, thickness=3)
            cv2.imshow('Python RGB', rgb)
            _ = cv2.waitKey(1)
        return

    def print_info(
        self,
        state_info: object,
        iteration: int
    ) -> None:
        print('it: %2d | %s, heading: %9.6f, elevation: %9.6f, viewIndex: %2d'
              % (iteration, state_info.location.viewpointId,
                 state_info.heading, state_info.elevation, state_info.viewIndex))
        return


def main():
    return


if __name__ == '__main__':
    main()
