from collections import deque
import torch
import numpy as np


class ReplayMemory():
    def __init__(
        self,
        config: dict,
        env: object,
        expert: str = '',
        on_policy: bool = False
    ) -> None:
        super().__init__()
        self.replay_memory_config = config['agent'][config['args']['mode']]['replay_memory']
        self.device = config['device']
        assert self.replay_memory_config['max_epi_len'] == config['r2r_env']['max_iteration']
        self.cv_utils = env.cv_utils
        self.on_policy = on_policy

        if expert:
            self.split = expert
            self.max_epi_num = len(env.datasets[self.split])
            self.min_epi_num = len(env.datasets[self.split])
        else:
            self.max_epi_num = self.replay_memory_config['max_epi_num']
            self.min_epi_num = self.replay_memory_config['min_epi_num']

        self.max_epi_len = self.replay_memory_config['max_epi_len']

        # rl information
        self.memory = deque(maxlen=self.max_epi_num)
        self.instr = deque(maxlen=self.max_epi_num)
        return

    def assert_batch_is_valid(
        self,
        batch_size: int
    ) -> None:
        if self.on_policy:
            assert batch_size == len(self.memory) == len(self.instr)
            assert batch_size <= self.max_epi_num
        else:
            assert batch_size <= len(self.memory)
        assert len(self.memory) >= self.min_epi_num
        return

    def append(
        self,
        trajs: list,
        instrs: list
    ) -> None:
        assert len(trajs) == len(instrs)
        assert all([len(traj) <= self.max_epi_len for traj in trajs])
        # push trajectories, instructions into deques
        # -------------------------------------------------------
        # tran = (location_info, action, reward, expert_len)
        # traj = [tran1, tran2, ..., tranT]
        # trajs = [traj1, traj2, ...]
        # instrs = [instruction1, instruction2, ...]
        self.memory.extend(trajs)
        self.instr.extend(instrs)
        return

    def sample(
        self,
        batch_size: int,
        indices_select: np.ndarray = None
    ) -> list:
        self.assert_batch_is_valid(batch_size)

        # select memories by random select indices
        if indices_select is None:
            indices_select = np.random.choice(len(self.memory), replace=False, size=batch_size)
        assert batch_size == len(indices_select)
        traj_len_select = np.array([len(self.memory[idx]) for idx in indices_select], dtype=np.int)
        memory_select = np.array([self.memory[idx] for idx in indices_select], dtype=object)
        instr_select = np.array([self.instr[idx] for idx in indices_select], dtype=object)

        # divide selected-memory to serveral batch tensor by different length
        mems = []
        for traj_len in set(traj_len_select):
            # transform transitions
            mem_tmp = memory_select[(traj_len_select == traj_len)]
            # convert to list of tensor
            # -------------------------------------------------------
            # [(traj_len, 1, feature_size), (traj_len, 1, feature_size), ...]
            location_info_list, action_list, reward_list, expert_len_list = [], [], [], []
            for traj in mem_tmp:
                location_info_list.append([tran[0] for tran in traj])
                action_list.append(torch.tensor([tran[1] for tran in traj], dtype=torch.int64).view(traj_len, 1, 1))
                reward_list.append(torch.tensor([(tran[2]) for tran in traj], dtype=torch.float).view(traj_len, 1, 1))
                expert_len_list.append(traj[0][3])
            # convert to tensor
            # -------------------------------------------------------
            # (traj_len, batch_size_tmp, feature_size)
            action_batch = torch.cat(action_list, dim=1)
            reward_batch = torch.cat(reward_list, dim=1)
            expert_len_batch = torch.tensor(expert_len_list, dtype=torch.int64)

            # transform instructions
            instr_tmp = instr_select[(traj_len_select == traj_len)]
            instr_list_batch = [torch.tensor(instr, dtype=torch.long) for instr in instr_tmp]

            mems.append(
                self.preprocess_mem(
                    (instr_list_batch, location_info_list, action_batch, reward_batch, expert_len_batch)
                )
            )

        if self.on_policy:
            self.memory.clear()
            self.instr.clear()
        return mems

    def __len__(
        self
    ) -> int:
        return len(self.memory)

    def __str__(
        self
    ) -> str:
        return str(self.memory)

    def get_intent_batch(
        self,
        location_info_list: list
    ) -> torch.Tensor:
        # traj_len, batch_size, action_space
        intent_batch = np.zeros(
            (
                len(location_info_list[0]),
                len(location_info_list),
                len(location_info_list[0][0].action_info.intents)
            ),
            dtype=np.float
        )
        for batch_idx, location_infos in enumerate(location_info_list):
            for time_step, location_info in enumerate(location_infos):
                intent_batch[time_step, batch_idx] = location_info.action_info.intents
        return torch.tensor(intent_batch, dtype=torch.int64, device=self.device)

    def get_candiate_action_embed_batch(
        self,
        location_info_list: list
    ) -> torch.Tensor:
        candiate_action_embed_batch = []
        for location_infos in location_info_list:
            candiate_action_embed_batch.append(
                self.cv_utils.get_candidate_action_features(location_infos, batch=False)
            )
        return torch.cat(candiate_action_embed_batch, dim=1)

    def get_vision_batch(
        self,
        location_info_list: list
    ) -> torch.Tensor:
        vision_batch = []
        for location_infos in location_info_list:
            vision_batch.append(
                self.cv_utils.get_vision_features(location_infos, batch=False)
            )
        return torch.cat(vision_batch, dim=1)

    def get_abs_pose_batch(
        self,
        location_info_list: list
    ) -> torch.Tensor:
        abs_pose_batch = []
        for location_infos in location_info_list:
            abs_pose_batch.append(
                self.cv_utils.get_abs_pose_features(location_infos, batch=False)
            )
        return torch.cat(abs_pose_batch, dim=1)

    def preprocess_mem(
        self,
        mem: tuple,
    ) -> tuple:
        instr_list_batch, location_info_list, action_batch, reward_batch, expert_len_batch = mem
        new_mem = (
            instr_list_batch,
            self.get_vision_batch(location_info_list),
            self.get_abs_pose_batch(location_info_list),
            action_batch.to(self.device),
            self.get_intent_batch(location_info_list),
            self.get_candiate_action_embed_batch(location_info_list),
            reward_batch.to(self.device),
            expert_len_batch.to(self.device),
        )
        return new_mem


def main():
    return


if __name__ == '__main__':
    main()
