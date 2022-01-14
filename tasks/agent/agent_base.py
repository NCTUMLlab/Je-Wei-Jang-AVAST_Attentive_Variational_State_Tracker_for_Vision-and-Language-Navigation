import os
import sys
import torch
import torch.nn as nn
import numpy as np
from agent.pomdp.ast import AST
from agent.pomdp.avast import AVAST
from agent.observation.observation_encoder import ObservationEncoder


class AgentBase():
    def __init__(
        self,
        config: dict,
        env: object
    ) -> None:
        super().__init__()
        self.config = config

        # setup observation encoder
        self.obs_encoder = ObservationEncoder(config, env.nlp_utils.vocab)

        self.cv_utils = env.cv_utils
        self.action_space = env.action_space

        # setup state tracker
        if config['args']['state_tracker'] == 'ast':
            self.state_tracker = AST(
                config=config,
                vision_dim=self.obs_encoder.vision_dim,
                abs_pose_feature_size=self.cv_utils.pose_feature_size,
                action_feature_size=self.cv_utils.action_feature_size,
                additional_track=config['args']['additional_track']
            ).to(config['device'])
        elif config['args']['state_tracker'] == 'avast':
            self.state_tracker = AVAST(
                config=config,
                vision_dim=self.obs_encoder.vision_dim,
                abs_pose_feature_size=self.cv_utils.pose_feature_size,
                action_feature_size=self.cv_utils.action_feature_size,
                additional_track=config['args']['additional_track']
            ).to(config['device'])
        else:
            sys.exit('Invalid state tracker mode')

        # init agent
        self.q_behavior, self.q_target, self.policy, self.log_alpha = None, None, None, None
        self.networks = {
            'agent': {'q_behavior': None, 'q_target': None, 'policy': None, 'log_alpha': None},
            'pomdp': {'ast': None, 'avast': None},
            'obs': {'vision': None, 'instr': None}
        }
        return

    def _update_param(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        network: torch.nn.Module = None,
        clip_grad: float = 0,
        retain: bool = False
    ) -> None:
        optimizer.zero_grad()
        loss.backward(retain_graph=retain)
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(network.parameters(), clip_grad)
        optimizer.step()
        return

    def _get_float_loss(
        self,
        loss: torch.Tensor or None
    ) -> torch.Tensor or None:
        return loss.item() if loss is not None else 0

    def _get_batch_mem(
        self,
        mems: list,
        enable_grad: bool = False
    ) -> torch.Tensor:
        belief_state_list = []
        critic_belief_state_list = []
        action_list, intent_list, candidate_action_feature_list = [], [], []
        reward_list, cummulative_reward_list = [], []
        agent_mask_list, done_list = [], []
        for mem in mems:
            instr, vision, abs_pose, action, intent, candidate_action_feature, reward, expert_len = mem
            traj_len, sub_batch_size = vision.shape[:2]
            # encode instruction
            with torch.set_grad_enabled(enable_grad):
                instr_embed, instr_mask, hiddens = self.obs_encoder.instr.encode(instr)
            # rollout
            for time_step in range(traj_len):
                if time_step == 0:
                    action_features = self.get_init_action(sub_batch_size)
                else:
                    action_features = torch.stack(
                        [candidate_action_feature[time_step - 1, batch_idx, action_idx, :] for batch_idx, action_idx in enumerate(action[time_step - 1].view(-1))],
                        dim=0
                    )
                belief_states, context_belief_states, instr_attn_weight, hiddens, _ = self.get_belief_states_with_dist(
                    vision_features=vision[time_step:time_step + 1, :, :, :],
                    abs_pose_features=abs_pose[time_step],
                    action_features=action_features,
                    instr_embed=instr_embed.clone(),
                    instr_mask=instr_mask,
                    hiddens=hiddens,
                    enable_grad=enable_grad
                )
                belief_state_list.append(belief_states)
                critic_belief_state_list.append(torch.cat([context_belief_states, hiddens[0]], dim=1))
                action_list.append(action[time_step])
                intent_list.append(intent[time_step])
                candidate_action_feature_list.append(candidate_action_feature[time_step])
                reward_list.append(reward[time_step])
                cummulative_reward_list.append(reward[time_step].clone())
                agent_mask_list.append((expert_len <= time_step))
                # agent_mask_list.append((expert_len <= 10))
                done_list.append(torch.ones_like(agent_mask_list[-1], device=self.config['device']) * (time_step == vision.shape[0] - 1))
            # get cummulative reward
            for shift in range(1, traj_len):
                cummulative_reward_list[-shift - 1] += self.gamma * cummulative_reward_list[-shift]

        batch_mem = (
            torch.cat(belief_state_list, dim=0),
            torch.cat(critic_belief_state_list, dim=0),
            torch.cat(action_list, dim=0),
            torch.cat(intent_list, dim=0),
            torch.cat(candidate_action_feature_list, dim=0),
            torch.cat(reward_list, dim=0),
            torch.cat(cummulative_reward_list, dim=0),
            torch.cat(agent_mask_list, dim=0),
            torch.cat(done_list, dim=0)
        )
        return batch_mem

    def intent_to_mask(
        self,
        intents: list or torch.Tensor,
        find_legal: bool
    ) -> np.ndarray or torch.Tensor:
        if isinstance(intents, list):
            # float for calculate prob of a random action
            masks = np.zeros((len(intents), len(intents[0])), dtype=np.float)
            for parallel_idx, intent in enumerate(intents):
                masks[parallel_idx] = (intent != -1).astype(np.float) if find_legal else (intent == -1).astype(np.float)
            return masks
        if isinstance(intents, torch.Tensor):
            # bool for filtering the illegal or legal actions
            return (intents != -1).to(torch.bool) if find_legal else (intents == -1).to(torch.bool)

    def get_init_action(
        self,
        size: int
    ) -> torch.Tensor:
        return torch.zeros(size, self.cv_utils.action_feature_size, requires_grad=False, device=self.config['device'])

    def get_belief_states(
        self,
        vision_features: torch.Tensor,
        abs_pose_features: torch.Tensor,
        action_features: torch.Tensor,
        instr_embed: torch.Tensor,
        instr_mask: torch.Tensor,
        hiddens: (torch.Tensor, torch.Tensor),
        enable_grad: bool,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, (torch.Tensor, torch.Tensor)):
        """
        act function pre-processing
        """
        with torch.set_grad_enabled(enable_grad):
            vision_embed, instr_embed = self.obs_encoder.encode(
                vision_features, instr_embed, hiddens[0]
            )
            belief_states, context_belief_states, instr_attn_weight, hiddens = self.state_tracker.inference(
                vision_embed, instr_embed, instr_mask, abs_pose_features, action_features, hiddens
            )
        return belief_states, context_belief_states, instr_attn_weight, hiddens

    def get_belief_states_with_dist(
        self,
        vision_features: torch.Tensor,
        abs_pose_features: torch.Tensor,
        action_features: torch.Tensor,
        instr_embed: torch.Tensor,
        instr_mask: torch.Tensor,
        hiddens: (torch.Tensor, torch.Tensor),
        enable_grad: bool,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, (torch.Tensor, torch.Tensor), dict):
        """
        act function pre-processing
        """
        with torch.set_grad_enabled(enable_grad):
            vision_embed, instr_embed = self.obs_encoder.encode(
                vision_features, instr_embed, hiddens[0]
            )
            belief_states, context_belief_states, instr_attn_weight, hiddens, dists = self.state_tracker.forward(
                vision_embed, instr_embed, instr_mask, abs_pose_features, action_features, hiddens
            )
        return belief_states, context_belief_states, instr_attn_weight, hiddens, dists

    def random_act(
        self,
        location_infos: list
    ) -> np.ndarray:
        legals = self.intent_to_mask(
            intents=[location_info.action_info.intents for location_info in location_infos],
            find_legal=True
        )
        actions = np.zeros(len(legals), dtype=np.int)
        for parallel_idx, legal in enumerate(legals):
            actions[parallel_idx] = np.random.choice(np.arange(self.action_space), p=legal / sum(legal))
        return actions

    def update_networks(
        self
    ) -> None:
        # agent
        if self.config['args']['agent'] == 'sacd':
            self.networks['agent']['q_behavior'] = self.q_behavior
            self.networks['agent']['q_target'] = self.q_target
            self.networks['agent']['policy'] = self.policy
            self.networks['agent']['log_alpha'] = self.log_alpha
        elif self.config['args']['agent'] == 'reinforce':
            self.networks['agent']['policy'] = self.policy
        elif self.config['args']['agent'] == 'seq2seq':
            self.networks['agent']['q_behavior'] = self.q_behavior
            self.networks['agent']['policy'] = self.policy
        else:
            sys.exit('Invalid agent model')

        # pomdp
        if self.config['args']['state_tracker'] == 'ast':
            self.networks['pomdp']['ast'] = self.state_tracker
        elif self.config['args']['state_tracker'] == 'avast':
            self.networks['pomdp']['avast'] = self.state_tracker
        else:
            sys.exit('Invalid pomdp model')

        # observation
        if self.obs_encoder.vision is not None:
            self.networks['obs']['vision'] = self.obs_encoder.vision
        if self.obs_encoder.instr is not None:
            self.networks['obs']['instr'] = self.obs_encoder.instr
        return

    def change_mode(
        self,
        is_train: bool,
        ignore_state_tracker: bool = False
    ) -> None:
        for net_type, nets in self.networks.items():
            if (net_type == 'obs' or net_type == 'pomdp') and ignore_state_tracker:
                continue
            for net_id, net in nets.items():
                if isinstance(net, torch.nn.Module):
                    net.train(is_train)
        return

    def save(
        self,
        exp_name: str,
        it_now: int
    ) -> str:
        # filename flag
        flag = [
            exp_name,
            'it%d' % it_now,
            self.config['args']['agent'],
            self.config['args']['state_tracker'],
            self.config['args']['additional_track']
        ]
        save_dir = os.path.join(self.config['save_dir'], '_'.join(flag))
        os.system('mkdir %s' % save_dir)

        self.update_networks()
        for net_type, nets in self.networks.items():
            for net_id, net in nets.items():
                net_dir = os.path.join(save_dir, '%s_%s.pt' % (net_type, net_id))
                if isinstance(net, torch.nn.Module):
                    torch.save(net.state_dict(), net_dir)
                if isinstance(net, torch.Tensor):
                    torch.save(net, net_dir)
        return save_dir

    def load(
        self,
        load_dir: str
    ) -> None:
        for net_type, nets in self.networks.items():
            for net_id, net in nets.items():
                net_dir = os.path.join(load_dir, '%s_%s.pt' % (net_type, net_id))
                if isinstance(net, torch.nn.Module):
                    net.load_state_dict(torch.load(net_dir))
                if isinstance(net, torch.Tensor):
                    net = torch.load(net_dir)
        return


def main():
    return


if __name__ == '__main__':
    main()
