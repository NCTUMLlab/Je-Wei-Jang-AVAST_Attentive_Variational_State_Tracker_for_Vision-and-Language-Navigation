import os
import torch
from torch.nn import functional as F
import torch.distributions as D
import numpy as np
from agent.model import CategoricalPolicy
from agent.agent_base import AgentBase


class AgentReinforce(AgentBase):
    def __init__(
        self,
        config: dict,
        env: object,
        weight_decay: float = 0.0005
    ) -> None:
        super().__init__(config, env)
        self.agent_learning_config = config['agent'][config['args']['mode']]['learning']

        assert config['args']['agent'] == 'reinforce'
        input_size = self.state_tracker.state_dim
        self.gamma = self.agent_learning_config['gamma']

        # policy
        self.policy = CategoricalPolicy(input_size, self.cv_utils.action_feature_size).to(config['device'])

        # setup loss function and optimizer
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self.agent_learning_config['lr'], weight_decay=weight_decay)

        self.update_networks()
        for net_type, nets in self.networks.items():
            for net_id, net in nets.items():
                if isinstance(net, torch.nn.Module):
                    print('%5s: %-10s' % (net_type, net_id), "<class 'torch.nn.Module'>")
                else:
                    print('%5s: %-10s' % (net_type, net_id), type(net))
        return

    def _get_actions_prob(
        self,
        belief_state: torch.Tensor,
        candidate_action_embed: torch.Tensor,
        illegal: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        policy_out = self.policy(belief_state, candidate_action_embed)
        policy_out[illegal] = -float('inf')
        actions_prob = F.softmax(policy_out, dim=1)
        tiny_prob = (actions_prob == 0.0).float() * 1e-8
        actions_log_prob = torch.log(actions_prob + tiny_prob)
        return policy_out, actions_prob, actions_log_prob

    def _get_policy_rl_loss(
        self,
        batch_mem: list
    ) -> torch.Tensor or None:
        belief_state, critic_belief_state, action, intent, candidate_action_feature, reward, cummulative_reward, agent_mask, done = batch_mem

        # get policy loss
        illegal = self.intent_to_mask(intents=intent, find_legal=False)
        logit = self.policy(belief_state, candidate_action_feature)
        logit[illegal] = -float('inf')

        logit = logit[agent_mask]
        action = action[agent_mask].squeeze()

        agent_actions_log_prob = F.cross_entropy(logit, action.clone(), reduction='none').unsqueeze(1)
        agent_cummulative_reward = cummulative_reward[agent_mask]

        assert agent_actions_log_prob.shape == agent_cummulative_reward.shape
        if len(agent_actions_log_prob) > 0:
            policy_rl_loss = agent_cummulative_reward * agent_actions_log_prob
            return policy_rl_loss.mean()
        else:
            return None

    def train(
        self,
        agent_mems: list
    ) -> (np.ndarray, float):
        batch_agent_mem = self._get_batch_mem(agent_mems)
        policy_loss = self._get_policy_rl_loss(batch_agent_mem)
        self._update_param(loss=policy_loss, optimizer=self.optimizer_policy, network=self.policy, clip_grad=1)
        policy_loss_float = self._get_float_loss(policy_loss)
        return np.array([0, 0, policy_loss_float, 0, 0]), 0

    def act(
        self,
        location_infos: list,
        instr_embed: torch.Tensor,
        instr_mask: torch.Tensor,
        last_action_features: torch.Tensor,
        hiddens: torch.Tensor,
        evaluate: bool
    ) -> (np.ndarray, torch.Tensor, torch.Tensor):
        # get belief state
        if evaluate:
            belief_states, _, instr_attn_weight, hiddens = self.get_belief_states(
                vision_features=self.cv_utils.get_vision_features(location_infos),
                abs_pose_features=self.cv_utils.get_abs_pose_features(location_infos).squeeze(0),
                action_features=last_action_features,
                instr_embed=instr_embed,
                instr_mask=instr_mask,
                hiddens=hiddens,
                enable_grad=False
            )
        else:
            belief_states, _, instr_attn_weight, hiddens, _ = self.get_belief_states_with_dist(
                vision_features=self.cv_utils.get_vision_features(location_infos),
                abs_pose_features=self.cv_utils.get_abs_pose_features(location_infos).squeeze(0),
                action_features=last_action_features,
                instr_embed=instr_embed,
                instr_mask=instr_mask,
                hiddens=hiddens,
                enable_grad=False
            )

        with torch.no_grad():
            # get candidate action features
            candidate_action_features = self.cv_utils.get_candidate_action_features(location_infos).squeeze(0)

            # generate actions
            legals = self.intent_to_mask(
                intents=[location_info.action_info.intents for location_info in location_infos],
                find_legal=True
            )
            policy_out = self.policy(belief_states, candidate_action_features)
            illegal = torch.logical_not(torch.tensor(legals, dtype=torch.bool))
            policy_out[illegal] = -float('inf')
            if evaluate:
                # select action by greedy
                actions = torch.argmax(policy_out, dim=1).cpu().view(-1).numpy()
            else:
                # sample action from categorical distribution
                actions_prob = F.softmax(policy_out, dim=1)
                actions_dist = D.Categorical(actions_prob)
                actions = actions_dist.sample().cpu().view(-1).numpy()
        return actions, candidate_action_features, hiddens

    def load_pre_train(
        self,
        load_dir: str
    ) -> None:
        models = {
            'pomdp': [self.config['args']['state_tracker']],
            'obs': ['vision', 'instr'],
            'agent': ['policy']
        }

        for net_type, net_ids in models.items():
            for net_id in net_ids:
                net_dir = os.path.join(load_dir, '%s_%s.pt' % (net_type, net_id))
                self.networks[net_type][net_id].load_state_dict(torch.load(net_dir))
                if net_type != 'agent':
                    self.networks[net_type][net_id].eval()
                    for param in self.networks[net_type][net_id].parameters():
                        param.requires_grad = False
                print('load %s_%s from %s' % (net_type, net_id, net_dir))
        return


def main():
    return


if __name__ == '__main__':
    main()
