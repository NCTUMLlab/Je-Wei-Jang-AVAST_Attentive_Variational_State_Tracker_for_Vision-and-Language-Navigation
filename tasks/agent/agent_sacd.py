import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as D
import numpy as np
from agent.agent_base import AgentBase
from agent.model import TwinnedQNetwork, CategoricalPolicy


class AgentSACD(AgentBase):
    def __init__(
        self,
        config: dict,
        env: object,
        weight_decay: float = 0.0005
    ) -> None:
        super().__init__(config, env)
        assert config['args']['agent'] == 'sacd'

        if config['args']['mode'] != 'test':
            self.agent_learning_config = config['agent'][config['args']['mode']]['learning']
            self.gamma = self.agent_learning_config['gamma']
            target_entropy_ratio = self.agent_learning_config['target_entropy_ratio']
            self.target_entropy = -np.log(1.0 / self.action_space) * target_entropy_ratio

        input_size = self.state_tracker.state_dim
        # critic
        self.q_behavior = TwinnedQNetwork(input_size, self.cv_utils.action_feature_size).to(config['device'])
        self.q_target = TwinnedQNetwork(input_size, self.cv_utils.action_feature_size).to(config['device'])
        self.q_target.load_state_dict(self.q_behavior.state_dict())
        # policy
        self.policy = CategoricalPolicy(input_size, self.cv_utils.action_feature_size).to(config['device'])
        # entropy coeficient
        # self.log_alpha = torch.tensor(self.agent_learning_config['log_alpha_init'], dtype=torch.float32, requires_grad=True, device=config['device'])
        # self.alpha = self.log_alpha.exp()
        self.alpha = 0

        # setup loss function and optimizer
        if config['args']['mode'] != 'test':
            self.optimizer_q1 = torch.optim.Adam(self.q_behavior.q_net1.parameters(), lr=self.agent_learning_config['lr'], weight_decay=weight_decay)
            self.optimizer_q2 = torch.optim.Adam(self.q_behavior.q_net2.parameters(), lr=self.agent_learning_config['lr'], weight_decay=weight_decay)
            self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self.agent_learning_config['lr'], weight_decay=weight_decay)
            # self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=self.agent_learning_config['lr'], eps=1e-4)
            self.cross_entropy = nn.CrossEntropyLoss()

        self.update_networks()
        for net_type, nets in self.networks.items():
            for net_id, net in nets.items():
                if isinstance(net, torch.nn.Module):
                    print('%5s: %-10s' % (net_type, net_id), "<class 'torch.nn.Module'>")
                else:
                    print('%5s: %-10s' % (net_type, net_id), type(net))
        return

    def get_actions_prob(
        self,
        belief_state: torch.Tensor,
        candidate_action_feature: torch.Tensor,
        illegal: torch.Tensor,
        min_prob: float = 1e-8
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        policy_out = self.policy(belief_state, candidate_action_feature)
        policy_out[illegal] = -float('inf')
        actions_prob = F.softmax(policy_out, dim=1)
        tiny_prob = (actions_prob == 0.0).float() * min_prob
        actions_log_prob = torch.log(actions_prob + tiny_prob)
        return policy_out, actions_prob, actions_log_prob

    def _get_critics_loss(
        self,
        batch_mem: list
    ) -> (torch.Tensor, torch.Tensor) or (None, None):
        belief_state, critic_belief_state, action, intent, candidate_action_feature, reward, cummulative_reward, agent_mask, done = batch_mem

        # get current q values
        current_q1_s, current_q2_s = self.q_behavior(critic_belief_state, candidate_action_feature)
        illegal = self.intent_to_mask(intents=intent, find_legal=False)
        current_q1_s[illegal] = -float('inf')
        current_q2_s[illegal] = -float('inf')
        current_q1 = current_q1_s.gather(dim=1, index=action)
        current_q2 = current_q2_s.gather(dim=1, index=action)

        # get target q values
        with torch.no_grad():
            _, actions_prob, actions_log_prob = self.get_actions_prob(
                belief_state, candidate_action_feature, illegal
            )
            next_q1_s, next_q2_s = self.q_target(critic_belief_state, candidate_action_feature)
            next_expect_q = (
                actions_prob * (torch.min(next_q1_s, next_q2_s) - self.alpha * actions_log_prob)
            ).mean(dim=1, keepdim=True)
            next_expect_q.roll(shifts=-1, dims=0)
            next_expect_q[done] = 0
            target_value = reward + (self.gamma * next_expect_q)

        # get critic loss
        assert target_value.shape == current_q1.shape == current_q2.shape == cummulative_reward.shape
        agent_target_value = target_value[agent_mask]
        agent_current_q1 = current_q1[agent_mask]
        agent_current_q2 = current_q2[agent_mask]
        agent_cummulative_reward = cummulative_reward[agent_mask]
        assert agent_target_value.shape == agent_current_q1.shape == agent_current_q2.shape == agent_cummulative_reward.shape
        if len(agent_target_value) > 0:
            q1_td_loss = torch.mean((agent_current_q1 - agent_target_value).pow(2))
            q1_mc_loss = torch.mean((agent_current_q1 - agent_cummulative_reward).pow(2))
            q2_td_loss = torch.mean((agent_current_q2 - agent_target_value).pow(2))
            q2_mc_loss = torch.mean((agent_current_q2 - agent_cummulative_reward).pow(2))

            q1_loss = q1_td_loss + q1_mc_loss
            q2_loss = q2_td_loss + q2_mc_loss
            return q1_loss, q2_loss
        else:
            return None, None

    def _get_policy_rl_loss(
        self,
        batch_mem: list
    ) -> (torch.Tensor, torch.Tensor) or (None, None):
        belief_state, critic_belief_state, action, intent, candidate_action_feature, reward, cummulative_reward, agent_mask, done = batch_mem

        # get policy loss
        illegal = self.intent_to_mask(intents=intent, find_legal=False)
        logit, actions_prob, actions_log_prob = self.get_actions_prob(
            belief_state, candidate_action_feature, illegal
        )
        with torch.no_grad():
            q1_s, q2_s = self.q_behavior(critic_belief_state, candidate_action_feature)

        pass_mask = agent_mask.clone()
        pass_mask[:] = True
        agent_logit = logit[pass_mask]
        agent_action = action[pass_mask].squeeze()
        agent_actions_prob = actions_prob[pass_mask]
        agent_actions_log_prob = actions_log_prob[pass_mask]
        agent_q1_s = q1_s[pass_mask]
        agent_q2_s = q2_s[pass_mask]
        agent_cummulative_reward = cummulative_reward[pass_mask]

        assert agent_actions_prob.shape == agent_actions_log_prob.shape == agent_q1_s.shape == agent_q2_s.shape
        if len(agent_actions_prob) > 0:
            entropy = -torch.sum(agent_actions_prob * agent_actions_log_prob, dim=1, keepdim=True)
            expect_q = torch.sum(torch.min(agent_q1_s, agent_q2_s) * agent_actions_prob, dim=1, keepdim=True)
            kl_loss = (- expect_q - self.alpha * entropy).mean()
            pg_loss = (
                agent_cummulative_reward * F.cross_entropy(agent_logit, agent_action.clone(), reduction='none').unsqueeze(1)
            ).mean()
            policy_rl_loss = kl_loss + pg_loss
            return policy_rl_loss, entropy
        else:
            return None, None

    def _get_entropy_loss(
        self,
        entropy: torch.Tensor or None
    ) -> torch.Tensor or None:
        if entropy is not None and self.log_alpha.requires_grad and self.alpha.requires_grad:
            entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropy.detach()))
            return entropy_loss
        else:
            return None

    def _get_policy_bc_loss(
        self,
        batch_mem: list
    ) -> torch.Tensor:
        belief_state, _, action, intent, candidate_action_feature, _, _, _, _ = batch_mem
        illegal = self.intent_to_mask(intents=intent, find_legal=False)
        policy_out = self.policy(belief_state, candidate_action_feature)
        policy_out[illegal] = -float('inf')
        policy_bc_loss = self.cross_entropy(policy_out, action.squeeze(1))
        return policy_bc_loss

    def train(
        self,
        now_it: int,
        agent_mems: list,
        expert_mems: list
    ) -> (np.ndarray, float):
        # update q_target
        if now_it % self.agent_learning_config['target_replace_iteration'] == 0:
            with torch.no_grad():
                for param_behavior, param_target in zip(self.q_behavior.parameters(), self.q_target.parameters()):
                    param_target.data.mul_(self.agent_learning_config['ema'])
                    param_target.data.add_((1 - self.agent_learning_config['ema']) * param_behavior.data)

        batch_agent_mem = self._get_batch_mem(agent_mems)
        # batch_expert_mem = self._get_batch_mem(expert_mems)

        q1_loss, q2_loss = self._get_critics_loss(batch_agent_mem)
        policy_rl_loss, entropy = self._get_policy_rl_loss(batch_agent_mem)
        # entropy_loss = self._get_entropy_loss(entropy)
        # policy_bc_loss = self._get_policy_bc_loss(batch_expert_mem)
        # policy_loss = policy_rl_loss + 0.5 * policy_bc_loss
        policy_loss = policy_rl_loss

        self._update_param(loss=q1_loss, optimizer=self.optimizer_q1, network=self.q_behavior, clip_grad=1)
        self._update_param(loss=q2_loss, optimizer=self.optimizer_q2, network=self.q_behavior, clip_grad=1)
        self._update_param(loss=policy_loss, optimizer=self.optimizer_policy, network=self.policy, clip_grad=1)
        # self._update_param(loss=entropy_loss, optimizer=self.optimizer_alpha)
        # self.alpha = self.log_alpha.exp()

        q1_loss_float = self._get_float_loss(q1_loss)
        q2_loss_float = self._get_float_loss(q2_loss)
        policy_loss_float = self._get_float_loss(policy_loss)
        # entropy_loss_float = self._get_float_loss(entropy_loss)
        # return np.array([q1_loss_float, q2_loss_float, policy_loss_float, entropy_loss_float, 0]), self.alpha.detach().cpu().numpy()
        return np.array([q1_loss_float, q2_loss_float, policy_loss_float, 0, 0]), self.alpha

    def act(
        self,
        location_infos: list,
        instr_embed: torch.Tensor,
        instr_mask: torch.Tensor,
        last_action_features: torch.Tensor,
        hiddens: torch.Tensor,
        evaluate: bool
    ) -> (np.ndarray, torch.Tensor, torch.Tensor):
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
            illegal = torch.logical_not(torch.tensor(legals, dtype=torch.bool))

            policy_out = self.policy(belief_states, candidate_action_features)
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
            'agent': ['policy', 'q_behavior']
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

        self.networks['agent']['q_target'].load_state_dict(
            self.networks['agent']['q_behavior'].state_dict()
        )
        return


def main():
    return


if __name__ == '__main__':
    main()
