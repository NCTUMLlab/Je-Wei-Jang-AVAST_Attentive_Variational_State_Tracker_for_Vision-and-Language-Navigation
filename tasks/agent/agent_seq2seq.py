import torch
import torch.nn as nn
import numpy as np
from agent.agent_base import AgentBase
from agent.model import CategoricalPolicy, TwinnedQNetwork


class AgentSeq2Seq(AgentBase):
    def __init__(
        self,
        config: dict,
        env: object,
        weight_decay: float = 0.0005
    ) -> None:
        super().__init__(config, env)
        assert config['args']['agent'] == 'seq2seq'

        if config['args']['mode'] != 'test':
            self.agent_learning_config = config['agent'][config['args']['mode']]['learning']
            self.optimizer_obs = torch.optim.Adam(self.obs_encoder.parameters(), lr=self.agent_learning_config['lr'], weight_decay=weight_decay)
            self.optimizer_pomdp = torch.optim.Adam(self.state_tracker.parameters(), lr=self.agent_learning_config['lr'], weight_decay=weight_decay)
            self.gamma = self.agent_learning_config['gamma']

        input_size = self.state_tracker.state_dim
        # critic
        self.q_behavior = TwinnedQNetwork(input_size, self.cv_utils.action_feature_size).to(config['device'])
        # policy
        self.policy = CategoricalPolicy(input_size, self.cv_utils.action_feature_size).to(config['device'])

        # setup loss function and optimizer
        if config['args']['mode'] != 'test':
            self.cross_entropy = nn.CrossEntropyLoss()
            self.optimizer_q1 = torch.optim.Adam(self.q_behavior.q_net1.parameters(), lr=self.agent_learning_config['lr'])
            self.optimizer_q2 = torch.optim.Adam(self.q_behavior.q_net2.parameters(), lr=self.agent_learning_config['lr'])
            self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self.agent_learning_config['lr'], weight_decay=weight_decay)

        self.update_networks()
        for net_type, nets in self.networks.items():
            for net_id, net in nets.items():
                if isinstance(net, torch.nn.Module):
                    print('%5s: %-10s' % (net_type, net_id), "<class 'torch.nn.Module'>")
                else:
                    print('%5s: %-10s' % (net_type, net_id), type(net))
        return

    def _kld_gauss(
        self,
        mean_1: torch.Tensor,
        std_1: torch.Tensor,
        mean_2: torch.Tensor,
        std_2: torch.Tensor
    ) -> torch.Tensor:
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element) / mean_1.shape[0]

    def train(
        self,
        pair_datas: dict
    ) -> np.ndarray:
        labels = torch.tensor(pair_datas['labels'], device=self.config['device'])
        logits = torch.cat(pair_datas['logits'], dim=0)
        q1_s = torch.cat(pair_datas['q1s'], dim=0)
        q2_s = torch.cat(pair_datas['q2s'], dim=0)
        target_values = torch.zeros_like(q1_s)
        for batch_idx, (action, curriculum_reward) in enumerate(zip(pair_datas['labels'], pair_datas['curriculum_rewards'])):
            target_values[batch_idx, action] = curriculum_reward

        loss = torch.zeros(1, device=self.config['device'])
        # get critic loss
        legal = (q1_s != -float('inf'))
        q1_loss = torch.mean((q1_s[legal] - target_values[legal]).pow(2))
        q2_loss = torch.mean((q2_s[legal] - target_values[legal]).pow(2))
        loss += (q1_loss + q2_loss)
        # get policy loss
        policy_loss = self.cross_entropy(logits, labels)
        loss += policy_loss

        # get avast loss
        if self.config['args']['state_tracker'] == 'avast':
            # kld
            kld_loss = self._kld_gauss(
                mean_1=torch.cat(pair_datas['posterior_means'], dim=0),
                std_1=torch.cat(pair_datas['posterior_stds'], dim=0),
                mean_2=torch.cat(pair_datas['prior_means'], dim=0),
                std_2=torch.cat(pair_datas['prior_stds'], dim=0)
            )
            loss += kld_loss

        # zero grad
        if self.config['args']['state_tracker'] == 'avast' and not self.config['args']['aug_data']:
            pass
        else:
            self.optimizer_obs.zero_grad()
            self.optimizer_pomdp.zero_grad()
        self.optimizer_policy.zero_grad()
        self.optimizer_q1.zero_grad()
        self.optimizer_q2.zero_grad()

        # get gradient
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(self.obs_encoder.instr.parameters(), 100)
        nn.utils.clip_grad_norm_(self.state_tracker.parameters(), 10)
        nn.utils.clip_grad_norm_(self.q_behavior.parameters(), 1)

        # update
        self.optimizer_obs.step()
        self.optimizer_pomdp.step()
        self.optimizer_policy.step()
        self.optimizer_q1.step()
        self.optimizer_q2.step()
        return np.array([q1_loss.item(), q2_loss.item(), policy_loss.item(), 0, kld_loss.item() if self.config['args']['state_tracker'] == 'avast' else 0])

    def act(
        self,
        location_infos: list,
        instr_embed: torch.Tensor,
        instr_mask: torch.Tensor,
        last_action_features: torch.Tensor,
        hiddens: torch.Tensor,
        evaluate: bool,
        act_by: str = 'policy'
    ) -> (np.ndarray, dict, torch.Tensor, torch.Tensor):
        # get belief state
        belief_states, context_belief_states, instr_attn_weight, hiddens = self.get_belief_states(
            vision_features=self.cv_utils.get_vision_features(location_infos),
            abs_pose_features=self.cv_utils.get_abs_pose_features(location_infos).squeeze(0),
            action_features=last_action_features,
            instr_embed=instr_embed,
            instr_mask=instr_mask,
            hiddens=hiddens,
            enable_grad=not evaluate
        )

        with torch.set_grad_enabled(not evaluate):
            # get candidate action feature
            candidate_action_features = self.cv_utils.get_candidate_action_features(location_infos).squeeze(0)

            # generate actions
            legals = self.intent_to_mask(
                intents=[location_info.action_info.intents for location_info in location_infos],
                find_legal=True
            )
            illegal = torch.logical_not(torch.tensor(legals, dtype=torch.bool).view(-1, self.action_space))

            q1_s, q2_s = self.q_behavior(torch.cat([context_belief_states, hiddens[0]], dim=1), candidate_action_features)
            q1_s[illegal] = -float('inf')
            q2_s[illegal] = -float('inf')

            policy_out = self.policy(belief_states, candidate_action_features)
            policy_out[illegal] = -float('inf')

            outputs = {
                'logits': policy_out,
                'q1s': q1_s,
                'q2s': q2_s
            }

            # select action by greedy
            if act_by == 'critic':
                q_s = q1_s if np.random.rand() < 0.5 else q2_s
                actions = torch.argmax(q_s, dim=1).cpu().view(-1).numpy()
            elif act_by == 'policy':
                actions = torch.argmax(policy_out, dim=1).cpu().view(-1).numpy()
            else:
                raise NotImplementedError
        return actions, outputs, candidate_action_features, hiddens

    def act_with_dists(
        self,
        location_infos: list,
        instr_embed: torch.Tensor,
        instr_mask: torch.Tensor,
        last_action_features: torch.Tensor,
        hiddens: torch.Tensor,
        evaluate: bool
    ) -> (np.ndarray, dict, torch.Tensor, torch.Tensor, dict):
        assert not evaluate
        # get belief state
        belief_states, context_belief_states, instr_attn_weight, hiddens, dists = self.get_belief_states_with_dist(
            vision_features=self.cv_utils.get_vision_features(location_infos),
            abs_pose_features=self.cv_utils.get_abs_pose_features(location_infos).squeeze(0),
            action_features=last_action_features,
            instr_embed=instr_embed,
            instr_mask=instr_mask,
            hiddens=hiddens,
            enable_grad=not evaluate
        )

        # get candidate action feature
        candidate_action_features = self.cv_utils.get_candidate_action_features(location_infos).squeeze(0)

        # generate actions
        legals = self.intent_to_mask(
            intents=[location_info.action_info.intents for location_info in location_infos],
            find_legal=True
        )
        illegal = torch.logical_not(torch.tensor(legals, dtype=torch.bool).view(-1, self.action_space))

        q1_s, q2_s = self.q_behavior(torch.cat([context_belief_states, hiddens[0]], dim=1), candidate_action_features)
        q1_s[illegal] = -float('inf')
        q2_s[illegal] = -float('inf')

        policy_out = self.policy(belief_states, candidate_action_features)
        policy_out[illegal] = -float('inf')

        outputs = {'logits': policy_out, 'q1s': q1_s, 'q2s': q2_s}

        # select action by greedy
        actions = torch.argmax(policy_out, dim=1).cpu().view(-1).numpy()
        return actions, outputs, candidate_action_features, hiddens, dists


def main():
    return


if __name__ == '__main__':
    main()

