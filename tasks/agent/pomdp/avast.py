import torch
import torch.nn as nn
from agent.pomdp.instruction_attention import InstructionAttention


class AVAST(nn.Module):
    def __init__(
        self,
        config: dict,
        vision_dim: int,
        abs_pose_feature_size: int,
        action_feature_size: int,
        additional_track: str
    ) -> None:
        super(AVAST, self).__init__()
        self.config = config
        tracker_config = config['state_tracker'][config['args']['state_tracker']]
        self.h_dim = tracker_config['hidden_dim']
        self.z_dim = tracker_config['latent_dim']
        self.n_layer = tracker_config['num_layers']
        self.drop = nn.Dropout(p=config['state_tracker']['dropout_ratio'])

        # addtional information
        addtional_dim = self.h_dim
        self.additional_track = additional_track
        if additional_track == 'pose':
            self.angle_fc = nn.Sequential(
                nn.Linear(abs_pose_feature_size, self.h_dim, bias=False),
                nn.Linear(self.h_dim, addtional_dim, bias=False)
            )
        elif additional_track == 'action':
            self.angle_fc = nn.Sequential(
                nn.Linear(action_feature_size, self.h_dim, bias=False),
                nn.Linear(self.h_dim, addtional_dim, bias=False)
            )
        else:
            raise NotImplementedError

        # context belief state
        self.instr_attention = InstructionAttention(
            embed_dim=self.h_dim,
            dropout_ratio=config['state_tracker']['dropout_ratio']
        )

        # tracking belief state
        # transform
        self.phi_x = nn.Linear(vision_dim + addtional_dim, self.h_dim, bias=False)
        self.phi_z = nn.Linear(self.z_dim, self.h_dim, bias=False)
        # recurrent
        self.lstm = nn.LSTMCell(
            input_size=self.h_dim + self.h_dim,
            hidden_size=self.h_dim
        )
        # inference
        self.enc = nn.Linear(self.h_dim + self.h_dim, self.h_dim, bias=False)
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim, bias=False)
        self.enc_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim, bias=False),
            nn.Softplus()
        )
        # prior
        self.prior = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.prior_mean = nn.Linear(self.h_dim, self.z_dim, bias=False)
        self.prior_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim, bias=False),
            nn.Softplus()
        )

        # belief state
        self.out_fc = nn.Sequential(
            nn.Linear(self.h_dim + self.z_dim, self.h_dim, bias=False),
            nn.Linear(self.h_dim, self.h_dim, bias=False)
        )

        self.state_dim = self.h_dim
        return

    def _reparameterized_sample(
        self,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> torch.Tensor:
        return mean + std * torch.randn_like(std)

    def forward(
        self,
        vision_embed: torch.Tensor,
        instr_embed: torch.Tensor,
        instr_mask: torch.Tensor,
        abs_pose_features: torch.Tensor,
        action_features: torch.Tensor,
        hiddens: (torch.Tensor, torch.Tensor)
    ) -> (torch.Tensor, torch.Tensor, (torch.Tensor, torch.Tensor), dict):
        if self.additional_track == 'pose':
            additional_embed = self.angle_fc(abs_pose_features)
        elif self.additional_track == 'action':
            additional_embed = self.angle_fc(action_features)
        else:
            raise NotImplementedError

        concat_input = torch.cat([vision_embed, additional_embed], 1)
        input_drop = self.drop(concat_input)

        phi_x_t = self.phi_x(input_drop)

        # inference
        enc_t = self.enc(torch.cat([phi_x_t, hiddens[0]], 1))
        enc_mean_t = self.enc_mean(enc_t)
        enc_std_t = self.enc_std(enc_t)

        # prior
        prior_t = self.prior(hiddens[0])
        prior_mean_t = self.prior_mean(prior_t)
        prior_std_t = self.prior_std(prior_t)

        # sampling and reparameterization
        z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
        phi_z_t = self.phi_z(z_t)

        # recurrent
        h_t, c_t = self.lstm(torch.cat([phi_x_t, phi_z_t], 1), hiddens)
        h_t_drop = self.drop(h_t)

        context_belief_states, instr_attn_weight = self.instr_attention(h_t_drop, instr_embed, instr_mask)
        belief_states = torch.cat(
            [context_belief_states, z_t],
            dim=1
        )
        belief_states = self.out_fc(belief_states)
        dists = {
            'enc_mean': enc_mean_t,
            'enc_std': enc_std_t,
            'prior_mean': prior_mean_t,
            'prior_std': prior_std_t
        }
        return belief_states, context_belief_states, instr_attn_weight, (h_t, c_t), dists

    def inference(
        self,
        vision_embed: torch.Tensor,
        instr_embed: torch.Tensor,
        instr_mask: torch.Tensor,
        abs_pose_features: torch.Tensor,
        action_features: torch.Tensor,
        hiddens: (torch.Tensor, torch.Tensor)
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, (torch.Tensor, torch.Tensor)):
        if self.additional_track == 'pose':
            additional_embed = self.angle_fc(abs_pose_features)
        elif self.additional_track == 'action':
            additional_embed = self.angle_fc(action_features)
        else:
            raise NotImplementedError

        concat_input = torch.cat([vision_embed, additional_embed], 1)
        input_drop = self.drop(concat_input)

        phi_x_t = self.phi_x(input_drop)

        # inference
        enc_t = self.enc(torch.cat([phi_x_t, hiddens[0]], 1))
        z_t = self.enc_mean(enc_t)
        phi_z_t = self.phi_z(z_t)

        # recurrent
        h_t, c_t = self.lstm(torch.cat([phi_x_t, phi_z_t], 1), hiddens)
        h_t_drop = self.drop(h_t)

        context_belief_states, instr_attn_weight = self.instr_attention(h_t_drop, instr_embed, instr_mask)
        belief_states = torch.cat(
            [context_belief_states, z_t],
            dim=1
        )
        belief_states = self.out_fc(belief_states)
        return belief_states, context_belief_states, instr_attn_weight, (h_t, c_t)


def main():
    return


if __name__ == '__main__':
    main()
