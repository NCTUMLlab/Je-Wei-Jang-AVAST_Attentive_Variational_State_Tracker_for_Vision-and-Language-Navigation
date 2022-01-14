import torch
import torch.nn as nn
from agent.pomdp.instruction_attention import InstructionAttention


class AST(nn.Module):
    def __init__(
        self,
        config: dict,
        vision_dim: int,
        abs_pose_feature_size: int,
        action_feature_size: int,
        additional_track: str
    ) -> None:
        super(AST, self).__init__()
        self.config = config
        tracker_config = config['state_tracker'][config['args']['state_tracker']]
        self.h_dim = tracker_config['hidden_dim']
        self.n_layer = tracker_config['num_layers']  # number of layers of LSTM
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
        self.lstm = nn.LSTMCell(
            input_size=vision_dim + self.h_dim,
            hidden_size=self.h_dim
        )

        # belief state
        self.out_fc = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim, bias=False),
            nn.Linear(self.h_dim, self.h_dim, bias=False)
        )

        self.state_dim = self.h_dim
        return

    def forward(
        self,
        vision_embed: torch.Tensor,
        instr_embed: torch.Tensor,
        instr_mask: torch.Tensor,
        abs_pose_features: torch.Tensor,
        action_features: torch.Tensor,
        hiddens: (torch.Tensor, torch.Tensor)
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, (torch.Tensor, torch.Tensor), dict):
        belief_states, context_belief_states, instr_attn_weight, (h_t, c_t) = self.inference(
            vision_embed, instr_embed, instr_mask, abs_pose_features, action_features, hiddens
        )
        return belief_states, context_belief_states, instr_attn_weight, (h_t, c_t), {}

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
        h_t, c_t = self.lstm(input_drop, hiddens)
        h_t_drop = self.drop(h_t)
        context_belief_states, instr_attn_weight = self.instr_attention(h_t_drop, instr_embed, instr_mask)
        belief_states = torch.cat(
            [context_belief_states, h_t],
            dim=1
        )
        belief_states = self.out_fc(belief_states)
        return belief_states, context_belief_states, instr_attn_weight, (h_t, c_t)


def main():
    return


if __name__ == '__main__':
    main()
