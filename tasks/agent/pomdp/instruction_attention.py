import torch
import torch.nn as nn


class InstructionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout_ratio: float
    ) -> None:
        super(InstructionAttention, self).__init__()
        self.drop = nn.Dropout(p=dropout_ratio)
        self.h_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.Linear(embed_dim, embed_dim, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
        return

    def forward(
        self,
        h_t: torch.Tensor,
        instr_embed: torch.Tensor,
        instr_mask: torch.Tensor or None
    ) -> torch.Tensor:
        h_latent = self.h_fc(h_t).unsqueeze(2)

        instr_attn_weight = torch.bmm(instr_embed, h_latent).squeeze(2)
        if instr_mask is not None:
            instr_attn_weight.data.masked_fill_(instr_mask, -float('inf'))
        instr_attn_weight = self.softmax(instr_attn_weight)
        weighted_instr_embed = torch.bmm(instr_attn_weight.unsqueeze(1), instr_embed).squeeze(1)
        return self.drop(weighted_instr_embed), instr_attn_weight


def main():
    return


if __name__ == '__main__':
    main()
