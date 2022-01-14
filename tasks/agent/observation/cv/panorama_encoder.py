import torch
import torch.nn as nn


class PanoramaEncoder(nn.Module):
    def __init__(
        self,
        config: dict,
        vision_feature_size: int
    ) -> None:
        super(PanoramaEncoder, self).__init__()
        query_dim = config['state_tracker']['obs']['vision']['attn']['query_dim']

        self.query_layer = nn.Sequential(
            nn.Linear(query_dim, vision_feature_size, bias=False),
            nn.Linear(vision_feature_size, vision_feature_size, bias=False),
            nn.Dropout(p=config['state_tracker']['dropout_ratio'])
        )
        self.softmax = nn.Softmax(dim=1)
        self.encode = self.forward
        return

    def forward(
        self,
        visions: torch.Tensor,
        h_t: torch.Tensor
    ) -> torch.Tensor:
        panorama = visions.squeeze(0)                                       # batch x v_num x v_dim
        query = self.query_layer(h_t).unsqueeze(2)                          # batch x v_dim x 1

        # Get attention
        attn = torch.bmm(panorama, query).squeeze(2)                        # batch x v_num
        attn = self.softmax(attn)

        vision_embed = torch.bmm(attn.unsqueeze(1), panorama).squeeze(1)    # batch x v_dim
        return vision_embed


def main():
    return


if __name__ == '__main__':
    main()
