import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int = 512
    ) -> None:
        super(DuelingQNetwork, self).__init__()
        # state value
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim * 2, embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.action_fc = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.out_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, 1)
        )
        return

    def forward(
        self,
        belief_states: torch.Tensor,
        candidate_action_features: torch.Tensor
    ) -> torch.Tensor:
        """
        belief_states:              (batch_size, state_dim * 2)
        candidate_action_features:  (batch_size, action_space, action_dim)
        """
        batch_size, action_space = candidate_action_features.shape[:2]

        # batch_size x 1 x embed_dim
        s_latent = self.state_fc(belief_states).unsqueeze(1)
        # batch_size x action_space x embed_dim
        a_latent = self.action_fc(candidate_action_features)

        # batch_size x action_space
        q_out = self.out_fc(s_latent + a_latent).squeeze(2)
        return q_out


class TwinnedQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        dueling: bool = True
    ) -> None:
        super(TwinnedQNetwork, self).__init__()
        if dueling:
            self.q_net1 = DuelingQNetwork(state_dim, action_dim)
            self.q_net2 = DuelingQNetwork(state_dim, action_dim)
        else:
            raise NotImplementedError
        return

    def forward(
        self,
        belief_states: torch.Tensor,
        candidate_action_features: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        return self.q_net1(belief_states, candidate_action_features), self.q_net2(belief_states, candidate_action_features)


class CategoricalPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int
    ) -> None:
        super(CategoricalPolicy, self).__init__()
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, action_dim),
            nn.Linear(action_dim, action_dim)
        )
        return

    def forward(
        self,
        belief_states: torch.Tensor,
        candidate_action_features: torch.Tensor
    ) -> torch.Tensor:
        """
        belief_states:              (batch_size, state_dim)
        candidate_action_features:  (batch_size, action_space, action_dim)
        """
        batch_size, action_space = candidate_action_features.shape[:2]
        # batch_size x action_dim x 1
        s_latent = self.state_fc(belief_states).unsqueeze(2)
        out = torch.bmm(
            candidate_action_features, s_latent
        ).squeeze(2)
        return out


def main():
    return


if __name__ == '__main__':
    main()
