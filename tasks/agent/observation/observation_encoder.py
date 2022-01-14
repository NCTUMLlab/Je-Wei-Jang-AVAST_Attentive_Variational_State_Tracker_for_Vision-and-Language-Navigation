import itertools
import torch
from agent.observation.cv.panorama_encoder import PanoramaEncoder
from agent.observation.nlp.instruction_encoder import InstructionEncoder


class ObservationEncoder():
    def __init__(
        self,
        config: dict,
        vocab: list
    ) -> None:
        # get vision feature size
        self.vision_feature_size = self._get_vision_feature_size(config)
        # get observation embedding size
        self.vision_dim, self.instr_dim = self._get_observation_space(config)

        # vision
        self.vision = PanoramaEncoder(config, self.vision_feature_size).to(config['device'])
        self.vision_encode = self.vision.encode

        # instruction
        self.instr = InstructionEncoder(config, self.instr_dim, vocab).to(config['device'])
        self.instr_encode = lambda *args: args[0]
        return

    def _get_vision_feature_size(
        self,
        config: dict
    ) -> int:
        vision_feature_size = config['r2r_env']['pano_feature_size'] + \
            config['r2r_env']['pose_space'] * config['r2r_env']['pose_repeat']
        return vision_feature_size

    def _get_observation_space(
        self,
        config: dict
    ) -> (int, int):
        # vision
        vision_dim = self._get_vision_feature_size(config)
        # instruction
        instr_dim = config['state_tracker']['obs']['instr']['lstm']['hidden_dim'] * \
            (2 if config['state_tracker']['obs']['instr']['lstm']['bidirectional'] else 1)
        return vision_dim, instr_dim

    def parameters(
        self
    ) -> itertools.chain:
        params = []
        for encoder in [self.vision, self.instr]:
            if encoder is not None:
                params.append(encoder.parameters())
        return itertools.chain(*params)

    def encode(
        self,
        vision: torch.Tensor,
        instr: torch.Tensor,
        h_t: torch.Tensor
    ) -> tuple:
        """
        vision.shape =  (1, batch_size, view_num, vision_dim)
        instr.shape =   (batch_size, max_len, instr_dim)
        h_t.shape =     (batch_size, h_dim)
        ---
        vision_embed.shape =    (batch_size, vision_dim)
        instr_embed.shape =     (batch_size, max_len, instr_dim)
        """
        vision_embed = self.vision_encode(vision, h_t)
        instr_embed = self.instr_encode(instr, h_t)
        return (vision_embed, instr_embed)


def main():
    return


if __name__ == '__main__':
    main()
