import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import numpy as np
from agent.observation.nlp.glove import Glove


class InstructionEncoder(nn.Module):
    def __init__(
        self,
        config: dict,
        instr_dim: int,
        vocab: list
    ) -> None:
        super(InstructionEncoder, self).__init__()
        self.config = config
        self.padding_value = vocab.index('<pad>')

        # init embedding layer
        try:
            self.emb_layer = torch.load(config['r2r_env']['word_embedding'] + '.pt')
            print('Loading Glove embedding from %s.pt' % config['r2r_env']['word_embedding'])
        except FileNotFoundError:
            print('Loading Glove embedding from %s.txt' % config['r2r_env']['word_embedding'])
            # setup glove
            glove = Glove(config['r2r_env']['word_embedding'] + '.txt')
            weights_matrix = self.get_weights_matrix(vocab, glove)
            # build embedding layer
            vocab_size, feature_dim = weights_matrix.shape
            self.emb_layer = nn.Embedding(vocab_size, feature_dim)
            self.emb_layer.from_pretrained(torch.tensor(weights_matrix), freeze=True)
            # save vocab and embedding
            vocab_path = '/'.join(config['r2r_env']['word_embedding'].split('/')[:-1]) + '/vocab.txt'
            with open(vocab_path, 'w') as txt_file:
                for word in vocab:
                    txt_file.write(word + '\n')
            torch.save(self.emb_layer, config['r2r_env']['word_embedding'] + '.pt')

        # init instruction encoder network
        self.drop = nn.Dropout(p=config['state_tracker']['dropout_ratio'])
        self.lstm = nn.LSTM(
            input_size=self.emb_layer.embedding_dim,
            hidden_size=config['state_tracker']['obs']['instr']['lstm']['hidden_dim'],
            num_layers=config['state_tracker']['obs']['instr']['lstm']['num_layers'],
            batch_first=True,
            bidirectional=config['state_tracker']['obs']['instr']['lstm']['bidirectional']
        )
        self.encoder2decoder = nn.Sequential(
            nn.Linear(instr_dim, instr_dim, bias=False),
            nn.Linear(instr_dim, instr_dim, bias=False)
        )
        self.encode = self.forward
        return

    def get_weights_matrix(
        self,
        vocab: list,
        glove: Glove
    ) -> np.ndarray:
        weights_matrix = np.zeros((len(vocab), glove.feature_dim))
        for i, word in enumerate(vocab):
            if word in glove.words:
                weights_matrix[i] = glove.w2v(word)
            else:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(glove.feature_dim, ))
        return weights_matrix

    def forward(
        self,
        instrs: list
    ) -> (torch.Tensor, torch.Tensor, (torch.Tensor, torch.Tensor)):
        """
        instrs = [instr1, instr2, ...]
        instr: [token1_id, token2_id, ...]
        type(instr): torch.Tensor
        """
        instrs_id_len = [len(instr) for instr in instrs]
        instrs_id_pad = pad_sequence(instrs, batch_first=True, padding_value=self.padding_value).to(self.config['device'])

        instrs_embed = self.emb_layer(instrs_id_pad)
        instrs_embed_pack = pack_padded_sequence(
            instrs_embed,
            lengths=instrs_id_len,
            batch_first=True,
            enforce_sorted=False
        )
        enc_h, (enc_h_t, enc_c_t) = self.lstm(instrs_embed_pack)
        if self.lstm.bidirectional:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]
        decoder_init = self.encoder2decoder(h_t)
        instr_embed, lengths = pad_packed_sequence(enc_h, batch_first=True)
        instr_embed = self.drop(instr_embed)

        instr_mask = torch.ones(len(instrs), max(instrs_id_len), dtype=torch.bool, device=self.config['device'])
        for idx, instr in enumerate(instrs):
            instr_mask[idx, :len(instr)] = torch.zeros_like(instr)
        return instr_embed, instr_mask, (decoder_init, c_t)


def main():
    return


if __name__ == '__main__':
    main()
