import os
import re
import csv
import sys
import json
import math
import base64
import multiprocessing
from collections import namedtuple
import numpy as np
import torch
from tqdm import tqdm


ActionInfo = namedtuple(
    "ActionInfo",
    ["intents", "rel_headings", "rel_elevations"]
)


LocationInfo = namedtuple(
    "LocationInfo",
    ["scan_id", "vp_id", "view_index", "action_info"]
)


class Statistic():
    def __init__(
        self,
        path_len: list,
        nav_error: list,
        succ_rate: list,
        succ_w_path_len: list,
        self_stop_rate: list,
        cov_w_len_score: list,
        path_instr_id: list,
    ) -> None:
        super().__init__()
        self.path_len = path_len
        self.nav_error = nav_error
        self.succ_rate = succ_rate
        self.succ_w_path_len = succ_w_path_len
        self.self_stop_rate = self_stop_rate
        self.cov_w_len_score = cov_w_len_score
        self.path_instr_id = path_instr_id
        return

    def __add__(self, stat):
        new_stat = Statistic(
            self.path_len + stat.path_len,
            self.nav_error + stat.nav_error,
            self.succ_rate + stat.succ_rate,
            self.succ_w_path_len + stat.succ_w_path_len,
            self.self_stop_rate + stat.self_stop_rate,
            self.cov_w_len_score + stat.cov_w_len_score,
            self.path_instr_id + stat.path_instr_id
        )
        return new_stat

    def __len__(self):
        return len(self.path_instr_id)

    def get_average(self):
        count = len(self.path_instr_id)
        path_instr_id = ' '.join(self.path_instr_id)
        if count:
            avg_stat = Statistic(
                path_len=[sum(self.path_len) / count],
                nav_error=[sum(self.nav_error) / count],
                succ_rate=[sum(self.succ_rate) / count],
                succ_w_path_len=[sum(self.succ_w_path_len) / count],
                self_stop_rate=[sum(self.self_stop_rate) / count],
                cov_w_len_score=[sum(self.cov_w_len_score) / count],
                path_instr_id=['avg of (%s)' % path_instr_id]
            )
        else:
            avg_stat = Statistic(
                path_len=[0],
                nav_error=[0],
                succ_rate=[0],
                succ_w_path_len=[0],
                self_stop_rate=[0],
                cov_w_len_score=[0],
                path_instr_id=['empty dataset']
            )
        return avg_stat


class NLPutils():
    def __init__(
        self
    ) -> None:
        super().__init__()
        self.tokenizer = re.compile(r'([\w|<|>]+)')
        self.pad_idx, self.unk_idx, self.eos_idx = 0, 1, 2
        self._vocab = []
        return

    @property
    def vocab(
        self
    ) -> list:
        return self._vocab

    def set_vocab(
        self,
        new_vocab: list,
        add_functional_token: bool
    ) -> None:
        if add_functional_token:
            self._vocab = ['<pad>', '<unk>', '<eos>'] + new_vocab + ['<bos>']
        else:
            self._vocab = new_vocab
        return

    def tokenize(
        self,
        instr: str,
        unk_filter: bool = False
    ) -> list:
        tokens = [tk.strip().lower() for tk in self.tokenizer.split(instr.strip()) if len(tk.strip()) > 0]
        if unk_filter:
            return [tk if tk in self.vocab else '<unk>' for tk in tokens]
        else:
            return tokens

    def tk2id(
        self,
        token: str
    ) -> int:
        if token in self.vocab:
            return self.vocab.index(token)
        else:
            return self.unk_idx


class CVutils():
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__()
        self.config = config
        # sub-feature size init
        self.pose_repeat = config['r2r_env']['pose_repeat']
        self.pose_space = config['r2r_env']['pose_space']
        self.pose_feature_size = self.pose_repeat * self.pose_space
        self.pano_space = config['r2r_env']['pano_space']
        self.pano_feature_size = config['r2r_env']['pano_feature_size']

        # feature size init
        self.word_embedding_size = config['r2r_env']['word_embedding_size']
        self.vision_feature_size = self.pano_feature_size + self.pose_feature_size
        self.action_feature_size = self.vision_feature_size

        # pre-loading feature
        self.pano_features = self._init_pano_features()
        self.abs_pose_features = self._init_abs_pose_features()
        self.pano_rel_pose_features = self._init_pano_rel_pose_features()
        return

    def _init_pano_features(
        self
    ) -> dict:
        try:
            pano_features = np.load(self.config['r2r_env']['pano_feature'] + '.npy', allow_pickle=True)[()]
            print('Loading panoramic features from %s.npy' % self.config['r2r_env']['pano_feature'])
        except FileNotFoundError:
            print('Loading panoramic features from %s.tsv' % self.config['r2r_env']['pano_feature'])
            csv.field_size_limit(sys.maxsize)
            pano_features = {}
            tsv_field_names = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
            with open(self.config['r2r_env']['pano_feature'] + '.tsv', "rt") as tsv_in_file:
                with tqdm(total=sum(1 for _ in tsv_in_file)) as pbar:
                    tsv_in_file.seek(0)
                    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_field_names)
                    for item in reader:
                        long_id = item['scanId'] + '_' + item['viewpointId']
                        pano_features[long_id] = np.frombuffer(
                            base64.b64decode(item['features']),
                            dtype=np.float32
                        ).reshape((1, 1, self.pano_space, self.pano_feature_size))
                        pbar.update(1)
            np.save(self.config['r2r_env']['pano_feature'] + '.npy', pano_features)
            # (time_step, batch_idx, num_view, view_dim)
        return pano_features

    def _init_abs_pose_features(
        self
    ) -> list:
        """
        36 x (1, 1, 128)
        """
        abs_pose_features = []
        rad30 = np.deg2rad(30)
        for abs_view_index in range(self.pano_space):
            abs_pose_feature = np.zeros((1, 1, self.pose_feature_size), np.float32)
            heading = (abs_view_index % 12) * rad30
            elevation = (abs_view_index // 12 - 1) * rad30
            abs_pose_feature[0, 0, self.pose_repeat * 0:self.pose_repeat * 1] = np.sin(heading)
            abs_pose_feature[0, 0, self.pose_repeat * 1:self.pose_repeat * 2] = np.cos(heading)
            abs_pose_feature[0, 0, self.pose_repeat * 2:self.pose_repeat * 3] = np.sin(elevation)
            abs_pose_feature[0, 0, self.pose_repeat * 3:self.pose_repeat * 4] = np.cos(elevation)
            abs_pose_features.append(abs_pose_feature)
        return abs_pose_features

    def _init_pano_rel_pose_features(
        self
    ) -> list:
        """
        36 x (1, 1, 36, 128)
        """
        pano_rel_pose_features = []
        rad30 = np.deg2rad(30)
        for view_index in range(self.pano_space):
            pano_rel_pose_feature = np.zeros((1, 1, self.pano_space, self.pose_feature_size), np.float32)
            for abs_view_index in range(self.pano_space):
                rel_view_index = (abs_view_index - view_index) % 12 + (abs_view_index // 12) * 12
                rel_heading = (rel_view_index % 12) * rad30
                rel_elevation = (rel_view_index // 12 - 1) * rad30
                pano_rel_pose_feature[0, 0, abs_view_index, self.pose_repeat * 0:self.pose_repeat * 1] = np.sin(rel_heading)
                pano_rel_pose_feature[0, 0, abs_view_index, self.pose_repeat * 1:self.pose_repeat * 2] = np.cos(rel_heading)
                pano_rel_pose_feature[0, 0, abs_view_index, self.pose_repeat * 2:self.pose_repeat * 3] = np.sin(rel_elevation)
                pano_rel_pose_feature[0, 0, abs_view_index, self.pose_repeat * 3:self.pose_repeat * 4] = np.cos(rel_elevation)
            pano_rel_pose_features.append(pano_rel_pose_feature)
        return pano_rel_pose_features

    def get_long_id(
        self,
        scan_id: str,
        vp_id: str
    ) -> str:
        return scan_id + '_' + vp_id

    def get_vision_features(
        self,
        location_infos: list,
        batch: bool = True
    ) -> torch.Tensor:
        vision_features = []
        for location_info in location_infos:
            vision_features.append(
                self.get_vision_feature(
                    scan_id=location_info.scan_id,
                    vp_id=location_info.vp_id,
                    view_index=location_info.view_index
                )
            )
        return torch.cat(vision_features, dim=1 if batch else 0).to(self.config['device'])

    def get_vision_feature(
        self,
        scan_id: str,
        vp_id: str,
        view_index: int
    ) -> torch.Tensor:
        """
        (traj_len, batch_size, view_angle, vision_feature_size)
        (1, 1, 36, 2176)
        """
        long_id = self.get_long_id(scan_id, vp_id)
        pano_feature = torch.from_numpy(self.pano_features[long_id])
        pano_rel_pose_feature = torch.from_numpy(self.pano_rel_pose_features[view_index])
        return torch.cat((pano_feature, pano_rel_pose_feature), dim=3)

    def get_abs_pose_features(
        self,
        location_infos: list,
        batch: bool = True
    ) -> torch.Tensor:
        abs_pose_features = []
        for location_info in location_infos:
            abs_pose_features.append(
                self.get_abs_pose_feature(
                    view_index=location_info.view_index
                )
            )
        return torch.cat(abs_pose_features, dim=1 if batch else 0).to(self.config['device'])

    def get_abs_pose_feature(
        self,
        view_index: int
    ) -> torch.Tensor:
        """
        (traj_len, batch_size, feature_size)
        (1, 1, 128)
        """
        return torch.from_numpy(self.abs_pose_features[view_index])

    def get_candidate_action_features(
        self,
        location_infos: list,
        batch: bool = True
    ) -> torch.Tensor:
        candidate_action_features = []
        for location_info in location_infos:
            candidate_action_features.append(
                self.get_candidate_action_feature(
                    scan_id=location_info.scan_id,
                    vp_id=location_info.vp_id,
                    intents=location_info.action_info.intents,
                    rel_headings=location_info.action_info.rel_headings,
                    rel_elevations=location_info.action_info.rel_elevations
                )
            )
        return torch.cat(candidate_action_features, dim=1 if batch else 0).to(self.config['device'])

    def get_candidate_action_feature(
        self,
        scan_id: str,
        vp_id: str,
        intents: np.ndarray,
        rel_headings: np.ndarray,
        rel_elevations: np.ndarray
    ) -> torch.Tensor:
        """
        (traj_len, batch_size, action_space, feature_size)
        (1, 1, 14, 2176)
        """
        candidate_action_feature = []
        for action_idx, (intent, rel_heading, rel_elevation) in enumerate(zip(intents, rel_headings, rel_elevations)):
            if intent == -1 or action_idx == 0:
                candidate_action_feature.append(torch.zeros(1, 1, 1, self.action_feature_size))
            else:
                long_id = self.get_long_id(scan_id, vp_id)
                action_img_feature = torch.from_numpy(self.pano_features[long_id][:, :, intent:intent + 1, :])

                action_rel_pose_feature = torch.zeros(1, 1, 1, self.pose_space * self.pose_repeat)
                action_rel_pose_feature[0, 0, 0, 0 * self.pose_repeat:1 * self.pose_repeat] = math.sin(rel_heading)
                action_rel_pose_feature[0, 0, 0, 1 * self.pose_repeat:2 * self.pose_repeat] = math.cos(rel_heading)
                action_rel_pose_feature[0, 0, 0, 2 * self.pose_repeat:3 * self.pose_repeat] = math.sin(rel_elevation)
                action_rel_pose_feature[0, 0, 0, 3 * self.pose_repeat:4 * self.pose_repeat] = math.cos(rel_elevation)

                candidate_action_feature.append(
                    torch.cat((action_img_feature, action_rel_pose_feature), dim=3)
                )
        return torch.cat(candidate_action_feature, dim=2)


def tokenize_one_data(
    nlp_utils: NLPutils,
    data: dict,
    max_length: int = None
) -> list:
    tokens = (nlp_utils.tokenize(data['instr_str'], unk_filter=True) + ['<eos>'])[:max_length]
    return np.array([nlp_utils.tk2id(token) for token in tokens], dtype=np.int)


def load_datasets(
    config: dict,
    scan_ids: list
) -> (dict, dict, NLPutils):
    # Loading datasets
    data_num, path_set, datasets = 0, {}, {}
    splits = ['train', 'val_seen', 'val_unseen', 'test']
    if config['args']['aug_data']:
        splits.append('train_aug')
    for split in splits:
        instr_map = []
        with open(config['r2r_env']['dataset_dir'] + 'R2R_%s.json' % split) as file_name:
            for data in json.load(file_name):
                data['expert'] = []
                # build path_set
                if len(scan_ids) == 0 or (len(scan_ids) > 0 and data['scan'] in scan_ids):
                    path_set.update({data['path_id']: data})
                    # map instruction to path_id
                    for idx, instr in enumerate(data['instructions']):
                        instr_map.append(
                            {
                                'instr_id': data_num + len(instr_map),
                                'instr_str': instr,
                                'path_id': data['path_id'],
                                'idx': idx
                            }
                        )
        data_num += len(instr_map)
        if split == 'train_aug':
            datasets['train'] += instr_map
        else:
            datasets.update({split: instr_map})

    nlp_utils = NLPutils()
    vocab = []
    vocab_path = '/'.join(config['r2r_env']['word_embedding'].split('/')[:-1]) + '/vocab.txt'
    if os.path.isfile(vocab_path) and os.path.isfile(config['r2r_env']['word_embedding'] + '.pt'):
        print('Load vocabulary from %s' % vocab_path)
        with open(vocab_path, 'r') as txt_file:
            for word in txt_file.readlines():
                vocab.append(word.rstrip('\n'))
        nlp_utils.set_vocab(vocab, add_functional_token=False)
    else:
        print('Build up vocabulary...')
        for data in datasets['train']:
            data['instr_tk'] = nlp_utils.tokenize(data['instr_str']) + ['<eos>']
        with tqdm(total=len(datasets['train'])) as pbar:
            for data in datasets['train']:
                for instr_token in data['instr_tk'][:-1]:
                    if instr_token not in vocab and instr_token != '<unk>':
                        vocab.append(instr_token)
                pbar.update(1)
        nlp_utils.set_vocab(vocab, add_functional_token=True)

    print('Tokenize the instructions...')
    with tqdm(total=data_num) as pbar:
        for split, datas in datasets.items():
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            instr_tk_ids = pool.starmap(tokenize_one_data, ([(nlp_utils, data, config['args']['max_len']) for data in datas]))
            for data, instr_tk_id in zip(datas, instr_tk_ids):
                data['instr_tk_id'] = instr_tk_id
            pbar.update(len(datas))

    if config['args']['load_expert']:
        print('Loading expert demonstrations from %s' % config['r2r_env']['expert_dir'])
        expert = {}
        with open(config['r2r_env']['expert_dir'], 'r') as in_file:
            tsv_reader = csv.reader(in_file, delimiter='\t')
            for row in tsv_reader:
                path_id = int(row[0])
                seq_action = [int(act) for act in re.findall(r'\d+', row[1])]
                expert.update({path_id: seq_action})
        # store expert trajectory into dictionary
        for path_id, data in path_set.items():
            if path_id in expert.keys():
                data['expert'] = expert[path_id]
    return path_set, datasets, nlp_utils


def main():
    return


if __name__ == '__main__':
    main()
