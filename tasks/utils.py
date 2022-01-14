import os
import sys
import json
import time
import random
import argparse
import torch
import numpy as np
from tensorboardX import SummaryWriter
from env.rxr_env import RxREnv
from env.env_utils import Statistic
from agent.agent_seq2seq import AgentSeq2Seq
from agent.agent_sacd import AgentSACD
from agent.agent_reinforce import AgentReinforce


def print_log(
    it_now: int,
    lecture: int,
    iterations: list,
    alpha: float,
    loss_list: np.ndarray,
    return_average: float,
    stat_average: Statistic
) -> None:
    if it_now >= 0:
        log = 'it:%6d [lec:%2d] | it: %.1f, alpha: %.2f, L_q1: %.3f, L_q2: %.3f, L_pi: %.3f, L_ent: %.3f, KLD: %.3f, R: %4.2f, PL: %5.2f, NE: %5.2f, SR: %.2f, SPL: %.2f, STP: %.2f, CLS: %.2f'\
            % (it_now, lecture,
                np.average(iterations), alpha,
                loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4],
                return_average,
                stat_average.path_len[0], stat_average.nav_error[0],
                stat_average.succ_rate[0], stat_average.succ_w_path_len[0],
                stat_average.self_stop_rate[0], stat_average.cov_w_len_score[0])
    else:
        log = 'R: %4.3f, PL: %6.3f, NE: %6.3f, SR: %.3f, SPL: %.3f, STP: %.3f, CLS: %.3f'\
            % (return_average,
                stat_average.path_len[0], stat_average.nav_error[0],
                stat_average.succ_rate[0], stat_average.succ_w_path_len[0],
                stat_average.self_stop_rate[0], stat_average.cov_w_len_score[0])
    print(log)
    sys.stdout.flush()
    return


def load_config(
    config_dir: str,
    show_info: bool = True
) -> (argparse.Namespace, dict):
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--state_tracker')
    parser.add_argument('--agent')
    parser.add_argument('--additional_track', default='pose')
    parser.add_argument('--max_len', default=-1, type=int)
    parser.add_argument('--demo_activate', default=False, action='store_true')
    parser.add_argument('--curriculum', default=False, action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--load_dir', default='')
    parser.add_argument('--load_pre_trained_dir', default='')
    parser.add_argument('--exp_name', default='tmp')
    parser.add_argument('--rendering_idx', default=-1, type=int)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--aug_data', default=False, action='store_true')
    parser.add_argument('--load_expert', default=True, action='store_true')
    args = parser.parse_args()

    # load config
    with open(config_dir) as file_name:
        config = json.load(file_name)

    if args.mode == 'train':
        assert args.agent == 'sacd' or args.agent == 'reinforce'
    elif args.mode == 'test':
        assert args.load_dir != ''
    elif args.mode == 'pre_train':
        assert args.agent == 'seq2seq'
    else:
        raise NotImplementedError

    if args.additional_track not in ['pose', 'action']:
        raise NotImplementedError

    config['args'] = {
        'mode': args.mode,
        'state_tracker': args.state_tracker,
        'agent': args.agent,
        'additional_track': args.additional_track,
        'max_len': args.max_len if args.max_len else None,
        'demo_activate': args.demo_activate,
        'curriculum': args.curriculum,
        'load_dir': args.load_dir,
        'load_pre_trained_dir': args.load_pre_trained_dir,
        'exp_name': args.exp_name,
        'rendering_idx': args.rendering_idx,
        'verbose': args.verbose,
        'aug_data': args.aug_data,
        'load_expert': args.load_expert
    }
    config['seed'] = args.seed

    # setting random seed
    os.environ['PYTHONHASHSEED'] = str(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # print info
    if show_info and config['args']['mode'] != 'test':
        print(json.dumps(config, indent=2) + '\n')

    # select device
    if torch.cuda.is_available():
        config['device'] = torch.device(config['device'])
    else:
        config['device'] = torch.device('cpu')
    return config


def init_tb_writer(
    config: dict
) -> SummaryWriter:
    result_dir = config['result_dir']
    # remove tmp
    if os.path.isdir(os.path.join(result_dir, config['args']['exp_name'])):
        os.system('rm -r %s' % os.path.join(result_dir, config['args']['exp_name']))
        time.sleep(5)
    # set result dir
    return SummaryWriter(os.path.join(result_dir, config['args']['exp_name']))


def init_agent(
    config: dict,
    env: RxREnv
) -> AgentSACD or AgentReinforce or AgentSeq2Seq:
    if config['args']['agent'] == 'sacd':
        agent = AgentSACD(config, env)
    elif config['args']['agent'] == 'reinforce':
        agent = AgentReinforce(config, env)
    elif config['args']['agent'] == 'seq2seq':
        agent = AgentSeq2Seq(config, env)
    else:
        raise NotImplementedError
    if config['args']['load_pre_trained_dir']:
        agent.load_pre_train(config['args']['load_pre_trained_dir'])
    return agent


def get_lecture(
    evaluate: bool,
    it_now: int,
    config: dict
) -> int:
    if evaluate or not config['args']['curriculum']:
        return 0
    else:
        agent_config = config['agent'][config['args']['mode']]
        total_iteration = agent_config['learning']['iteration']
        progress = agent_config['replay_memory']['demonstration']['curriculum_progress']
        last_lecture = agent_config['replay_memory']['demonstration']['last_lecture']
        if progress > 0:
            return min(1 + int(progress * (it_now / total_iteration)), last_lecture)
        else:
            return last_lecture


def main():
    return


if __name__ == '__main__':
    main()
