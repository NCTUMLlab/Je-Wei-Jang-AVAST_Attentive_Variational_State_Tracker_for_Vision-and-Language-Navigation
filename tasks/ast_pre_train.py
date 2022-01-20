import os
import torch
import numpy as np
from utils import load_config, print_log
from utils import init_tb_writer, init_agent
from env.rxr_env import RxREnv
from env.env_utils import Statistic
from agent.agent_seq2seq import AgentSeq2Seq


def run_data_parallel(
    data_indices: list,
    split: str,
    env: RxREnv,
    gen_gif: bool,
    agent: AgentSeq2Seq,
    evaluate: bool,
    act_by: str = 'policy'
) -> (Statistic, np.ndarray, list, list):
    # setup recorder, expert
    pair_datas = {
        'logits': [], 'labels': [],
        'q1s': [], 'q2s': [], 'curriculum_rewards': []
    }
    returns = np.zeros(len(data_indices))
    expert_trajs = env.get_expert_trajs(data_indices, split)

    # setup environment
    dones, location_infos = env.reset(data_indices, split, gen_gif)
    instr_embed, instr_mask, hiddens = agent.obs_encoder.instr.encode(
        [torch.tensor(instr, dtype=torch.long) for instr in env.instrs]
    )
    last_action_features = agent.get_init_action(len(data_indices))

    # run one episode
    while True:
        actions, outputs, candidate_action_features, hiddens = agent.act(
            location_infos=location_infos,
            instr_embed=instr_embed,
            instr_mask=instr_mask,
            last_action_features=last_action_features,
            hiddens=hiddens,
            evaluate=evaluate,
            act_by=act_by
        )

        if not evaluate:
            actions = np.zeros(len(data_indices), dtype=np.int)
            for parallel_idx, expert_traj in enumerate(expert_trajs):
                actions[parallel_idx] = expert_traj[env.iterations[parallel_idx]]

        # mask done action & interact with environment
        actions[dones >= 1] = -1
        rewards, dones, next_location_infos = env.step(actions)

        # update return
        returns += rewards

        # store trajectory
        if not evaluate:
            for parallel_idx, done in enumerate(dones):
                if done <= 1:
                    iteration = env.iterations[parallel_idx]
                    expert_len = len(expert_trajs[parallel_idx])
                    pair_datas['logits'].append(
                        outputs['logits'][parallel_idx: parallel_idx + 1]
                    )
                    pair_datas['labels'].append(
                        actions[parallel_idx]
                    )
                    pair_datas['q1s'].append(
                        outputs['q1s'][parallel_idx: parallel_idx + 1]
                    )
                    pair_datas['q2s'].append(
                        outputs['q2s'][parallel_idx: parallel_idx + 1]
                    )
                    pair_datas['curriculum_rewards'].append(
                        2 * env.reward_scale * (agent.gamma ** (expert_len - iteration - 1))
                    )

        # terminate all trajectories
        if all(dones >= 1):
            stats = env.get_statistics()
            break

        # update states (the last thing in iteration)
        location_infos = next_location_infos
        last_action_features = torch.stack(
            [candidate_action_features[parallel_idx, action_idx, :] for parallel_idx, action_idx in enumerate(actions)],
            dim=0
        )
    return stats, returns, pair_datas


def rollout(
    config: dict,
    split: str,
    env: RxREnv,
    agent: AgentSeq2Seq,
    it_now: int,
    evaluate: bool,
    act_by: str = 'policy'
) -> (np.ndarray, float, Statistic):
    env.set_env('dict')

    # rollout
    loss_list = np.zeros(5)  # critic1, critic2, policy, entropy, kld
    if evaluate:
        # set parallel size
        parallel_size = config['r2r_env']['mp']['evaluate_parallel']

        # rollout all data
        returns, stats = [], Statistic([], [], [], [], [], [], [])
        for it_evaluate in range(env.get_it_num(split, parallel_size)):
            # get data_indices
            data_indices = env.get_data_indices(split, it_evaluate, parallel_size, evaluate)

            # rollout one parallel data
            stats_tmp, returns_tmp, _ = run_data_parallel(
                data_indices=data_indices, split=split, env=env,
                gen_gif=False, agent=agent, evaluate=evaluate,
                act_by=act_by
            )

            # record returns, statistics
            returns += returns_tmp.tolist()
            stats += stats_tmp
    else:
        assert split == 'train'
        # set parallel size
        parallel_size = config['r2r_env']['mp']['training_parallel']

        # get data_indices
        data_indices = env.get_data_indices(split, it_now, parallel_size, evaluate)

        # rollout one parallel data
        stats, returns, pair_datas = run_data_parallel(
            data_indices=data_indices, split=split, env=env,
            gen_gif=False, agent=agent, evaluate=evaluate
        )

        # update
        loss_list += agent.train(pair_datas)

    # average the record of loss, return, statistic
    return_average = np.average(returns)
    stat_average = stats.get_average()

    # log
    print_log(it_now, 0, env.iterations, 0, loss_list, return_average, stat_average)
    return loss_list, return_average, stat_average


def test(
    config: dict
) -> None:
    # initialize environment, agent
    env = RxREnv(config)
    agent = init_agent(config, env)
    agent.load(config['args']['load_dir'])

    # switch to evaluation mode
    agent.change_mode(is_train=False)
    with torch.no_grad():
        print('-----test begin-----')
        for split in ['val_seen', 'val_unseen']:
            print(split)
            # evaluate from rollout
            _, return_average, stat_average = rollout(
                config=config, split=split, env=env,
                agent=agent, it_now=-1, evaluate=True
            )
            _, return_average, stat_average = rollout(
                config=config, split=split, env=env,
                agent=agent, it_now=-1, evaluate=True,
                act_by='critic'
            )
        print('-----test end  -----')
    return


def train_test(
    config: dict
) -> None:
    # initialize tensorboard, environment, agent
    writer = init_tb_writer(config)
    env = RxREnv(config)
    agent = init_agent(config, env)

    # bc training
    for it_now in range(config['agent']['pre_train']['learning']['iteration'] + 1):
        # training stage
        if it_now > 0:
            loss_list_all, _, _ = rollout(
                config=config, split='train', env=env,
                agent=agent, it_now=it_now, evaluate=False
            )
            # tensorboard for training information
            writer.add_scalar('loss/critic_1', loss_list_all[0], it_now)
            writer.add_scalar('loss/critic_2', loss_list_all[1], it_now)
            writer.add_scalar('loss/policy', loss_list_all[2], it_now)
            writer.add_scalar('loss/entropy', loss_list_all[3], it_now)
            writer.add_scalar('loss/KL divergence', loss_list_all[4], it_now)

        # evaluation stage
        if it_now % 50 == 0:
            # switch to evaluation mode
            agent.change_mode(is_train=False)
            with torch.no_grad():
                print('-----test begin-----')
                for split in ['val_seen', 'val_unseen']:
                    # evaluate from rollout
                    _, return_average, stat_average = rollout(
                        config=config, split=split, env=env,
                        agent=agent, it_now=it_now, evaluate=True
                    )

                    # tensorboard for evaluation information
                    writer.add_scalar('%s/navigation error' % split, stat_average.nav_error[0], it_now)
                    writer.add_scalar('%s/path length' % split, stat_average.path_len[0], it_now)
                    writer.add_scalar('%s/success rate' % split, stat_average.succ_rate[0], it_now)
                    writer.add_scalar('%s/success rate weighted by path length' % split, stat_average.succ_w_path_len[0], it_now)
                    writer.add_scalar('%s/self stop rate' % split, stat_average.self_stop_rate[0], it_now)
                    writer.add_scalar('%s/coverage weighted by length score' % split, stat_average.cov_w_len_score[0], it_now)
                    writer.add_scalar('%s/reward' % split, return_average, it_now)

                    # evaluate from rollout
                    _, return_average, stat_average = rollout(
                        config=config, split=split, env=env,
                        agent=agent, it_now=it_now, evaluate=True,
                        act_by='critic'
                    )

                    # tensorboard for evaluation information
                    writer.add_scalar('%s_q/navigation error' % split, stat_average.nav_error[0], it_now)
                    writer.add_scalar('%s_q/path length' % split, stat_average.path_len[0], it_now)
                    writer.add_scalar('%s_q/success rate' % split, stat_average.succ_rate[0], it_now)
                    writer.add_scalar('%s_q/success rate weighted by path length' % split, stat_average.succ_w_path_len[0], it_now)
                    writer.add_scalar('%s_q/self stop rate' % split, stat_average.self_stop_rate[0], it_now)
                    writer.add_scalar('%s_q/coverage weighted by length score' % split, stat_average.cov_w_len_score[0], it_now)
                    writer.add_scalar('%s_q/reward' % split, return_average, it_now)
                print('-----test end  -----')

            # save
            agent.save(config['args']['exp_name'], it_now)
            # switch to training mode
            agent.change_mode(is_train=True)
    return


def main():
    # load args, config
    config = load_config('/root/mount/AVAST_R2R/tasks/config.json')
    assert config['args']['agent'] == 'seq2seq'

    # select mode
    if config['args']['mode'] == 'pre_train':
        # delete old weight
        if os.path.isdir(config['save_dir']):
            os.system('rm -r %s' % config['save_dir'])
            os.system('mkdir %s' % config['save_dir'])
        train_test(config=config)
    elif config['args']['mode'] == 'test':
        test(config=config)
    else:
        print('Invalid mode')
    return


if __name__ == '__main__':
    main()
