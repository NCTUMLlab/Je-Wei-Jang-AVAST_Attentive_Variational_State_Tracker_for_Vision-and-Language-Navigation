import os
import torch
import numpy as np
from utils import load_config, print_log
from utils import init_tb_writer, init_agent
from env.rxr_env import RxREnv
from env.env_utils import Statistic
from agent.replay_memory import ReplayMemory
from agent.agent_reinforce import AgentReinforce


@torch.no_grad()
def run_data_parallel(
    data_indices: list,
    split: str,
    env: RxREnv,
    gen_gif: bool,
    agent: AgentReinforce or None,
    evaluate: bool,
) -> (Statistic, np.ndarray, list, list):
    # setup recorder, expert
    trajs = [[] for _ in range(len(data_indices))]
    returns = np.zeros(len(data_indices))
    expert_lens = [0 for _ in range(len(data_indices))]

    # setup environment
    dones, location_infos = env.reset(data_indices, split, gen_gif)
    instr_embed, instr_mask, hiddens = agent.obs_encoder.instr.encode(
        [torch.tensor(instr, dtype=torch.long) for instr in env.instrs]
    )
    last_action_features = agent.get_init_action(len(data_indices))

    # run one episode
    while True:
        actions, candidate_action_features, hiddens = agent.act(
            location_infos=location_infos,
            instr_embed=instr_embed,
            instr_mask=instr_mask,
            last_action_features=last_action_features,
            hiddens=hiddens,
            evaluate=evaluate
        )

        # mask done action & interact with environment
        actions[dones >= 1] = -1
        rewards, dones, next_location_infos = env.step(actions)

        # update return
        returns += rewards

        # store trajectory
        if not evaluate:
            for parallel_idx, (done, location_info, action, reward, expert_len) in enumerate(zip(dones, location_infos, actions, rewards, expert_lens)):
                if done <= 1:
                    trajs[parallel_idx].append([location_info, action, reward, expert_len])

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
    return stats, returns, trajs, env.instrs


def rollout(
    config: dict,
    split: str,
    env: RxREnv,
    replay_memory_agent: ReplayMemory or None,
    agent: AgentReinforce,
    it_now: int,
    evaluate: bool,
    beam_size: int = 8
) -> (np.ndarray, float, float, Statistic):
    env.set_env('dict')

    # rollout
    loss_list_all, alpha = np.zeros(5), 0
    if evaluate:
        # set parallel size
        parallel_size = config['r2r_env']['mp']['evaluate_parallel']

        # rollout all data
        returns, stats = [], Statistic([], [], [], [], [], [], [])
        for it_evaluate in range(env.get_it_num(split, parallel_size)):
            # get data_indices
            data_indices = env.get_data_indices(split, it_evaluate, parallel_size, evaluate)

            # rollout one parallel data
            stats_tmp, returns_tmp, _, _ = run_data_parallel(
                data_indices=data_indices, split=split, env=env,
                gen_gif=False, agent=agent, evaluate=evaluate
            )

            # record returns, statistics
            returns += returns_tmp.tolist()
            stats += stats_tmp
    else:
        assert split == 'train'
        # set parallel size
        parallel_size = config['r2r_env']['mp']['training_parallel'] // beam_size

        # get data_indices
        data_indices = env.get_data_indices(split, it_now, parallel_size, evaluate)

        for _ in range(beam_size):
            stats, returns, trajs, instrs = run_data_parallel(
                data_indices=data_indices, split=split, env=env,
                gen_gif=False, agent=agent, evaluate=evaluate
            )

            # store in replay memory
            replay_memory_agent.append(trajs, instrs)

        # update
        agent_mems = replay_memory_agent.sample(config['agent']['train']['learning']['batch_size'])
        loss_list_all, alpha = agent.train(agent_mems)

    # average the record of return, statistic
    return_average = np.average(returns)
    stat_average = stats.get_average()

    # log
    print_log(it_now, 0, env.iterations, alpha, loss_list_all, return_average, stat_average)
    return loss_list_all, alpha, return_average, stat_average


def train_test(
    config: dict
) -> None:
    # initialize tensorboard, environment, agent, replaymemory
    writer = init_tb_writer(config)
    env = RxREnv(config)
    agent = init_agent(config, env)
    replay_memory_agent = ReplayMemory(config, env, on_policy=True)

    # rl training
    for it_now in range(config['agent']['train']['learning']['iteration'] + 1):
        # training stage
        if it_now > 0:
            loss_list_all, alpha, _, _ = rollout(
                config=config, split='train', env=env,
                replay_memory_agent=replay_memory_agent, agent=agent,
                it_now=it_now, evaluate=False
            )

            # tensorboard for training information
            writer.add_scalar('loss/alpha', alpha, it_now)
            writer.add_scalar('loss/critic_1', loss_list_all[0], it_now)
            writer.add_scalar('loss/critic_2', loss_list_all[1], it_now)
            writer.add_scalar('loss/policy', loss_list_all[2], it_now)
            writer.add_scalar('loss/entropy', loss_list_all[3], it_now)
            writer.add_scalar('loss/KL divergence', loss_list_all[4], it_now)
        # evaluation stage
        if it_now % 50 == 0:
            # switch to evaluation mode
            agent.change_mode(is_train=False, ignore_state_tracker=True)
            with torch.no_grad():
                print('-----test begin-----')
                return_it_now = {}
                for split in ['val_seen', 'val_unseen']:
                    # evaluate from rollout
                    _, _, return_average, stat_average = rollout(
                        config=config, split=split, env=env,
                        replay_memory_agent=None, agent=agent,
                        it_now=it_now, evaluate=True
                    )
                    return_it_now.update({split: return_average})

                    # tensorboard for evaluation information
                    writer.add_scalar('%s/navigation error' % split, stat_average.nav_error[0], it_now)
                    writer.add_scalar('%s/path length' % split, stat_average.path_len[0], it_now)
                    writer.add_scalar('%s/success rate' % split, stat_average.succ_rate[0], it_now)
                    writer.add_scalar('%s/success rate weighted by path length' % split, stat_average.succ_w_path_len[0], it_now)
                    writer.add_scalar('%s/self stop rate' % split, stat_average.self_stop_rate[0], it_now)
                    writer.add_scalar('%s/coverage weighted by length score' % split, stat_average.cov_w_len_score[0], it_now)
                    writer.add_scalar('%s/reward' % split, return_average, it_now)
                print('-----test end  -----')
            # save
            # agent.save(config['args']['exp_name'], it_now)
            # switch to training mode
            agent.change_mode(is_train=True, ignore_state_tracker=True)
    return


def main():
    # load args, config
    config = load_config('/root/mount/AVAST_R2R/tasks/config.json')
    assert config['args']['agent'] == 'reinforce'

    # select mode
    if config['args']['mode'] == 'train':
        # delete old weight
        if os.path.isdir(config['save_dir']):
            os.system('rm -r %s' % config['save_dir'])
            os.system('mkdir %s' % config['save_dir'])
        train_test(config=config)
    elif config['args']['mode'] == 'test':
        raise NotImplementedError
    else:
        print('Invalid mode')
    return


if __name__ == '__main__':
    main()
