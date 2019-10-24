from __future__ import print_function
import numpy as np
import csv
import argparse
import torch
import os
import copy
import operator
from env import Env
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from gym.spaces import Box, Discrete
from pathlib import Path
from algorithms.attention_sac import AttentionSAC
from utils.buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()

# for test
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# EPSILON = 0.1
# ALPHA = 0.1
# GAMMA = 0.9
# MAX_STEP = 50

# np.random.seed(0)

def data_reprocess(replay_buffer_data, length):
    all_result = []
    for a in range(6):
        result = copy.deepcopy(replay_buffer_data[a][:length])
        all_result.append(result)
    
    return all_result

def run(config):

    # data_ave = []
    # data_success = 0
    # data_false = 0
    # data_average = 0
    # with open('check_data_file/episode_result_4_4_4.csv')as c:
    #
    #     r = list(csv.reader(c))
    #     index_size = len(r)
    #     for i in r:
    #         if float(i[0])>0:
    #             data_ave.append(float(i[2]))
    #             if float(i[2]) == 6:
    #                 data_success = data_success+1
    #             else:
    #                 data_false = data_false +1
    #             if float(i[0])%1000 == 0:
    #                 data_average = np.mean(data_ave)
    #                 a_res = [data_average,data_success,data_false]
    #                 with open('check_data_file/analyse.csv','a',newline='')as file_a:
    #                     file_a_csv = csv.writer(file_a)
    #                     file_a_csv.writerow(a_res)
    #                 date_ave = []
    #                 data_success = 0
    #                 data_false = 0
    #                 data_average = 0
    #             if float(i[0]) == index_size-1:
    #                 data_average = np.mean(date_ave)
    #                 a_res = [data_average,data_success,data_false]
    #                 with open('check_data_file/analyse.csv','a',newline='')as file_a:
    #                     file_a_csv = csv.writer(file_a)
    #                     file_a_csv.writerow(a_res)
    # c.close()

    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(1804)
    agent_num = 6
    np.random.seed(1804)
    e = Env(map_l=8, map_w=8, robot_num=agent_num)
    model = AttentionSAC.init_from_env(e,
                                        tau=config.tau,
                                        #    attend_tau=config.attend_tau,
                                        pi_lr=config.pi_lr,
                                        q_lr=config.q_lr,
                                        gamma=config.gamma,
                                        pol_hidden_dim=config.pol_hidden_dim,
                                        critic_hidden_dim=config.critic_hidden_dim,
                                        attend_heads=config.attend_heads,
                                        reward_scale=config.reward_scale)
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                    [obsp.shape[0] for obsp in e.observation_space],
                                    [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                    for acsp in e.action_space])

    seed = 0
    tseed = 0
    t = 0
    # for single env
    window_size = 12
    ep_dones_window = []
    ep_steps_window = []
    ep_rews_window = []
    ep_ready_steps_window = []
    rewards_zeros = tuple([[0 for i in range(model.nagents)]])
    rewards_all_zeros = np.stack(rewards_zeros)
    # for check
    file_num = 1
    case_size = 1000
    count_start = 60000
    result = []
    for ep_i in range(config.n_episodes):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                    ep_i + 1 + config.n_rollout_threads,
                                    config.n_episodes))
        # for test
        # np.random.seed(seed)
        # seed += 1
        false_number = 0
        true_number = 0
        if ep_i !=0:
            for  set_i in range(len(result)):
                if result[set_i] == False:
                    false_number = false_number+1
                else:
                    true_number = true_number+1
            result_ep = [ep_i-1,false_number,true_number]
            for step_i in range(len(step_list)):
                result_ep.append(step_list[step_i])
            result = []
            with open('check_data_file/episode_result_6_6_6.csv','a',newline='')as f:
                f_csv = csv.writer(f)
                f_csv.writerow(result_ep)
            f.close()
            with open('check_data_file/episode_result{}.txt'.format(1),'a') as file:
                file.write('episode {} : \n{}\n'.format(ep_i-1, result_ep))
            file.close()
        e.create_map()
        model.prep_rollouts(device='cpu')
        step_list = np.zeros(agent_num)
        total_reward_list = np.zeros(agent_num)
        success = np.zeros(
            (config.n_rollout_threads, model.nagents), dtype=bool)
        steps = np.zeros(
            (config.n_rollout_threads, model.nagents))
        
        # c = 0
        s = ''
        while any([robot.is_end is False for robot in e.robot_list]) and \
                all([step_list[i] < config.episode_length for i in range(len(step_list))]):
        # while all([step_list[i] < config.episode_length for i in range(len(step_list))]):
            dones = []
            # for check
            is_print = False
            for robot in e.robot_list:
                # 未到达目的地
                if robot.is_end is False:
                    ready, current_state = e.present_state(robot)
                robot.state = e.get_obs(robot)
                dones.append(robot.is_end)
                
            dones = tuple([dones])
            dones = np.stack(dones)
            success = np.logical_or(success, dones)
            steps += dones
            next_obs = tuple([[robot.state for robot in e.robot_list]])
            next_obs = np.stack(next_obs)
            
            if all([step_list[i] > 0 for i in range(len(step_list))]):
                if all(operator.eq(e.rewards[0], rewards_all_zeros[0])):
                    print('all zeros')
                else:
                    # c += 1
                    replay_buffer.push(e.obs, e.agent_actions, e.rewards, next_obs, dones)
                # replay_buffer.push(e.obs, e.agent_actions, e.rewards, next_obs, dones)
            
            e.obs = next_obs
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(next_obs[:, i])),
                                    requires_grad=False)
                        for i in range(model.nagents)]
            # for test
            # torch.manual_seed(tseed)
            # tseed += 1
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            e.agent_actions = agent_actions
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            
            reach_now = np.zeros(agent_num, dtype=np.bool)
            for robot in e.robot_list:
                # 已经到达目的地
                if robot.is_end:
                    robot.reward = 0
                    continue
                # set action for agent
                e.set_action(robot, actions[0][robot.id])
                if robot.is_ready:
                    step_list[robot.id] += 1
                # get reward
                robot.reward = e.get_reward(robot)
                total_reward_list[robot.id] += robot.reward
                # for check
                is_print = True
            #     # 刚刚到达目的地
            #     if robot.is_end:
            #         reach_now[robot.id] = True
            # # for check
            if ep_i > count_start:
                s = e.write_map2file(s, is_print)
            # # 更新刚到达目的地的agent的obs
            # for robot_id, robot in enumerate(e.robot_list):
            #     if reach_now[robot_id]:
            #         robot.state = e.get_obs(robot)robot
                
            rewards = tuple([[robot.reward for robot in e.robot_list]])
            e.rewards = np.stack(rewards)
            if e.is_end:
                if all(operator.eq(e.rewards[0], rewards_all_zeros[0])):
                    print('all zeros')
                else:
                    # c += 1
                    dones = tuple([[robot.is_end for robot in e.robot_list]])
                    dones = np.stack(dones)
                    success = np.logical_or(success, dones)
                    steps += dones
                    next_obs = tuple([[robot.state for robot in e.robot_list]])
                    next_obs = np.stack(next_obs)
                    replay_buffer.push(e.obs, e.agent_actions, e.rewards, next_obs, dones)
                
            t += config.n_rollout_threads
            
            if (len(replay_buffer) >= max(config.pi_batch_size, config.q_batch_size) and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_critic_updates):
                    sample = replay_buffer.sample(config.q_batch_size,
                                                to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                for u_i in range(config.num_pol_updates):
                    sample = replay_buffer.sample(config.pi_batch_size,
                                                to_gpu=config.use_gpu)
                    model.update_policies(sample, logger=logger)
                model.update_all_targets()
                model.prep_rollouts(device='cpu')
        # exit(0)
        dones = tuple([[robot.is_end for robot in e.robot_list]])
        result = np.stack(dones)[0]
        if not e.is_end:
            if all(operator.eq(e.rewards[0], rewards_all_zeros[0])):
                print('all zeros')
            else:
                # c += 1
                dones = tuple([[robot.is_end for robot in e.robot_list]])
                dones = np.stack(dones)
                success = np.logical_or(success, dones)
                steps += dones
                next_obs = tuple([[robot.state for robot in e.robot_list]])
                next_obs = np.stack(next_obs)
                replay_buffer.push(e.obs, e.agent_actions, e.rewards, next_obs, dones)
        # for check
        if ep_i > count_start and not e.is_end:
            if ep_i % case_size == 0:
                file_num += 1
            with open('check_data_file/unreach_case{}.txt'.format(file_num),'a') as file:
                file.write('episode {} : \n{}'.format(ep_i+1, s))
        # print(c)
        ep_dones = np.mean(success, axis=0)
        ep_steps = 1 - np.mean(steps / config.episode_length, axis=0)
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        ep_ready_steps = step_list
        # for single env
        ep_dones_window.append(copy.deepcopy(ep_dones))
        ep_steps_window.append(copy.deepcopy(ep_steps))
        ep_rews_window.append(copy.deepcopy(ep_rews))
        ep_ready_steps_window.append(copy.deepcopy(ep_ready_steps))
        if len(ep_dones_window) > window_size:
            ep_dones_window.pop(0)
            ep_steps_window.pop(0)
            ep_rews_window.pop(0)
            ep_ready_steps_window.pop(0)
        if len(ep_dones_window) == window_size:
            ep_rews = np.mean(ep_rews_window, axis=0).tolist()
            ep_dones = np.mean(ep_dones_window, axis=0).tolist()
            ep_steps = np.mean(ep_steps_window, axis=0).tolist()
            ep_ready_steps = np.mean(ep_ready_steps_window, axis=0).tolist()
            for a_i, a_ep_rew in enumerate(ep_rews):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
            for a_i, a_ep_done in enumerate(ep_dones):
                logger.add_scalar('agent%i/mean_episode_dones' % a_i, a_ep_done, ep_i)
            for a_i, a_ep_step in enumerate(ep_steps):
                logger.add_scalar('agent%i/mean_episode_steps' % a_i, a_ep_step, ep_i)
            for a_i, a_ep_ready_step in enumerate(ep_ready_steps):
                logger.add_scalar('agent%i/mean_episode_ready_steps' % a_i, a_ep_ready_step, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    # for test
    # obs_buffs = data_reprocess(replay_buffer.obs_buffs, replay_buffer.__len__())
    # next_obs_buffs = data_reprocess(replay_buffer.next_obs_buffs, replay_buffer.__len__())
    # ac_buffs = data_reprocess(replay_buffer.ac_buffs, replay_buffer.__len__())
    # rew_buffs = data_reprocess(replay_buffer.rew_buffs, replay_buffer.__len__())
    # np.save('check_data_file\{}_geek'.format('obs_buffs'), obs_buffs)
    # np.save('check_data_file\{}_geek'.format('next_obs_buffs'), next_obs_buffs)
    # np.save('check_data_file\{}_geek'.format('ac_buffs'), ac_buffs)
    # np.save('check_data_file\{}_geek'.format('rew_buffs'), rew_buffs)



    model.save(run_dir / 'model.pt')
    # e.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id",
                        default='test',
                        help="Name of environment")
    parser.add_argument("--model_name",
                        default='train',
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=150000, type=int)
    parser.add_argument("--episode_length", default=50, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_critic_updates", default=6, type=int,
                        help="Number of critic updates per update cycle")
    parser.add_argument("--num_pol_updates", default=6, type=int,
                        help="Number of policy updates per update cycle")
    parser.add_argument("--pi_batch_size",
                        default=1024, type=int,
                        help="Batch size for policy training")
    parser.add_argument("--q_batch_size",
                        default=1024, type=int,
                        help="Batch size for critic training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=48, type=int)
    parser.add_argument("--attend_heads", default=6, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.04, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()

    run(config)