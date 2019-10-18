import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AStarAgent
import heapq
import numpy as np
import sys

MSELoss = torch.nn.MSELoss()

class A_Star(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 map_height, map_width,
                 reward_scale=10.,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                
            sa_size (list of (int, int)): Size of state and action space for
            
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            
        """
        self.nagents = len(sa_size)

        self.agents = [AStarAgent(**params)
                         for params in agent_init_params]
        self.map_height = map_height
        self.map_width = map_width
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.niter = 0

    def a_star(self, agent, current_position, target, block=False):
        
        start_point = current_position
        end_point = target

        found = False
        map_size = list((self.map_width, self.map_height))
        cost_size = map_size + [5]
        cost = np.full(cost_size, fill_value=sys.maxsize, dtype=np.int64)
        cost[start_point[0], start_point[1], :] = [0, 0, 0, 0, 0]

        open_set = set()
        open_list = []
        heapq.heappush(open_list, (0, [start_point[0], start_point[1], 0, 0, 0, 0, 0]))
        open_set.add(str(start_point[0]) + "," + str(start_point[1]))
        close_list = []
        close_set = set()
        parents = np.zeros(map_size + [2], dtype=np.int64)

        end_point_str = str(end_point[0]) + ',' + str(end_point[1])
        robot_map = np.zeros(map_size)

        jam_weight = 60
        turn_weight = 20

        move_cost = 10
        if jam_weight != 0:
            for robot in self.agents:
                for cell in robot.path:
                    robot_map[cell[0], cell[1]] += 1

        while open_list and not found:
            min_value = heapq.heappop(open_list)
            current_point_state = [min_value[1][0], min_value[1][1]]
            close_list.append(current_point_state)
            close_set.add(str(current_point_state[0]) + ',' + str(current_point_state[1]))

            current_point_state_str = str(current_point_state[0]) + ',' + str(current_point_state[1])

            open_set.remove(current_point_state_str)
            x = current_point_state[0]
            y = current_point_state[1]
            if x == 0 or x == self.map_width-1 or y == 0 or y == self.map_height-1:
                avail_direction = []
            elif block and x == start_point[0] and y == start_point[1]:
                avail_direction = [0, 1, 2, 3]
                for robot in self.agents:
                    robot_current_position = np.array(robot.current_position)
                    if np.sum(np.abs(robot_current_position - start_point)) == 1:
                        #  np.linalg.norm(robot_current_position - start_point,ord=1) == 1:
                        if robot_current_position[0] - start_point[0] == 1:
                            if 3 in avail_direction:
                                avail_direction.remove(3)
                        elif robot_current_position[0] - start_point[0] == -1:
                            if 1 in avail_direction:
                                avail_direction.remove(1)
                        elif robot_current_position[1] - start_point[1] == 1:
                            if 0 in avail_direction:
                                avail_direction.remove(0)
                        elif robot_current_position[1] - start_point[1] == -1:
                            if 2 in avail_direction:
                                avail_direction.remove(2)

            else:
                avail_direction = [0, 1, 2, 3]
            np.random.shuffle(avail_direction)
            if len(avail_direction) != 0:
                search_point = [0, 0]
                back_direction = [0, 0]
                for a in avail_direction:
                    if a == 0:
                        search_point = [x, y + 1]
                        back_direction = [0, -1]
                    elif a == 1:
                        search_point = [x - 1, y]
                        back_direction = [1, 0]
                    elif a == 2:
                        search_point = [x, y - 1]
                        back_direction = [0, 1]
                    elif a == 3:
                        search_point = [x + 1, y]
                        back_direction = [-1, 0]
                    # 如果这些邻近方格不在close_list中，则计算search_point的代价
                    search_str = str(search_point[0]) + ',' + str(search_point[1])
                    if search_str not in close_set:
                        # if search_point not in close_list:
                        cost_g = cost[x, y, 0] + move_cost
                        cost_h = sum(abs(np.subtract(search_point, end_point))) * move_cost
                        cost_r = cost[x, y, 2] + robot_map[x, y] * jam_weight
                        if sum(abs(np.subtract(current_point_state, start_point))) == 0:
                            cost_t = 0
                        else:
                            po = [parents[x, y, 0], parents[x, y, 1]]
                            if sum(abs(np.array(po) - np.array(back_direction))) == 0:
                                cost_t = cost[x, y, 3]
                            else:
                                cost_t = cost[x, y, 3] + turn_weight
                        cost_f = cost_g + cost_h + cost_r + cost_t
                        # 如果这些邻近方格不在openList中，则把这些点加入到openList当中去
                        # 更新cost表和parents表
                        if len(open_list) == 0 or search_str not in open_set:
                            parents[search_point[0], search_point[1], :] = back_direction.copy()
                            cost[search_point[0], search_point[1], :] = [cost_g, cost_h, cost_r, cost_t, cost_f]
                            heapq.heappush(open_list, (cost_f, [search_point[0], search_point[1],
                                                                cost_g, cost_h, cost_r, cost_t, cost_f]))
                            open_set.add(search_str)
                        else:
                            # 如果该临近方格已经在openList中，则判断该格的总代价costF是否会更低，
                            # 如果是，则更新cost表和parents表
                            if cost[search_point[0], search_point[1], 4] > cost_f:
                                cost[search_point[0], search_point[1], :] = [cost_g, cost_h, cost_r, cost_t, cost_f]
                                parents[search_point[0], search_point[1], :] = back_direction.copy()
            if end_point_str in close_set:
                found = True
                break
        route = []
        if found:
            route.append(end_point)
            search_point = end_point
            while any(np.subtract(search_point, start_point)):
                search_point = list(np.add(route[-1],
                                           [parents[route[-1][0], route[-1][1], 0],
                                            parents[route[-1][0], route[-1][1], 1]]))
                route.append(search_point)
            route.reverse()
        else:
            route = []
        return route, found
    
    # added
    def cal_action(self, current_pos, next_pos):
        u_onehot = [0,0,0,0,0]
        if not current_pos or not next_pos:
            u_onehot[0] += 1
            return u_onehot
            
        distance = [int(next_pos[0] - current_pos[0]), 
                    int(next_pos[1] - current_pos[1])]
        delta_x = distance[0]
        delta_y = distance[1]
        
        if delta_x:
            if delta_x == 1:
                u_onehot[1] += 1
            else: 
                u_onehot[2] += 1
        
        elif delta_y:
            if delta_y == 1:
                u_onehot[3] += 1
            else:
                u_onehot[4] += 1
        
        else:
            u_onehot[0] += 1
            
        return u_onehot
            

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        actions = []
        # print(len(self.agents))
        # print(len(observations[0]))
        # exit(0)
        observations = observations[0]
        for agent, obs in zip(self.agents, observations):
            call_a_star = obs[0]
            pop_route_top = obs[1]
            current_pos = [obs[2], obs[3]]
            target_pos = [obs[4], obs[5]]
            next_pos = None
            
            agent.current_position = current_pos
            
        for agent, obs in zip(self.agents, observations):
            call_a_star = obs[0]
            pop_route_top = obs[1]
            current_pos = [obs[2], obs[3]]
            target_pos = [obs[4], obs[5]]
            next_pos = None
            
            if call_a_star:
                path, found = self.a_star(agent, current_pos, target_pos, block=True)
                if found:
                    agent.path = path
                    if len(path) > 1:
                        next_pos = agent.path[1]
            elif pop_route_top:
                agent.path.pop(0)
                if len(agent.path):
                    next_pos = agent.path[0]
                
            else:
                next_pos = current_pos
            
            actions.append(self.cal_action(current_pos, next_pos))
        
        # print(actions)
        # exit(0)
        return actions

    def prep_rollouts(self, device='cpu'):
        # for a in self.agents:
            # a.step.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.step = fn(a.step)
            self.pol_dev = device


    @classmethod
    def init_from_env(cls, env, jam_weight=60, turn_weight=20,
                      reward_scale=10.,move_cost=10,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        """
        agent_init_params = []
        sa_size = []
        for acsp, obsp in zip(env.action_space,
                              env.observation_space):
            agent_init_params.append({'num_in_pol': obsp.shape[0],
                                      'num_out_pol': acsp.n})
            sa_size.append((obsp.shape[0], acsp.n))

        init_dict = {'jam_weight': jam_weight, 'turn_weight': turn_weight,
                     'reward_scale': reward_scale,'move_cost':move_cost,
                     'map_height': 10,
                     'map_width': 10,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    