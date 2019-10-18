from __future__ import print_function
from tkinter import _flatten
import sys
import numpy as np
import copy
import operator
import random
import heapq
from gym import spaces
import csv

MAP = \
    '''
..........
.        .
.        .
.        .
.        .
.        .
.        .
.        .
.        .
..........
'''

MAP = MAP.strip().split('\n')
MAP = [[c for c in line] for line in MAP]


DX = [-1, 1, 0, 0, 0]
DY = [0, 0, -1, 1, 0]

# position_list = [[1, 1], [3, 7], [2, 3], [3, 1], [3, 2], [4,4]]
# target_list = [[2, 6], [1, 2], [1, 7], [1, 6], [1, 5],[8,8]]

# added
# load samples
def load_sample():
    samples = []
    with open('task.csv','r') as file:
        reader = csv.reader(file)
        for line in reader:
            sample = []
            if line[0] == 'position_list':
                continue
            
            for i in range(6):
                sample.append([int(line[0][2+i*8]), int(line[0][5+i*8])])
                
            for i in range(6):
                sample.append([int(line[1][2+i*8]), int(line[1][5+i*8])])
                
            samples.append(sample)
        
    return samples

# added
# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        self.u_onehot = None
        self.pre_u_onehot = None
        self.action_choice = 4

class Robot(object):
    def __init__(self, id_=None, current_x=None, current_y=None, target_x=None, target_y=None):
        self.id = id_
        self.current_position = np.array([current_x, current_y], dtype=np.int8)
        self.next_position = self.current_position
        self.target = np.array([target_x, target_y], dtype=np.int8)
        self.is_ready = True
        self.is_end = False
        
        self.path = []
        self.counter = 0
        
        self.state = None
        # modified
        self.action = Action()
        
        self.reward = None
        

class Env(object):
    def __init__(self, map_l, map_w, robot_num):
        # for test
        # self.sample_index = -1
        # self.samples = load_sample()
        # self.seed_int = 0
        
        self.map = copy.deepcopy(MAP)
        self.robot_num = robot_num
        self.robot_list = []
        self.is_end = False
        
        # added
        self.obs = []
        self.rewards = []
        self.agent_actions = []
        self.map_l = map_l
        self.map_w = map_w
        self.dim_p = 2
        
        self.create_map()
        
        self.action_space = []
        self.observation_space = []
        for agent in self.robot_list:
            # physical action space
            u_action_space = spaces.Discrete(self.dim_p * 2 + 1)
            self.action_space.append(u_action_space)
            # observation space
            obs_dim = len(self.get_obs(agent))
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,)))

    def create_map(self, sample=None):
        """ create a map with length 'map_l' and width 'map_w' excluding the edge"""
        map_new = [list(map(lambda j: ' ' if 1 <= j <= self.map_l else '.', list(range(self.map_l+2)))) for _ in range(self.map_w)]
        map_edge = list('.' for _ in range(self.map_l+2))
        map_new.append(map_edge)
        map_new.insert(0, map_edge)
        self.map = map_new
        
        self.robot_list = []
        map_env = [(x,y) for x in range(1,self.map_l) for y in range(1,self.map_w)]
        map_env = np.array(map_env)
        
        if not sample:
            sample = map_env[np.random.choice(
                    len(map_env), self.robot_num*2, replace=False)]
        # for test
        # print(sample[:self.robot_num])
        # print(sample[self.robot_num:])
        for i in range(self.robot_num):
            self.robot_list.append(Robot(i, sample[i][0], sample[i][1],
                                         sample[i+self.robot_num][0], sample[i+self.robot_num][1]))
        
        self.is_end = False
        
        # added
        for robot in self.robot_list:
            robot.action.pre_u_onehot = np.zeros(self.dim_p * 2 + 1)
            robot.action.u_onehot = np.zeros(self.dim_p * 2 + 1)
        

    # 之前在到达目的地前一步会直接跳一步到目的地，相当于少了一步，后来修改为如下逻辑 
    def interact(self, agent, action):
        reward = 0
        assert not agent.is_end
        if all(operator.eq(agent.current_position, agent.next_position)):
            if all(operator.eq(agent.current_position, agent.target)):
                agent.is_end = True  # a is end
                self.is_end = all([robot.is_end for robot in self.robot_list])
                reward = 1000
                return reward
            new_x = agent.current_position[0] + DX[action]
            new_y = agent.current_position[1] + DY[action]
            new_pos_char = self.map[new_x][new_y]
            collision = [((operator.eq([new_x, new_y], robot.current_position)).all() or
                          (operator.eq([new_x, new_y], robot.next_position)).all())
                         for robot in self.robot_list if robot != agent]
            # 碰撞
            if new_pos_char == '.' or any(collision):
                reward = -10  # penalty and do not change position
            # 其他
            else:
                agent.next_position = np.array([new_x, new_y])
                if any(operator.ne(agent.current_position, agent.next_position)):
                    agent.is_ready = False
                reward = -1
        return reward
    
    # interact for a_star
    def interact_astar(self, agent):
        reward = 0
        # action为onehot数组，每个位置为1时分别表示：不动 南 北 东 西
        action = np.zeros(5)
        assert not agent.is_end
        if all(operator.eq(agent.current_position, agent.next_position)):
            if all(operator.eq(agent.current_position, agent.target)):
                agent.is_end = True  # a is end
                self.is_end = all([robot.is_end for robot in self.robot_list])
                reward = 1000
                action[0] = 1
                return reward, action
            new_x = agent.path[1][0]
            new_y = agent.path[1][1]
            # calculate action
            # 正南负北
            delta_x = new_x - agent.current_position[0]
            # 正东负西
            delta_y = new_y - agent.current_position[1]
            new_one_hot = [abs(delta_x+delta_y), delta_x, delta_x, delta_y, delta_y]
            D = [0, 1, -1, 1, -1]
            for i, item in enumerate(new_one_hot):
                # print(new_one_hot, D)
                if item == D[i]:
                    action[i] = 1
                    break
                    
                
            new_pos_char = self.map[new_x][new_y]
            collision = [((operator.eq([new_x, new_y], robot.current_position)).all() or
                          (operator.eq([new_x, new_y], robot.next_position)).all())
                         for robot in self.robot_list if robot != agent]
            # 碰撞
            if new_pos_char == '.' or any(collision):
                reward = -10  # penalty and do not change position
                if np.random.random() < 0.5:
                    path, found = self.a_star(agent, block=True)
                    if found:
                        agent.counter = 0
                        agent.path = path
                else:
                    action[0] = 1
            # 其他
            else:
                agent.next_position = np.array([new_x, new_y])
                agent.path.pop(0)
                if any(operator.ne(agent.current_position, agent.next_position)):
                    agent.is_ready = False
                reward = -1
        return reward, action

    def present_state(self, agent):
        # for test
        # self.seed(self.seed_int)
        # self.seed_int += 1
        
        if (not agent.is_ready) and np.random.random() < 0.5:
            agent.is_ready = True
            agent.current_position = agent.next_position
        state = [[-1 if c == '.' else 0 for c in line] for line in self.map]
        state[agent.current_position[0]][agent.current_position[1]] = 1
        state[agent.target[0]][agent.target[1]] = 2
        for robot in self.robot_list:
            if robot != agent:
                state[robot.current_position[0]][robot.current_position[1]] = -2
        state = np.array(list(_flatten(state)))
        return agent.is_ready, state

    def find_path(self):
        for robot in self.robot_list:
            path, found = self.a_star(robot)
            if found:
                robot.path = path
            else:
                print("The agent can not found path")

    def a_star(self, agent, block=False):
        start_point = agent.current_position
        end_point = agent.target

        jam_weight = 60
        turn_weight = 20

        move_cost = 10
        found = False
        map_size = list(np.shape(self.map))
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

        if jam_weight != 0:
            for robot in self.robot_list:
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
            if self.map[x][y] == '.':
                avail_direction = []
            elif block and x == start_point[0] and y == start_point[1]:
                avail_direction = [0, 1, 2, 3]
                for robot in self.robot_list:
                    if np.sum(np.abs(robot.current_position - agent.current_position)) == 1:
                        #  np.linalg.norm(robot.current_position - agent.current_position,ord=1) == 1:
                        if robot.current_position[0] - agent.current_position[0] == 1:
                            if 3 in avail_direction:
                                avail_direction.remove(3)
                        elif robot.current_position[0] - agent.current_position[0] == -1:
                            if 1 in avail_direction:
                                avail_direction.remove(1)
                        elif robot.current_position[1] - agent.current_position[1] == 1:
                            if 0 in avail_direction:
                                avail_direction.remove(0)
                        elif robot.current_position[1] - agent.current_position[1] == -1:
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

    def print_map_with_reprint(self, output_list):
        printed_map = copy.deepcopy(self.map)
        for robot in self.robot_list:
            printed_map[robot.target[0]][robot.target[1]] = chr(ord('A') + robot.id)
        for robot in self.robot_list:
            printed_map[robot.next_position[0]][robot.next_position[1]] = chr(ord('1') + robot.id)
        for robot in self.robot_list:
            printed_map[robot.current_position[0]][robot.current_position[1]] = chr(ord('a') + robot.id)
        printed_list = [''.join([c for c in line]) for line in printed_map]
        for i, line in enumerate(printed_list):
            output_list[i] = line 

    def print_map(self):
        printed_map = copy.deepcopy(self.map)
        for robot in self.robot_list:
            printed_map[robot.target[0]][robot.target[1]] = chr(ord('A') + robot.id)
        for robot in self.robot_list:
            printed_map[robot.next_position[0]][robot.next_position[1]] = chr(ord('1') + robot.id)
        for robot in self.robot_list:
            printed_map[robot.current_position[0]][robot.current_position[1]] = chr(ord('a') + robot.id)
        print('\n'.join([''.join([c for c in line]) for line in printed_map]))
        
    def write_map2file(self, in_s, is_print=False):
        printed_map = copy.deepcopy(self.map)
        for robot in self.robot_list:
            printed_map[robot.target[0]][robot.target[1]] = chr(ord('A') + robot.id)
        for robot in self.robot_list:
            printed_map[robot.next_position[0]][robot.next_position[1]] = chr(ord('1') + robot.id)
        for robot in self.robot_list:
            printed_map[robot.current_position[0]][robot.current_position[1]] = chr(ord('a') + robot.id)
        
        s = 'not change\n'
        if is_print:
            s = '\n'.join([''.join([c for c in line]) for line in printed_map]) + '\n'
        return in_s + s
    
    # added
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(0)
        else:
            np.random.seed(seed)
    
    def reset(self):
        # self.sample_index += 1
        # self.create_map(self.samples[self.sample_index])
        
        self.create_map()
        obs_n = []
        for agent in self.robot_list:
            obs_n.append(self.get_obs(agent))
        return obs_n
    
    def step(self, action_n):
        # print(action_n)
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'is_ready': []}
        # set action for each agent
        for i, agent in enumerate(self.robot_list):
            self.set_action(agent, action_n[i])
                
        # record observation for each agent
        for agent in self.robot_list:
            # cannot change the calling order of 
            # functions get_reward and _get_obs
            reward_n.append(self.get_reward(agent))
        
        for agent in self.robot_list:
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))
            info_n['is_ready'].append(self._get_info(agent))
        return obs_n, reward_n, done_n, info_n    
    
    def set_action(self, agent, action):
        agent.action.u = np.zeros(self.dim_p, dtype=np.int)
        
        # action为onehot数组，每个位置为1时分别表示：不动 南 北 东 西
        agent.action.u[0] += int(action[1] - action[2])
        agent.action.u[1] += int(action[3] - action[4])
        # print('agent {} action_u: {}'.format(agent.id, agent.action.u))
        agent.action.pre_u_onehot = agent.action.u_onehot
        agent.action.u_onehot = action
        # 默认为不动
        agent.action.action_choice = 4
        
        # 不动
        if agent.action.u_onehot[0] == 1:
            agent.action.action_choice = 4
        # 往南走一格
        elif agent.action.u_onehot[1] == 1:
            agent.action.action_choice = 1
        # 往北走一格
        elif agent.action.u_onehot[2] == 1:
            agent.action.action_choice = 0
        # 往东走一格
        elif agent.action.u_onehot[3] == 1:
            agent.action.action_choice = 3
        # 往西走一格
        elif agent.action.u_onehot[4] == 1:
            agent.action.action_choice = 2
            
        # print('agent {} action_choice: {}'.format(agent.id, agent.action.action_choice))
    
    def get_obs(self, agent):
        entity_pos = []
        for robot in self.robot_list:
            entity_pos.append(robot.target - agent.current_position)
        other_pos = []
        pre_actions = []
        for other in self.robot_list:
            pre_actions.append(other.action.pre_u_onehot)
            if other is agent: continue
            other_pos.append(other.current_position - agent.current_position)
        agent_pos = [robot.current_position for robot in self.robot_list]
        target_pos = [robot.target for robot in self.robot_list]
        obs = np.concatenate([agent.current_position] + \
                               entity_pos + other_pos + agent_pos + target_pos + \
                                [np.concatenate(pre_actions)])
        return np.concatenate([obs, 
                               np.zeros_like(obs)+int(agent.is_ready)])  
    
    def _get_obs(self, agent):
        self.present_state(agent)
        entity_pos = []
        for robot in self.robot_list:
            entity_pos.append(robot.target - agent.current_position)
        other_pos = []
        pre_actions = []
        for other in self.robot_list:
            pre_actions.append(other.action.pre_u_onehot)
            if other is agent: continue
            other_pos.append(other.current_position - agent.current_position)
        obs = np.concatenate([agent.current_position] + \
                               entity_pos + other_pos + \
                               [np.concatenate(pre_actions)])
        return np.concatenate([obs, 
                               np.zeros_like(obs)+int(agent.is_ready)])    
    
    def get_reward(self, robot):
        # 当e.is_end=False时，其他已到达目的地的agent的reward设为0
        if robot.is_end: return 0
        # 当agent移动中，鼓励其保持上一个动作（trick）
        if not robot.is_ready:
            if np.all(robot.action.u_onehot == robot.action.pre_u_onehot): 
                # print('agent {} kept going'.format(robot.id))
                return -1
            else:
                # 还原为原本的动作
                robot.action.u_onehot = robot.action.pre_u_onehot
                # print('agent {} changed action'.format(robot.id))
                return -10
        # 当agent.is_ready=True,正常调用interact，robot.action.action_choice为符合interact传入形式的值
        else:
            return self.interact(robot, robot.action.action_choice)
    
    def _get_info(self, agent):
        return agent.is_ready

    def _get_done(self, agent):
        return agent.is_end