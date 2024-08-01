import math
import random
import numpy as np
import copy
from scipy.special import softmax
import time

from collections import namedtuple
from collections import defaultdict
from itertools import count
import itertools

import os
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
    sys.path.append("../") 

from utils import utils_buf, utils_op, utils_attack
from envs.target_def import TARGET

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

#LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE
T_max = config.VICTIM.TMAX


class VictimAgent():
    
    def __init__(self, env, MEMORY_SIZE, discount_factor=1.0, alpha=0.1, epsilon=0.1):
        self.env = env
        self.memory_size = MEMORY_SIZE
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        
        seed = int( time.time() )
        self.np = np
        self.env.seed(seed)
        self.np.random.seed(seed)
        self.Q = self.np.zeros((16,self.env.action_space.n)) #defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.MEM = utils_buf.Memory(MEMORY_SIZE)
        
    def reset(self):
        self.Q = self.np.zeros((16,self.env.action_space.n)) #defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.MEM = utils_buf.Memory(self.memory_size)
        

    """
    Q-learning algorithm with transfer_Q
    """
    def MakeEpsilonGreedyPolicy(self):
        nA = self.env.action_space.n
        def policy_fn(observation):
            A = self.np.ones(nA, dtype = float) * self.epsilon/nA
            best_action = self.np.argmax(self.Q[observation])
            A[best_action] += (1.0 - self.epsilon)
            return A
        return policy_fn

    def Train_Model(self, num_episodes):

        # The policy we're following
        #policy = self.MakeEpsilonGreedyPolicy()
        Q_matrix = self.Q #utils_op.DicQ_To_MatrixQ(self.Q, self.env)
        victim_transitions = self.np.ones((16,2)) * 4
        victim_transitions[:,0] = self.np.arange(16)

        for i_episode in range(num_episodes):
            # logger
            #trajectory_list = []
            score = 0
            
            # Reset the environment and pick the first action
            state = self.env.reset()

            # One step in the environment
            for t in itertools.count():
                # logger
                t_sample = []

                # Take a step
                action_probs = softmax(Q_matrix[state]) #action_probs = policy(state)
                action = self.np.random.choice(self.np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(action)
                
                victim_transitions[state,1] = action

                # TD Update
                best_next_action = self.np.argmax(self.Q[next_state])    
                td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta
                
                # save transitions
                """if t!=0:
                    t_sample.append(state)
                    t_sample.append(action)
                    trajectory_list.append(t_sample)"""
                    
                # update state
                state = copy.deepcopy(next_state)
                score += reward

                if done:
#                     print(f"Timsteps = {t} | Scores = {score}")
                    break

            # Trajectory to MEMORY with head_padding
            """if len(trajectory_list) < LEN_TRAJECTORY:
                padding_state = 0
                padding_action = 0
                n_padding = LEN_TRAJECTORY - len(trajectory_list)
                for i in range(n_padding):
                    self.MEM.push(padding_state, padding_action)
                for i in range(len(trajectory_list)):
                    state = trajectory_list[i][0]
                    action = trajectory_list[i][1]
                    self.MEM.push(state, action)
            else:
                for i in range(LEN_TRAJECTORY):
                    state = trajectory_list[i][0]
                    action = trajectory_list[i][1]
                    self.MEM.push(state, action)"""
                
#             # display training progress
#             if (i_episode + 1) % 100 == 0:
#                 print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
#                 sys.stdout.flush()

        return victim_transitions #[[0,1,2,3,7,11], :]


    def train_for_eval(self, num_episodes):

        # The policy we're following
        #policy = self.MakeEpsilonGreedyPolicy()
        Q_matrix = self.Q #utils_op.DicQ_To_MatrixQ(self.Q, self.env)
        victim_transitions = self.np.ones((16,2)) * 4
        victim_transitions[:,0] = self.np.arange(16)
        
        #stats_timestep = []
        accuracy_episode = []
        accuracy_softmax_episode = []
        accuracy_softmax_complete_episode = []

        for i_episode in range(num_episodes):
            # logger
            trajectory_list = []
            score = 0

            # Reset env
            state = self.env.reset()

            for t in itertools.count(): #for t in range(10):
                # logger
                t_sample = []

                # Take a step
                action_probs = softmax(Q_matrix[state]) #action_probs = policy(state)
                action = self.np.random.choice(self.np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                victim_transitions[state,1] = action

                # TD Update
                best_next_action = self.np.argmax(self.Q[next_state])    
                td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta
                
                # save transitions
                """if t!=0:
                    t_sample.append(state)
                    t_sample.append(action)
                    trajectory_list.append(t_sample)"""
                
                state = copy.deepcopy(next_state)
                
                if done:     
                    done_attack, accuracy, accuracy_softmax, accuracy_softmax_complete = utils_attack.Attack_Done_Identify(self.env, TARGET, self.Q)
                    accuracy_episode.append(accuracy)
                    accuracy_softmax_episode.append(accuracy_softmax)
                    accuracy_softmax_complete_episode.append(accuracy_softmax_complete)
                    break
            
            # Trajectory to MEMORY with head_padding
            """if len(trajectory_list) < LEN_TRAJECTORY:
                padding_state = 0
                padding_action = 0
                n_padding = LEN_TRAJECTORY - len(trajectory_list)
                for i in range(n_padding):
                    self.MEM.push(padding_state, padding_action)
                for i in range(len(trajectory_list)):
                    state = trajectory_list[i][0]
                    action = trajectory_list[i][1]
                    self.MEM.push(state, action)
            else:
                for i in range(LEN_TRAJECTORY):
                    state = trajectory_list[i][0]
                    action = trajectory_list[i][1]
                    self.MEM.push(state, action)"""
            
        return accuracy_episode, accuracy_softmax_episode, accuracy_softmax_complete_episode, victim_transitions #[[0,1,2,3,7,11], :]
    
                
                
if __name__ == "__main__":
    from envs.env3D_4x4 import GridWorld_3D_env
    env = GridWorld_3D_env()
    
    victim_args = {
        "env": env, 
        "MEMORY_SIZE": 100, 
        "discount_factor": 1.0, 
        "alpha": 0.1, 
        "epsilon": 0.1,
    }
    
    victim = VictimAgent(**victim_args)
    victim.Train_Model(10)
    victim.Show_PolicyQ()
    victim.Eval_Model()
    
    print(f"Size of Memory = {victim.MEM.__len__()}")
    print(f"First 5 Trajectory is \n{victim.MEM.memory[0:5]}")
    
    