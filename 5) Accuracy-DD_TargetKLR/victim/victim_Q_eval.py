import math
import random
import numpy as np
import copy
from scipy.special import softmax

from collections import namedtuple
from collections import defaultdict
from itertools import count
import itertools

import os
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
    sys.path.append("../") 

from utils import utils_buf

import utils.utils_attack as utils
import utils.utils_op as utils_op
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


class VictimAgent_Q():
    
    def __init__(self, env, MEMORY_SIZE, discount_factor=1.0, alpha=0.1, epsilon=0.1):
        self.env = env
        self.init_T = env.T.copy()
        
        self.memory_size = MEMORY_SIZE
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.Q = np.zeros((16,self.env.action_space.n)) #defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.MEM = utils_buf.Memory(MEMORY_SIZE)
        
    def reset(self):
        self.Q = np.zeros((16,self.env.action_space.n)) #defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.MEM = utils_buf.Memory(self.memory_size)
        

    """
    Q-learning algorithm with transfer_Q
    """
    def MakeEpsilonGreedyPolicy(self):
        nA = self.env.action_space.n
        def policy_fn(observation):
            A = np.ones(nA, dtype = float) * self.epsilon/nA
            best_action = np.argmax(self.Q[observation])
            A[best_action] += (1.0 - self.epsilon)
            return A
        return policy_fn
                
    def train_for_eval(self, num_episodes):

        # The policy we're following
        #policy = self.MakeEpsilonGreedyPolicy()
        Q_matrix = self.Q #utils_op.DicQ_To_MatrixQ(self.Q, self.env)
        victim_transitions = np.ones((16,2)) * 4
        victim_transitions[:,0] = np.arange(16)
        
        stats_timestep = []
        stats_episode = []

        for i_episode in range(num_episodes):
            # logger
            trajectory_list = []
            score = 0

            # Reset env
            state = self.env.reset()

            for t in range(10): #in itertools.count():
                # logger
                t_sample = []

                # Take a step
                action_probs = softmax(Q_matrix[state]) #action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                victim_transitions[state,1] = action

                # TD Update
                best_next_action = np.argmax(self.Q[next_state])    
                td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta
                
                # save transitions
                """if t!=0:
                    t_sample.append(state)
                    t_sample.append(action)
                    trajectory_list.append(t_sample)"""
                    
                done, rate_timestep = utils.Attack_Done_Identify(self.env, TARGET, self.Q)
                stats_timestep.append(rate_timestep)
                    
                if done:
                    break

                state = copy.deepcopy(next_state)
        
            # evaluate attack_performance
            done, rate_episode = utils.Attack_Done_Identify(self.env, TARGET, self.Q)
            stats_episode.append(rate_episode)
            
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
            
        return stats_episode, stats_timestep, victim_transitions #[[0,1,2,3,7,11], :]


    """
    Function to generate trajectories
    """
    """def Eval_Model(self, n_episode=1):
        # logger
        score = 0

        policy = self.Get_Policy()

        for i_episode in range(n_episode):

            # Reset the environment and pick the first action
            state = self.env.reset()

            # One step in the environment
            for t in range(T_max):

                # Take a step
                action = policy[state].argmax()
                next_state, reward, done, _ = self.env.step(action)
                score += reward

                if done or t+1==T_max:
                    print(f"Q-Victim: Timsteps = {t} | Scores = {score}")
                    break

                # update state
                state = copy.deepcopy(next_state)"""
    
                
                
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
    
    victim = VictimAgent_Q(**victim_args)
    victim.train(10)
    victim.eval_model()
    
    print(f"Size of Memory = {victim.MEM.__len__()}")
    print(f"First 5 Trajectory is \n{victim.MEM.memory[0:5]}")
    
    victim.train_for_eval(10)
    
    