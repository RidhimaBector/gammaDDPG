import numpy as np
import copy
from scipy.special import softmax

from collections import defaultdict

import os
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
    sys.path.append("../") 


"""
Show policy from Q
"""
def Show_PolicyQ(Q_dict, env):
    policy_Q = np.zeros(env.shape)

    for i in Q_dict:
        best_action = np.argmax(Q_dict[i])
        policy_Q[np.unravel_index(i, env.shape)] = best_action

    print(policy_Q)

"""
Get policy-table from Q-dictionary
"""

def Get_Policy(Q_dict, env):
    #Q = copy.deepcopy(Q_dict)
    policy = softmax(Q_dict, axis=1) #np.zeros((Q.shape[0], env.nA))

    return policy

"""def Get_Policy(Q_dict, env):
    Q = copy.deepcopy(Q_dict)
    policy = np.zeros((Q.shape[0], env.nA))
    ind = np.arange(Q.shape[0])
    best_action = np.argmax(Q, axis=1) #.cpu().detach().numpy()
    policy[ind,best_action] = 1
    policy = (policy+0.001)/(1+0.001*env.nA)
    return policy

def Get_Policy_Old(Q_dict, env):
    policy = np.zeros((env.nS, env.nA))
    for i in range(16): #Q_dict:
        best_action = np.argmax(Q_dict[i])
        policy[i][best_action] = 1
    for i in range(len(policy)):
        policy[i] = (policy[i]+0.001)/(1+0.001*env.nA)
    return policy"""

"""
Function to convert dictionary-Q to matrix-Q
"""
def DicQ_To_MatrixQ(dict_Q, env):
    Q = copy.deepcopy(dict_Q)
    dict_key = np.arange(env.nS)
    Q_matrix = np.array([Q[i] for i in dict_key])

    return Q_matrix


"""
Function to convert matrix to dictionary
"""
def Matrix_to_Dict(input_matrix):
    matrix = input_matrix.copy()
    dictionary = defaultdict(lambda: np.zeros(len(matrix)))
    for i in range(len(matrix)):
        dictionary[i] = matrix[i]

    return dictionary




if __name__ == "__main__":
    
    # Env
    from envs.env3D_4x4 import GridWorld_3D_env
    env = GridWorld_3D_env()
    
    # Victim
    from victim.victim_Q import VictimAgent_Q
    victim_args = {
        "env": env, 
        "discount_factor": 1.0, 
        "alpha": 0.1, 
        "epsilon": 0.1,
    }
    victim = VictimAgent_Q(**victim_args)
    victim.train(5)
    
    ''' Evaluation '''
    Show_PolicyQ(victim.Q)
    
    policy = Get_Policy(victim.Q, victim.env)
    print(policy)
    
    Q_matrix = DicQ_To_MatrixQ(victim.Q, victim.env)
    print(Q_matrix)
    
    Q_dict = Matrix_to_Dict(Q_matrix)
    print(Q_dict)
    
    