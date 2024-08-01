import numpy as np
import torch
import gym
import argparse
import os
import sys
import time
import copy
import ot

from collections import defaultdict
from itertools import count

# TensorBoard
import tensorboardX
import datetime

# Configuration
"""from yacs.config import CfgNode as CN
yaml_name='config/config_default.yaml'
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()"""

#SEQ_LEN = 6 #config.AE.SEQ_LEN
EMBEDDING_SIZE = 5 #config.AE.EMBEDDING_SIZE
MEMORY_SIZE = 50 #config.AE.MEMORY_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attack package
from utils import utils_buf, utils_op, utils_attack, utils_log
from attack.DDPG import DDPG
from envs.target_def import TARGET

# Environment object
from envs.env3D_4x4 import GridWorld_3D_env
env = GridWorld_3D_env()
INIT_T = env.T.copy()

# Victim object
from victim.victim_Q import VictimAgent
#from victim.victim_Sarsa_eval import VictimAgent_Sarsa
#from victim.victim_MC_eval import VictimAgent_MC

# AutoEncoder
from ae.ae import AutoEncoder

#Cost Matrix
grid = np.array([[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]])
cost_matrix = ot.dist(grid, grid, metric='cityblock') * 20
np.fill_diagonal(cost_matrix, 10)
cost_matrix = np.repeat(cost_matrix, env.nA, axis=1)
cost_matrix = np.repeat(cost_matrix, env.nA, axis=0)
np.fill_diagonal(cost_matrix, 0)

''' ... PATH ... '''
policy_no = str(sys.argv[1])
PATH = "storage/R_0818/good_model_" + policy_no
PATH_ae = "storage/R_0818/" + "340240" + "_f-o_AutoEncoder_SftMx" #"14800" + "_f-o_AutoEncoder"

# Set seeds
seed = 0
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

""" parameter of DDPG """

discount=0.95                # Discount factor
tau=0.005                    # Target network update rate

''' ..... Attack Network ..... '''
# Input / Output size
state_dim = EMBEDDING_SIZE + env.nS
action_dim = env.Attack_ActionSpace.shape[0]
max_action = float(env.Attack_ActionSpace.high[0])

attack_args = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": discount,
    "tau": tau,
}

kwargsNew = {
    "seed": 0,
    "nb_states": state_dim,
    "nb_actions": action_dim,
    "max_action": max_action,
    "hidden1": 400,
    "hidden2": 300,
    "init_w": 0.003,
    "prate": 0.0001,
    "rate": 0.001,
    "ou_theta": 0.15,
    "ou_mu": 0.0,
    "ou_sigma": 0.2,
    "bsize": 256,
    "tau": tau,
    "discount": discount,
    "epsilon_divisor": 1000, #Number of episodes over which to decrease actual epsilon
    "is_training": True
}

""" load Policy """ 
Policy = DDPG(**kwargsNew)
Policy.load(PATH)

''' ..... Victim ..... '''

victim_args = {
    "env": env, 
    "MEMORY_SIZE": MEMORY_SIZE,
    "discount_factor": 0.9, 
    "alpha": 0.1, 
    "epsilon": 0.1,
}

victim = VictimAgent(**victim_args)

""" parameter of AutoEncoder """

ae_enc_in_size = 32 #SEQ_LEN*2
ae_enc_out_size = 5 #EMBEDDING_SIZE
ae_dec_in_size = 6 #1+EMBEDDING_SIZE
ae_dec_out_size = 5 #4 #action_dim

ae_args = {
    "enc_in_size": ae_enc_in_size, 
    "enc_out_size": ae_enc_out_size, 
    "dec_in_size": ae_dec_in_size, 
    "dec_out_size": ae_dec_out_size, 
    "lr": 0.001, 
}

ae = AutoEncoder(**ae_args)
ae.load(PATH_ae)

""" Func: evaluation """
Tmax = 15 #50 #100
Plot_Seed = "Same"
Model = "Accuracy-WD"
n_victim_pop = 5 #5 #10
def eval_policy(policy):
    # log
    distance_K_atk_timestep = [] #complete data
    distance_grid_K_atk_timestep = [] #complete data
    distance_behavior_K_atk_timestep = [] #complete data
    distance_W_atk_timestep = [] #complete data
    distance_grid_W_atk_timestep = [] #complete data
    distance_behavior_W_atk_timestep = [] #complete data
    accuracy_victim_timestep = [] #complete data
    accuracy_atk_timestep = [] #complete data
    accuracy_sftmx_victim_timestep = [] #complete data
    accuracy_sftmx_atk_timestep = [] #complete data
    accuracy_sftmx_complete_victim_timestep = [] #complete data
    accuracy_sftmx_complete_atk_timestep = [] #complete data
    effort_atk_timestep = [] #complete data
    time_atk_timestep = [] #complete data

    # reset victim's env and Q
    env.reset_altitude()
    victim.reset()

    # Initialize the attacker's state
    victim_info = np.zeros((1,EMBEDDING_SIZE))
    victim_tensor = torch.from_numpy(victim_info)
    victim_tensor_4d = victim_tensor.unsqueeze(0).unsqueeze(0)

    env_info = env.altitude.copy()
    env_tensor = torch.from_numpy(env_info)
    env_tensor = env_tensor.view(1, env.nS)
    env_tensor_4d = env_tensor.unsqueeze(0).unsqueeze(0)

    state = torch.cat((victim_tensor_4d, env_tensor_4d), 3)
    
    orgA = env.altitude.copy().reshape((16, 1)) #New
    curA =  orgA.copy()

    for t in range(Tmax):
        tic_timestep = time.time()

        # select action
        action = Policy.select_on_policy_action(np.array(state)) #Policy.select_action(np.array(state))

        ## perform attack_action on Env
        env.Attack_Env(action)
        
        # next_victim_info: compute victim's updated policy
        accuracy_array, accuracy_sftmx_array, accuracy_sftmx_complete_array, victim_transitions = victim.train_for_eval(80)
        ### ... updated victim.Q
        next_victim_info = ae.Policy_Embedding(victim_transitions)
        next_victim_tensor = torch.from_numpy(next_victim_info[-1]).unsqueeze(0)
        next_victim_tensor_4d = next_victim_tensor.unsqueeze(0).unsqueeze(0)
        ### ... updated env altitude
        next_env_info = env.altitude.copy()
        next_env_tensor = torch.from_numpy(next_env_info)
        next_env_tensor = next_env_tensor.view(1, env.nS)
        next_env_tensor_4d = next_env_tensor.unsqueeze(0).unsqueeze(0)
        ### ... next_state
        next_state = torch.cat((next_victim_tensor_4d, next_env_tensor_4d), 3)
        
        # Step: cost
        distance_K = utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim.Q, TARGET, cost_matrix, distance_type=0) #system.victim.Q
        distance_grid_K = utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim.Q, TARGET, cost_matrix, distance_type=1)
        distance_behavior_K = utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim.Q, TARGET, cost_matrix, distance_type=2)
        distance_W = utils_attack.Attack_Cost_Compute_W(env, INIT_T, victim.Q, TARGET, cost_matrix, distance_type=0) #system.victim.Q
        distance_grid_W = utils_attack.Attack_Cost_Compute_W(env, INIT_T, victim.Q, TARGET, cost_matrix, distance_type=1)
        distance_behavior_W = utils_attack.Attack_Cost_Compute_W(env, INIT_T, victim.Q, TARGET, cost_matrix, distance_type=2)
        done, _, _, _ = utils_attack.Attack_Done_Identify(env, TARGET, victim.Q) #system.victim.Q)
        effort, curA = utils_attack.Attack_Effort(curA, env)
        #effort = - effort
        toc_timestep = time.time()
        time_timestep = toc_timestep - tic_timestep #tic_timestep - toc_timestep #- (toc_timestep - tic_timestep)
        
        # Step: log
        distance_K_atk_timestep.append(distance_K)
        distance_grid_K_atk_timestep.append(distance_grid_K)
        distance_behavior_K_atk_timestep.append(distance_behavior_K)
        distance_W_atk_timestep.append(distance_W)
        distance_grid_W_atk_timestep.append(distance_grid_W)
        distance_behavior_W_atk_timestep.append(distance_behavior_W)
        accuracy_victim_timestep += accuracy_array
        accuracy_atk_timestep.append(accuracy_array[-1])
        accuracy_sftmx_victim_timestep += accuracy_sftmx_array
        accuracy_sftmx_atk_timestep.append(accuracy_sftmx_array[-1])
        accuracy_sftmx_complete_victim_timestep += accuracy_sftmx_complete_array
        accuracy_sftmx_complete_atk_timestep.append(accuracy_sftmx_complete_array[-1])
        effort_atk_timestep.append(effort)
        time_atk_timestep.append(time_timestep)

        # update state 
        state = copy.deepcopy(next_state)

        if done:
            
            for _ in range(t+1,Tmax):
                
                # Step: log
                distance_K_atk_timestep.append(distance_K)
                distance_grid_K_atk_timestep.append(distance_grid_K)
                distance_behavior_K_atk_timestep.append(distance_behavior_K)
                distance_W_atk_timestep.append(distance_W)
                distance_grid_W_atk_timestep.append(distance_grid_W)
                distance_behavior_W_atk_timestep.append(distance_behavior_W)
                accuracy_victim_timestep += [accuracy_array[-1]]*len(accuracy_array)
                accuracy_atk_timestep.append(accuracy_array[-1])
                accuracy_sftmx_victim_timestep += [accuracy_sftmx_array[-1]]*len(accuracy_sftmx_array)
                accuracy_sftmx_atk_timestep.append(accuracy_sftmx_array[-1])
                accuracy_sftmx_complete_victim_timestep += [accuracy_sftmx_complete_array[-1]]*len(accuracy_sftmx_complete_array)
                accuracy_sftmx_complete_atk_timestep.append(accuracy_sftmx_complete_array[-1])
                effort_atk_timestep.append(0)
                time_atk_timestep.append(0)
            
            break

    #print("---------------------------------------")
    #cum_reward = cumulative_reward #/(t+1)
    #print(f"Evaluation over {t} timesteps: {cumulative_reward:.3f}")
    #utils_op.Show_PolicyQ(victim.Q, env)
    #print("---------------------------------------")


    return distance_K_atk_timestep, distance_grid_K_atk_timestep, distance_behavior_K_atk_timestep, distance_W_atk_timestep, distance_grid_W_atk_timestep, distance_behavior_W_atk_timestep, accuracy_victim_timestep, accuracy_atk_timestep, accuracy_sftmx_victim_timestep, accuracy_sftmx_atk_timestep, accuracy_sftmx_complete_victim_timestep, accuracy_sftmx_complete_atk_timestep, effort_atk_timestep, time_atk_timestep



""" Evaluate """
distance_K_atk_timestep_cluster = [] #complete data
distance_grid_K_atk_timestep_cluster = [] #complete data
distance_behavior_K_atk_timestep_cluster = [] #complete data
distance_W_atk_timestep_cluster = [] #complete data
distance_grid_W_atk_timestep_cluster = [] #complete data
distance_behavior_W_atk_timestep_cluster = [] #complete data
accuracy_victim_timestep_cluster = [] #complete data
accuracy_atk_timestep_cluster = [] #complete data
accuracy_sftmx_victim_timestep_cluster = [] #complete data
accuracy_sftmx_atk_timestep_cluster = [] #complete data
accuracy_sftmx_complete_victim_timestep_cluster = [] #complete data
accuracy_sftmx_complete_atk_timestep_cluster = [] #complete data
effort_atk_timestep_cluster = [] #complete data
time_atk_timestep_cluster = [] #complete data


for i_victim_pop in range(n_victim_pop):
    distance_K_atk_timestep, distance_grid_K_atk_timestep, distance_behavior_K_atk_timestep, distance_W_atk_timestep, distance_grid_W_atk_timestep, distance_behavior_W_atk_timestep, accuracy_victim_timestep, accuracy_atk_timestep, accuracy_sftmx_victim_timestep, accuracy_sftmx_atk_timestep, accuracy_sftmx_complete_victim_timestep, accuracy_sftmx_complete_atk_timestep, effort_atk_timestep, time_atk_timestep = eval_policy(Policy)

    distance_K_atk_timestep_cluster.append(distance_K_atk_timestep)
    distance_grid_K_atk_timestep_cluster.append(distance_grid_K_atk_timestep)
    distance_behavior_K_atk_timestep_cluster.append(distance_behavior_K_atk_timestep)
    distance_W_atk_timestep_cluster.append(distance_W_atk_timestep)
    distance_grid_W_atk_timestep_cluster.append(distance_grid_W_atk_timestep)
    distance_behavior_W_atk_timestep_cluster.append(distance_behavior_W_atk_timestep)
    accuracy_victim_timestep_cluster.append(accuracy_victim_timestep)
    accuracy_atk_timestep_cluster.append(accuracy_atk_timestep)
    accuracy_sftmx_victim_timestep_cluster.append(accuracy_sftmx_victim_timestep)
    accuracy_sftmx_atk_timestep_cluster.append(accuracy_sftmx_atk_timestep)
    accuracy_sftmx_complete_victim_timestep_cluster.append(accuracy_sftmx_complete_victim_timestep)
    accuracy_sftmx_complete_atk_timestep_cluster.append(accuracy_sftmx_complete_atk_timestep)
    effort_atk_timestep_cluster.append(effort_atk_timestep)
    time_atk_timestep_cluster.append(time_atk_timestep)

    
""" unify length """
max_size = Tmax * 80 #4800 #2500 #5000 #Tmax * 80

"""for i in range(n_victim):
    last_value = Rate_List[i][-1]
    print(last_value)
    if len(Rate_List[i]) < max_size:
        delta_size = max_size - len(Rate_List[i])
        for j in range(delta_size):
            Rate_List[i].append(last_value)"""
            
            
""" average """
avg_step = 5 #25
accuracy_victim_timestep_avg = []
accuracy_sftmx_victim_timestep_avg = []
accuracy_sftmx_complete_victim_timestep_avg = []

N = max_size
N_avg = N//avg_step

for i_victim_pop in range(n_victim_pop):
    accuracy_victim_timestep_avg_temp = [0]
    accuracy_sftmx_victim_timestep_avg_temp = [0]
    accuracy_sftmx_complete_victim_timestep_avg_temp = [0]
    
    for i in range(0, N_avg):
        start = i*avg_step
        end = (i+1)*avg_step

        tmp_acc = sum(accuracy_victim_timestep_cluster[i_victim_pop][start: end])
        tmp_acc_sftmx = sum(accuracy_sftmx_victim_timestep_cluster[i_victim_pop][start: end])
        tmp_acc_sftmx_c = sum(accuracy_sftmx_complete_victim_timestep_cluster[i_victim_pop][start: end])
        
        accuracy_victim_timestep_avg_temp.append(tmp_acc/avg_step)
        accuracy_sftmx_victim_timestep_avg_temp.append(tmp_acc_sftmx/avg_step)
        accuracy_sftmx_complete_victim_timestep_avg_temp.append(tmp_acc_sftmx_c/avg_step)
        
    accuracy_victim_timestep_avg.append(accuracy_victim_timestep_avg_temp)
    accuracy_sftmx_victim_timestep_avg.append(accuracy_sftmx_victim_timestep_avg_temp)
    accuracy_sftmx_complete_victim_timestep_avg.append(accuracy_sftmx_complete_victim_timestep_avg_temp)
    #print(len(per_avg_rate))
    #print(np.round(per_avg_rate,2))
            
            
""" dataframe """
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
sns.set(font_scale = 1.8)
sns.set_style("whitegrid")

time_compression = np.arange( len(accuracy_victim_timestep_avg[0]) ) #np.array(range(0, len(accuracy_victim_timestep_avg[0]))) #list(range(0, len(avg_rate[0])))
vic_time = (time_compression * avg_step) / 80 #list(np.array(time_compression) * avg_step)
atk_time = np.arange(1,Tmax+1)
vic_Pop_name = ["VP1", "VP2", "VP3", "VP4", "VP5", "VP6", "VP7", "VP8", "VP9", "VP10", "VP11", "VP12", "VP13", "VP14", "VP15", "VP16", "VP17", "VP18", "VP19", "VP20"]

df_acc_V = pd.DataFrame({"Victim Timestep":vic_time, "Model":Model, "Accuracy V":accuracy_victim_timestep_avg[0], "Sftmx Accuracy V":accuracy_sftmx_victim_timestep_avg[0], "C Sftmc Accuracy V":accuracy_sftmx_complete_victim_timestep_avg[0], "Victim Pop":vic_Pop_name[0], "Seed":Plot_Seed })
df_acc_A = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Accuracy A":accuracy_atk_timestep_cluster[0], "Sftmx Accuracy A":accuracy_sftmx_atk_timestep_cluster[0], "C Sftmx Accuracy A":accuracy_sftmx_complete_atk_timestep_cluster[0], "Victim Pop":vic_Pop_name[0], "Seed":Plot_Seed })
df_dis_K = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Distance":distance_K_atk_timestep_cluster[0], "Grid Distance":distance_grid_K_atk_timestep_cluster[0], "Behavior Distance":distance_behavior_K_atk_timestep_cluster[0], "Victim":vic_Pop_name[0], "Seed":Plot_Seed })
df_dis_W = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Distance":distance_W_atk_timestep_cluster[0], "Grid Distance":distance_grid_W_atk_timestep_cluster[0], "Behavior Distance":distance_behavior_W_atk_timestep_cluster[0], "Victim":vic_Pop_name[0], "Seed":Plot_Seed })
df_eff = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Effort":effort_atk_timestep_cluster[0], "Attack Timestep Time":time_atk_timestep_cluster[0], "Victim":vic_Pop_name[0], "Seed":Plot_Seed })
for i in range(1, n_victim_pop):
    tmp_df_acc_V = pd.DataFrame({"Victim Timestep":vic_time, "Model":Model, "Accuracy V":accuracy_victim_timestep_avg[i], "Sftmx Accuracy V":accuracy_sftmx_victim_timestep_avg[i], "C Sftmc Accuracy V":accuracy_sftmx_complete_victim_timestep_avg[i], "Victim Pop":vic_Pop_name[i], "Seed":Plot_Seed })
    tmp_df_acc_A = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Accuracy A":accuracy_atk_timestep_cluster[i], "Sftmx Accuracy A":accuracy_sftmx_atk_timestep_cluster[i], "C Sftmx Accuracy A":accuracy_sftmx_complete_atk_timestep_cluster[i], "Victim Pop":vic_Pop_name[i], "Seed":Plot_Seed })
    tmp_df_dis_K = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Distance":distance_K_atk_timestep_cluster[i], "Grid Distance":distance_grid_K_atk_timestep_cluster[i], "Behavior Distance":distance_behavior_K_atk_timestep_cluster[i], "Victim":vic_Pop_name[i], "Seed":Plot_Seed })
    tmp_df_dis_W = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Distance":distance_W_atk_timestep_cluster[i], "Grid Distance":distance_grid_W_atk_timestep_cluster[i], "Behavior Distance":distance_behavior_W_atk_timestep_cluster[i], "Victim":vic_Pop_name[i], "Seed":Plot_Seed })
    tmp_df_eff = pd.DataFrame({"Attacker Timestep":atk_time, "Model":Model, "Effort":effort_atk_timestep_cluster[i], "Attack Timestep Time":time_atk_timestep_cluster[i], "Victim":vic_Pop_name[i], "Seed":Plot_Seed })
    df_acc_V = df_acc_V.append(tmp_df_acc_V, ignore_index=True)
    df_acc_A = df_acc_A.append(tmp_df_acc_A, ignore_index=True)
    df_dis_K = df_dis_K.append(tmp_df_dis_K, ignore_index=True)
    df_dis_W = df_dis_W.append(tmp_df_dis_W, ignore_index=True)
    df_eff = df_eff.append(tmp_df_eff, ignore_index=True)
df_acc_V.to_csv(policy_no + "_accuracy_V_" + Model + "_Seed" + Plot_Seed + ".csv", index=False)
df_acc_A.to_csv(policy_no + "_accuracy_A_" + Model + "_Seed" + Plot_Seed + ".csv", index=False)
df_dis_K.to_csv(policy_no + "_distance_K_" + Model + "_Seed" + Plot_Seed + ".csv", index=False)
df_dis_W.to_csv(policy_no + "_distance_W_" + Model + "_Seed" + Plot_Seed + ".csv", index=False)
df_eff.to_csv(policy_no + "_effort_" + Model + "_Seed" + Plot_Seed + ".csv", index=False)

""" Figure & Data """
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
sns_plot_acc_V = sns.lineplot(x = "Victim Timestep", y = "Accuracy V", hue="Model", data=df_acc_V, ax=axs[0,0])
sns_plot_acc_A = sns.lineplot(x = "Attacker Timestep", y = "Accuracy A", hue="Model", data=df_acc_A, ax=axs[0,1])
sns_plot_acc_sft_V = sns.lineplot(x = "Victim Timestep", y = "Sftmx Accuracy V", hue="Model", data=df_acc_V, ax=axs[1,0])
sns_plot_acc_sft_A = sns.lineplot(x = "Attacker Timestep", y = "Sftmx Accuracy A", hue="Model", data=df_acc_A, ax=axs[1,1])
sns_plot_acc_sft_C_V = sns.lineplot(x = "Victim Timestep", y = "C Sftmc Accuracy V", hue="Model", data=df_acc_V, ax=axs[2,0])
sns_plot_acc_sft_C_A = sns.lineplot(x = "Attacker Timestep", y = "C Sftmx Accuracy A", hue="Model", data=df_acc_A, ax=axs[2,1])
plt.savefig(policy_no + "_accuracy_" + Model + "_Seed" + Plot_Seed + ".png")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
sns_plot_dis_grid_K = sns.lineplot(x = "Attacker Timestep", y = "Grid Distance", hue="Model", data=df_dis_K, ax=axs[0,0])
sns_plot_dis_beh_K = sns.lineplot(x = "Attacker Timestep", y = "Behavior Distance", hue="Model", data=df_dis_K, ax=axs[0,1])
sns_plot_dis_K = sns.lineplot(x = "Attacker Timestep", y = "Distance", hue="Model", data=df_dis_K, ax=axs[1,0])
sns_plot_time = sns.lineplot(x = "Attacker Timestep", y = "Attack Timestep Time", hue="Model", data=df_eff, ax=axs[1,1])
plt.savefig(policy_no + "_distance_K_time_" + Model + "_Seed" + Plot_Seed + ".png")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
sns_plot_dis_grid_W = sns.lineplot(x = "Attacker Timestep", y = "Grid Distance", hue="Model", data=df_dis_W, ax=axs[0,0])
sns_plot_dis_beh_W = sns.lineplot(x = "Attacker Timestep", y = "Behavior Distance", hue="Model", data=df_dis_W, ax=axs[0,1])
sns_plot_dis_W = sns.lineplot(x = "Attacker Timestep", y = "Distance", hue="Model", data=df_dis_W, ax=axs[1,0])
sns_plot_eff = sns.lineplot(x = "Attacker Timestep", y = "Effort", hue="Model", data=df_eff, ax=axs[1,1])
plt.savefig(policy_no + "_distance_W_effort_" + Model + "_Seed" + Plot_Seed + ".png")


# plt.show() # to show graph
#fig = sns_plot.get_figure()
#fig.savefig(TITLE_figure, bbox_inches='tight')