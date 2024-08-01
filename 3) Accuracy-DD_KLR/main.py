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
#from victim.system import System
from victim.victim_Q import VictimAgent
from ae.ae import AutoEncoder

#Cost Matrix
grid = np.array([[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]])
cost_matrix = ot.dist(grid, grid, metric='cityblock') * 20
np.fill_diagonal(cost_matrix, 10)
cost_matrix = np.repeat(cost_matrix, env.nA, axis=1)
cost_matrix = np.repeat(cost_matrix, env.nA, axis=0)
np.fill_diagonal(cost_matrix, 0)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_dir", default="R_0818")               # TensorBoard folder
    
    parser.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    #parser.add_argument("--atk_training_start_episodes", default=100, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eps_greedy_start_episodes", default=30, type=int)  # Time steps initial random policy is used
    parser.add_argument("--max_timesteps", default=15, type=int)   # Max time steps to run environment
    parser.add_argument("--max_episodes_num", default=30000, type=int)   # Max episodes to run environment
    parser.add_argument("--eval_freq_episode", default=200, type=int)        # How often (time steps) we evaluate
    parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.95, type=float)     # Discount factor of attack network
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
    
    parser.add_argument("--victim_n_episodes", default=80, type=int)  # number of episodes for victim's updated in poisoned Env
    parser.add_argument("--ae_n_epochs", default=10, type=int)         # number of training epoch
    
    args = parser.parse_args()

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ''' ..... Tensorboard Settings ..... '''

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"results_{date}"
    
    model_name = args.model_dir or default_model_name
    model_dir = utils_log.get_model_dir(model_name)
    
    ''' ..... Attack Network ..... '''
    # Input / Output size
    state_dim = EMBEDDING_SIZE + env.nS
    action_dim = env.Attack_ActionSpace.shape[0]
    max_action = float(env.Attack_ActionSpace.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    
    kwargsNew = {
        "seed": args.seed,
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
        "tau": args.tau,
        "discount": args.discount,
        "epsilon_divisor": 1000, #Number of episodes over which to decrease actual epsilon
        "is_training": True
    }

    # Initialize Policy
    Policy = DDPG(**kwargsNew)
    Buffer = utils_buf.ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    
    ''' ..... Victim ..... '''
    victim_args = {
        "env": env, 
        "MEMORY_SIZE": MEMORY_SIZE,
        "discount_factor": 0.9, #1.0, 
        "alpha": 0.1, 
        "epsilon": 0.1,
    }
    
    victim = VictimAgent(**victim_args)
    
    ae_enc_in_size = 32 #SEQ_LEN*2
    ae_enc_out_size = 5 #EMBEDDING_SIZE
    ae_dec_in_size = 6 #1+EMBEDDING_SIZE
    ae_dec_out_size = 5 #4 #action_dim
    
    ae_args = {
        "enc_in_size": ae_enc_in_size, 
        "enc_out_size": ae_enc_out_size, 
        "dec_in_size": ae_dec_in_size, 
        "dec_out_size": ae_dec_out_size, 
        "lr": 0.001, #0.001, 
    }
    
    ae = AutoEncoder(**ae_args)
    ae.load("storage/R_0818/" + "340240" + "_f-o_AutoEncoder_SftMx") #"14800" + "_f-o_AutoEncoder") #44780 - p-o M/H 
    #system = System(victim_args, ae_args)

    atk_n_epoch = 1 #15
    atk_n_batch = 15
    ddpg_loss = []
    distance_buffer = [] #complete data
    distance_grid_buffer = [] #complete data
    distance_behavior_buffer = [] #complete data
    accuracy_buffer = [] #complete data
    accuracy_sftmx_buffer = [] #complete data
    accuracy_sftmx_complete_buffer = [] #complete data
    effort_buffer = [] #complete data
    time_buffer = [] #complete data
    model_data = [] #per episode statistic wrt all metrics
    model_good_data = [] #all model statistic for best models
    model_bad_data = [] #all model statistic for worst models
    
    best_model_stats = np.zeros((24,1)) #best statistics so far (across diff models)
    worst_model_stats = np.zeros((24,1)) #worst statistics so far (across diff models)


    ''' ..... Training ..... '''

    for i_episode in range(args.max_episodes_num):
        print(f"\n--------- Episode: {i_episode} ----------") #txt_logger.info(f"\n--------- Episode: {i_episode} ----------")
        cumulative_distance = 0
        cumulative_distance_grid = 0
        cumulative_distance_behavior = 0
        cumulative_accuracy = 0
        cumulative_accuracy_sftmx = 0
        cumulative_accuracy_sftmx_complete = 0
        cumulative_effort = 0
        cumulative_time = 0
        
        # reset victim's env and Q
        env.reset_altitude()
        victim.reset() #victim_Q = np.zeros((16,4)) #system.victim.reset()

        # Initialize the attacker's state
        victim_info = np.zeros((1,EMBEDDING_SIZE))
        victim_tensor = torch.from_numpy(victim_info)
        victim_tensor_4d = victim_tensor.unsqueeze(0).unsqueeze(0)

        env_info = env.altitude.copy()
        env_tensor = torch.from_numpy(env_info)
        env_tensor = env_tensor.view(1, env.nS)
        env_tensor_4d = env_tensor.unsqueeze(0).unsqueeze(0)

        x = torch.cat((victim_tensor_4d, env_tensor_4d), 3)
        curA = env.altitude.copy().reshape((16, 1))
        
        for t in range(args.max_timesteps):
            tic_timestep = time.time()

            # Select attack_action
            if i_episode < args.eps_greedy_start_episodes:
                u = env.Attack_ActionSpace.sample()
            else:
                u = Policy.select_ddpg_action(np.array(x))

            # Step: implement attack_action
            env.Attack_Env(u)

            # Step: victim updates = get next_x
            victim_transitions = victim.Train_Model(80) #system.train(args.victim_n_episodes, args.ae_n_epochs)
            ### ... updated victim.Q
            next_victim_info = ae.Policy_Embedding(victim_transitions) #next_victim_info = system.ae.Embedding(system.victim.MEM)
            next_victim_tensor = torch.from_numpy(next_victim_info[-1]).unsqueeze(0)
            next_victim_tensor_4d = next_victim_tensor.unsqueeze(0).unsqueeze(0)
            ### ... updated env altitude
            next_env_info = env.altitude.copy() #system.victim.env.altitude.copy()
            next_env_tensor = torch.from_numpy(next_env_info)
            next_env_tensor = next_env_tensor.view(1, env.nS)
            next_env_tensor_4d = next_env_tensor.unsqueeze(0).unsqueeze(0)
            ### ... next_state
            next_x = torch.cat((next_victim_tensor_4d, next_env_tensor_4d), 3)
            
            # Step: cost
            distance = - utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim.Q, TARGET, cost_matrix, distance_type=0) #Dynamic Discount KLR - Algorithm 2 Line 12
            distance_grid = - utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim.Q, TARGET, cost_matrix, distance_type=1)
            distance_behavior = - utils_attack.Attack_Cost_Compute_K(env, INIT_T, victim.Q, TARGET, cost_matrix, distance_type=2)
            done, accuracy, accuracy_sftmx, accuracy_sftmx_complete = utils_attack.Attack_Done_Identify(env, TARGET, victim.Q)
            effort, curA = utils_attack.Attack_Effort(curA, env)
            effort = - effort
            toc_timestep = time.time()
            time_timestep = tic_timestep - toc_timestep #- (toc_timestep - tic_timestep)

            ### log
            no_episodes = i_episode+1
            no_timesteps = t+1
            distance_buffer.append([no_episodes, no_timesteps, distance])
            distance_grid_buffer.append([no_episodes, no_timesteps, distance_grid])
            distance_behavior_buffer.append([no_episodes, no_timesteps, distance_behavior])
            accuracy_buffer.append([no_episodes, no_timesteps, accuracy])
            accuracy_sftmx_buffer.append([no_episodes, no_timesteps, accuracy_sftmx])
            accuracy_sftmx_complete_buffer.append([no_episodes, no_timesteps, accuracy_sftmx_complete])
            effort_buffer.append([no_episodes, no_timesteps, effort])
            time_buffer.append([no_episodes, no_timesteps, time_timestep])

            reward = accuracy
            cumulative_distance += distance
            cumulative_distance_grid += distance_grid
            cumulative_distance_behavior += distance_behavior
            cumulative_accuracy += accuracy
            cumulative_accuracy_sftmx += accuracy_sftmx
            cumulative_accuracy_sftmx_complete += accuracy_sftmx_complete
            cumulative_effort += effort
            cumulative_time += time_timestep

            # Replay buffer
            Buffer.add(x.view(state_dim), u, next_x.view(state_dim), reward, -distance, done) #$\gamma$DDPG with Dynamic Discount (-discount) KLR - Algorithm 2 Line 13

            # Update state
            x = copy.deepcopy(next_x)
            if (done or (no_timesteps == args.max_timesteps)):
                
                cur_model_stats = np.array([distance, cumulative_distance/no_timesteps, cumulative_distance, \
                                            distance_grid, cumulative_distance_grid/no_timesteps, cumulative_distance_grid, \
                                            distance_behavior, cumulative_distance_behavior/no_timesteps, cumulative_distance_behavior, \
                                            accuracy, cumulative_accuracy/no_timesteps, cumulative_accuracy, \
                                            accuracy_sftmx, cumulative_accuracy_sftmx/no_timesteps, cumulative_accuracy_sftmx, \
                                            accuracy_sftmx_complete, cumulative_accuracy_sftmx_complete/no_timesteps, cumulative_accuracy_sftmx_complete, \
                                            effort, cumulative_effort/no_timesteps, cumulative_effort, \
                                            time_timestep, cumulative_time/no_timesteps, cumulative_time])
                model_data.append([no_episodes, no_timesteps])
                model_data[-1].extend(cur_model_stats.tolist())
                
                if(i_episode == 0):
                    best_model_stats = copy.deepcopy(cur_model_stats)
                    worst_model_stats = copy.deepcopy(cur_model_stats)
                    break
                
                cur_model_is_good = cur_model_stats >= best_model_stats
                cur_model_is_bad = cur_model_stats <= worst_model_stats
                
                if(np.sum(cur_model_is_good) > 0):
                    model_good_data.append([no_episodes, no_timesteps])
                    model_good_data[-1].extend(cur_model_stats.tolist())
                    model_good_data[-1].extend(best_model_stats.tolist())
                    model_good_data[-1].extend(worst_model_stats.tolist())
                    Policy.save(f"./{model_dir}/good_model_{no_episodes}")
                    
                    best_model_stats[cur_model_is_good] = cur_model_stats[cur_model_is_good]
                    
                elif(np.sum(cur_model_is_bad) > 0):
                    model_bad_data.append([no_episodes, no_timesteps])
                    model_bad_data[-1].extend(cur_model_stats.tolist())
                    model_bad_data[-1].extend(best_model_stats.tolist())
                    model_bad_data[-1].extend(worst_model_stats.tolist())
                    Policy.save(f"./{model_dir}/bad_model_{no_episodes}")
                    
                    worst_model_stats[cur_model_is_bad] = cur_model_stats[cur_model_is_bad]

                break

        # Attack_Policy Update
        if i_episode >= args.eps_greedy_start_episodes:
            ddpg_loss = Policy.train(Buffer, atk_n_epoch, atk_n_batch, args.batch_size, ddpg_loss, i_episode)      
        ''' save Attack_Policy '''
        if no_episodes % args.eval_freq_episode == 0:
            Buffer.saveBuffer(f"./{model_dir}/")
            Policy.save(f"./{model_dir}/{no_episodes}")
            np.savetxt("ddpg_loss.csv", np.array(ddpg_loss), delimiter=",")
            np.savetxt("distance_buffer.csv", np.array(distance_buffer), delimiter=",")
            np.savetxt("distance_grid_buffer.csv", np.array(distance_grid_buffer), delimiter=",")
            np.savetxt("distance_behavior_buffer.csv", np.array(distance_behavior_buffer), delimiter=",")
            np.savetxt("accuracy_buffer.csv", np.array(accuracy_buffer), delimiter=",")
            np.savetxt("accuracy_sftmx_buffer.csv", np.array(accuracy_sftmx_buffer), delimiter=",")
            np.savetxt("accuracy_sftmx_complete_buffer.csv", np.array(accuracy_sftmx_complete_buffer), delimiter=",")
            np.savetxt("effort_buffer.csv", np.array(effort_buffer), delimiter=",")
            np.savetxt("time_buffer.csv", np.array(time_buffer), delimiter=",")
            np.savetxt("model_data.csv", np.array(model_data), delimiter=",")
            np.savetxt("model_good_data.csv", np.array(model_good_data), delimiter=",")
            np.savetxt("model_bad_data.csv", np.array(model_bad_data), delimiter=",")