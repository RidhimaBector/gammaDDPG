import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from attack.util import *
from attack.random_process import OrnsteinUhlenbeckProcess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# Implementation: https://github.com/ghliu/pytorch-ddpg


criterion = nn.MSELoss()


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, max_action, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
        self.max_action = max_action
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.max_action * self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        #x, a = xs
        out = self.fc1(state)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,action],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out

class DDPG(object):
    def __init__(self, seed, nb_states, nb_actions, max_action, hidden1, hidden2, init_w, prate, rate, ou_theta, ou_mu, ou_sigma, bsize, tau, discount, epsilon_divisor, is_training):
        
        if seed > 0:
            self.seed(seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        self.max_action = max_action
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':hidden1, 
            'hidden2':hidden2, 
            'init_w':init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, self.max_action, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, self.max_action, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        #self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=ou_theta, mu=ou_mu, sigma=ou_sigma)

        # Hyper-parameters
        self.batch_size = bsize
        self.tau = tau
        self.discount = discount
        self.depsilon = 1.0 / epsilon_divisor

        # 
        self.epsilon = 1.0
        #self.s_t = None # Most recent state
        #self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()
        
    def select_ddpg_action(self, state, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(state.reshape(1, -1)))
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon = max(self.epsilon - self.depsilon, 0.01)
        
        #self.a_t = action
        return action
    
    def select_random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions) #(16,)
        #self.a_t = action
        return action
    
    def select_on_policy_action(self, state): #state: (1,21)
        action = to_numpy(
            self.actor(to_tensor(state.reshape(1, -1)))
        ).squeeze(0)

        #self.a_t = action
        return action
    
    def train(self, replay_buffer, atk_n_epoch, atk_n_batch, batch_size, ddpg_loss, i_episode):
        
        self.is_training = True
        for i_atk_n_epoch in range(atk_n_epoch):
            
            loss_critic = 0.0
            loss_actor = 0.0
            for i_atk_n_batch in range(atk_n_batch):

                # Sample batch
                state_batch, action_batch, next_state_batch, \
                reward_batch, terminal_batch = replay_buffer.sample(self.batch_size) #self.memory.sample_and_split(self.batch_size)
        
                # Prepare for the target q batch
                next_q_values = self.critic_target(
                    to_tensor(next_state_batch, volatile=True),
                    self.actor_target(to_tensor(next_state_batch, volatile=True)))
                next_q_values.volatile=False
        
                target_q_batch = to_tensor(reward_batch) + \
                    self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values
        
                # Critic update
                self.critic.zero_grad()
        
                q_batch = self.critic( to_tensor(state_batch), to_tensor(action_batch) )
                
                value_loss = criterion(q_batch, target_q_batch)
                value_loss.backward()
                self.critic_optim.step()
        
                # Actor update
                self.actor.zero_grad()
        
                policy_loss = -self.critic(
                    to_tensor(state_batch),
                    self.actor(to_tensor(state_batch))
                )
        
                policy_loss = policy_loss.mean()
                policy_loss.backward()
                self.actor_optim.step()
        
                # Target update
                soft_update(self.actor_target, self.actor, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)
                
                loss_critic += value_loss.item()
                loss_actor += policy_loss.item()
            
            ddpg_loss.append([i_episode, i_atk_n_epoch, loss_critic/atk_n_batch, loss_actor/atk_n_batch])
        
        return ddpg_loss
    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optim.state_dict(), filename + "_critic_optimizer")
		
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optim.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=torch.device('cpu')))
        self.critic_optim.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=torch.device('cpu')))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
        self.actor_optim.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))
        self.actor_target = copy.deepcopy(self.actor)

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
    
    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
		