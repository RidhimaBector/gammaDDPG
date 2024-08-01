import numpy as np
from collections import namedtuple
import torch

import matplotlib
import matplotlib.pyplot as plt

""" DDPG buffer"""
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		"""return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)"""
        
		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind]
		)
    
	def saveBuffer(self, filename):
		np.save(filename +'_ptr.npy', np.array([self.ptr]))
		np.save(filename +'_size.npy', np.array([self.size]))
		np.save(filename +'_state.npy', self.state)
		np.save(filename +'_action.npy', self.action)
		np.save(filename +'_next_state.npy', self.next_state)
		np.save(filename +'_reward.npy', self.reward)
		np.save(filename +'_not_done.npy', self.not_done)

	def loadBuffer(self, filename):
		self.ptr = np.load(filename +'_ptr.npy')[0]
		self.size = np.load(filename +'_size.npy')[0]
		self.state = np.load(filename +'_state.npy')
		self.action = np.load(filename +'_action.npy')
		self.next_state = np.load(filename +'_next_state.npy')
		self.reward = np.load(filename +'_reward.npy')
		self.not_done = np.load(filename +'_not_done.npy')
    

""" Victim Memory"""
Transition = namedtuple('Transition', ('state', 'action'))
class Memory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


