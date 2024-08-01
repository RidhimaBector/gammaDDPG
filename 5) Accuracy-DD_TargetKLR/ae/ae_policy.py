import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import argparse
import os
import copy

from collections import defaultdict
from itertools import count

import os
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
    sys.path.append("../") 

from utils.utils_buf import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
SEQ_LEN = config.AE.SEQ_LEN
EMBEDDING_SIZE = config.AE.EMBEDDING_SIZE
MEMORY_SIZE = config.AE.MEMORY_SIZE


""" Define the Policy_Embedding Network"""

class Policy_Representation(nn.Module):
    def __init__(self, input_size, embedding_size, fc1_units=36, fc2_units=36):
        super(Policy_Representation, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.ln2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, embedding_size)
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
#         print(f"x_1 = {x}")
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x
    

""" Define the Policy_Imitation Network"""

class Policy_Imitation(nn.Module):
    def __init__(self, input_size, action_size, fc1_units=36, fc2_units=36):
        super(Policy_Imitation, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    
""" Combine Policy_Representation with Policy_Imitation """

class Encoder_Decoder(nn.Module):
    def __init__(self, embedding, imitation, lr):
        super(Encoder_Decoder, self).__init__()
        self.encoder = embedding
        self.decoder = imitation
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        
    def forward(self, s, a):
        x = torch.cat((s, a),1)
        z = self.encoder(x)
        
        x = z.repeat(len(s),1) #torch.cat(len(s)*[z])
        x = torch.cat((s, x), 1)
        
        x = self.decoder(x)
        return x

class Encoder_Decoder_new(nn.Module):
    def __init__(self, enc_in_size, enc_out_size, dec_in_size, dec_out_size, lr, fc1_units=36, fc2_units=36):
        super(Encoder_Decoder, self).__init__()
        self.enc_in_size = enc_in_size
        self.e_fc1 = nn.Linear(enc_in_size, fc1_units)
        self.e_ln1 = nn.LayerNorm(fc1_units)
        self.e_fc2 = nn.Linear(fc1_units, fc2_units)
        self.e_ln2 = nn.LayerNorm(fc2_units)
        self.e_fc3 = nn.Linear(fc2_units, enc_out_size)
        
        self.dec_in_size = dec_in_size
        self.d_fc1 = nn.Linear(input_size, fc1_units)
        self.d_ln1 = nn.LayerNorm(fc1_units)
        self.d_fc2 = nn.Linear(fc1_units, dec_out_size)
        
        self.encoder = embedding
        self.decoder = imitation
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        
    def forward(self, s, a):
        x = torch.cat((s, a),1)
        z = self.encoder(x).data
        
        x = x.view(-1, self.input_size)
#         print(f"x_1 = {x}")
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        
        x = torch.cat(len(s)*[z])
        x = torch.cat((s, x), 1)
        
        x = self.decoder(x)
        
        x = x.view(-1, self.input_size)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class AutoEncoder():
    
    def __init__(self, enc_in_size, enc_out_size, dec_in_size, dec_out_size, n_epochs=50, lr=0.001):
        
        self.n_epochs = n_epochs
        self.lr = lr
        self.Encoder = Policy_Representation(enc_in_size, enc_out_size).to(device)
        self.Decoder = Policy_Imitation(dec_in_size, dec_out_size).to(device)
        self.Model = Encoder_Decoder(self.Encoder, self.Decoder, self.lr).to(device)
    
    """ Training the Encoder-Decoder network """

    def Train(self, memory):

        self.Model.train()
        
        n_trajectory = MEMORY_SIZE//SEQ_LEN
        vae_loss_list = []
        for i_epoch in range(self.n_epochs):
            # monitor training loss
            train_loss = 0.0

            for n in range(n_trajectory):
                """ training data """
                i_start = n*SEQ_LEN
                i_end = i_start + SEQ_LEN
                transitions = memory.memory[i_start: i_end]

                # extract state and action
                batch = Transition(*zip(*transitions))
                
                state = torch.FloatTensor([batch.state]).view(-1,1).to(device)
                action = torch.FloatTensor([batch.action]).view(-1,1).to(device)
                target = action.long().view(1,SEQ_LEN).squeeze(0)

                """ train """
                # clear the gradients of all optimized variables
                self.Model.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.Model(state, action)
                # calculate the loss
                loss = self.Model.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.Model.optimizer.step()
                # update running training loss
                train_loss += loss.item() #*state.size(0)

            train_loss = train_loss/(n_trajectory) #*SEQ_LEN)
            vae_loss_list.append(train_loss)
        return vae_loss_list
#             print('Epoch: {} \tTraining Loss: {:.6f}'.format(i_epoch+1, train_loss))


    """ Embedding using Encoder network """
    
    def Embedding(self, memory):

        """ Generate Embedding """
        self.Model.eval() # prep model for *evaluation*
        
        n_trajectory = MEMORY_SIZE//SEQ_LEN
        print
        
        Z_list = []

        for n in range(n_trajectory):
            """ training data """
            i_start = n*SEQ_LEN
            i_end = i_start + SEQ_LEN
            transitions = memory.memory[i_start: i_end]

            # extract state and action
            batch = Transition(*zip(*transitions))
            state = torch.FloatTensor([batch.state]).view(-1,1).to(device)
            action = torch.FloatTensor([batch.action]).view(-1,1).to(device)

            x = torch.cat((state, action), 1)

            """ Get embedding """
            z = self.Encoder(x)
            Z_list.append(z[0].cpu().data.numpy())
            
        Z_np = np.vstack(Z_list)

        return Z_np
    
    def save(self, filename):
        torch.save(self.Model.state_dict(), filename + "_AutoEncoder")

    def load(self, filename):
        load_model = self.Model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

        return load_model
    

    
if __name__ == "__main__":
    
    """ Intialize AutoEncoder """
    state_dim = 1
    action_dim = 4

    ae_enc_in_size = SEQ_LEN*2
    ae_enc_out_size = EMBEDDING_SIZE
    ae_dec_in_size = state_dim+EMBEDDING_SIZE
    ae_dec_out_size = action_dim
    
    ae_args = {
        "enc_in_size": ae_enc_in_size, 
        "enc_out_size": ae_enc_out_size, 
        "dec_in_size": ae_dec_in_size, 
        "dec_out_size": ae_dec_out_size, 
        "n_epochs": 50, 
        "lr": 0.001, 
    }
    
    ae = AutoEncoder(**ae_args)
    print("Initialize AE")
    
    """ intialize victim """
    from envs.env3D_4x4 import GridWorld_3D_env
    env = GridWorld_3D_env()
    
    victim_args = {
        "env": env, 
        "MEMORY_SIZE": 100, 
        "discount_factor": 1.0, 
        "alpha": 0.1, 
        "epsilon": 0.1,
    }
    
    from victim.victim_Q import VictimAgent
    victim = VictimAgent(**victim_args)
    victim.Train_Model(10)
    print("Initialize Victim")
    
    """ evaluate train() """
#     ae.Embedding(victim.MEM)
    print("Training Start")
    ae.Train(victim.MEM)
    print("Training Terminate")




