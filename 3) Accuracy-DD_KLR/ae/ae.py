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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        x = F.softmax(self.fc2(x), dim=1)
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
        
    def forward(self, x, n_target_states, states):
        #x = torch.cat((s, a),1)
        z = self.encoder(x)
        
        z = torch.unsqueeze(z, 1)
        x = z.repeat(1,n_target_states,1) #torch.cat(len(s)*[z])
        x = torch.cat((states, x), 2)
        
        x = self.decoder(x)
        return x

    
class AutoEncoder():
    def __init__(self, enc_in_size, enc_out_size, dec_in_size, dec_out_size, lr):

        self.lr = lr
        self.Encoder = Policy_Representation(enc_in_size, enc_out_size).to(device)
        self.Decoder = Policy_Imitation(dec_in_size, dec_out_size).to(device)
        self.Model = Encoder_Decoder(self.Encoder, self.Decoder, self.lr).to(device)
    
    """ Training the Encoder-Decoder network """
    def Train(self, epoch, n_epoch, n_batch, batch_size):

        self.Model.train()
        for i_epoch in range(n_epoch):

            total_loss = 0.0 #Assuming always 1 epoch of training at a time
            for i_batch in range(n_batch):
                
                n_target_states = 16 #6
                states = np.tile(np.expand_dims(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), axis=0),(batch_size,1))
                states = np.expand_dims(states, axis=2)
                actions = np.random.choice([0,1,2,3,4], size=(batch_size,n_target_states,1), replace=True)
                batch = np.concatenate((states, actions),2)
                
                states = torch.FloatTensor(states).to(device) #torch.FloatTensor([states]).view(-1,1).to(device)
                actions = torch.FloatTensor(actions).to(device) #torch.FloatTensor([actions]).view(-1,1).to(device)
                batch = torch.FloatTensor(batch).to(device)
                target = actions.long().reshape(batch_size*n_target_states,1).squeeze(1) #actions.long().view(1,6).squeeze(0)

                """ train """
                # clear the gradients of all optimized variables
                self.Model.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.Model(batch, n_target_states, states)
                # calculate the loss
                loss = self.Model.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.Model.optimizer.step()
                # update running training loss
                #loss_list.append([epoch, i_batch, loss.item()]) #*state.size(0)
                total_loss += loss.item()
                print("Epoch: " + str(epoch) + ", Batch: " + str(i_batch) + ", AE Loss = " + str(loss.item()))
            
        return total_loss/n_batch
#             print('Epoch: {} \tTraining Loss: {:.6f}'.format(i_epoch+1, train_loss))


    """ Embedding using Encoder network """
    def Policy_Embedding(self, victim_transitions):
        self.Model.eval() # prep model for *evaluation*
        batch = torch.FloatTensor(victim_transitions).to(device)
        z = self.Model.encoder(batch).cpu().data.numpy()
        return z
    
    def save(self, filename):
        torch.save(self.Model.state_dict(), filename + "_AutoEncoder")

    def load(self, filename):
        load_model = self.Model.load_state_dict(torch.load(filename, map_location=device))  #torch.device('cpu')))
        return load_model

    
if __name__ == "__main__":
    
    """ Intialize AutoEncoder """
    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)

    ae_enc_in_size = 32 # 12 as 6 target states #SEQ_LEN*2 = 5*2
    ae_enc_out_size = 5 # EMBEDDING_SIZE = 5
    ae_dec_in_size = 6 #1+EMBEDDING_SIZE
    ae_dec_out_size = 5 #action_dim 0,1,2,3,4 - 4 for target states that victim has not visited yet
    
    ae_args = {
        "enc_in_size": ae_enc_in_size, 
        "enc_out_size": ae_enc_out_size, 
        "dec_in_size": ae_dec_in_size, 
        "dec_out_size": ae_dec_out_size, 
        "lr": 0.001, #0.001, 
    }
    
    ae = AutoEncoder(**ae_args)
    print("AE Initialized")
    
    epoch = 1
    n_epoch = 1
    n_batch = 500
    batch_size = 256
    ae_loss_list = []
    while epoch > 0:
        ae_loss_list.append([epoch, ae.Train(epoch, n_epoch, n_batch, batch_size)])
        ae.save(f"./{epoch}")
        if(epoch % 20 == 0):
            np.savetxt("AE_loss_buffer.csv", np.array(ae_loss_list), delimiter=",")
        
        epoch = epoch + 1

    """states = np.expand_dims(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), axis=1)
    actions = np.random.choice([0,1,2,3,4], size=(16,1), replace=True)
    victim_transitions = np.concatenate((states, actions),1)
    ae.Policy_Embedding(victim_transitions)"""
