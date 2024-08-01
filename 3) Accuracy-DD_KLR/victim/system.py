import os
from os.path import dirname, abspath
import sys
if "../" not in sys.path:
    sys.path.append("../") 
    
from victim.victim_Q import VictimAgent
from ae.ae_policy import AutoEncoder

''' import configuration '''
from yacs.config import CfgNode as CN
yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
MEMORY_SIZE = config.AE.MEMORY_SIZE



''' import target_Memory '''
from envs.target_def import TARGET

class System():
    def __init__(self, victim_args, ae_args):
        self.victim = VictimAgent(**victim_args)
        self.ae = AutoEncoder(**ae_args)
        
    def train(self, victim_episodes_num = 50, ae_episodes_num = 1):
        self.ae.n_epochs = ae_episodes_num
        vae_loss_list = []
        for i_episode in range(victim_episodes_num):
            
            self.victim.Train_Model(1)
            
            if self.victim.MEM.__len__() >= MEMORY_SIZE:
                vae_loss_list.extend(self.ae.Train(self.victim.MEM))
        return vae_loss_list
                
    def eval_victim(self):
        self.victim.Eval_Model()