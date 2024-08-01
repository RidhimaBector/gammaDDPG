# Attack action space U: Box=(-10, +10, (16, ), float 32) -- continuous

import io
import numpy as np
import sys
import math

from gym import Env, spaces
from gym.utils import seeding


# STILL = 0
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3



"""def categorical_sample(prob_n, np_random):
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n>np_random.rand()).argmax()"""



class GridWorld_3D_env(Env):
    metadata = {'render.modes': ['human', 'ansi']}
        
    """def _generate_altitude(self, shape):
        s = 1
        wsize = shape[0]
        A = wsize*np.random.randn(s,s)
        
        # generate land
        while s < wsize:
            s = s*2
            B = (wsize/s)*np.random.rand(s,s)
            for x in range(0, s):
                for y in range(0, s):
                    B[x][y] = B[x][y] + A[math.floor(x/2)][math.floor(y/2)]
            A = B
            
        # diffuse
        B = A
        for d in range(0,2):
            for x in range(0, wsize):
                for y in range(0, wsize):
                    if (x==0 or x==(wsize-1)) and (y==0 or y==(wsize-1)):
                        pass
                    elif x==0 or x==wsize-1:
                        B[x][y] = np.mean(A[x, y-1:y+2])
                    elif y==0 or y==wsize-1:
                        B[x][y] = np.mean(A[x-1:x+2, y])
                    else:
                        B[x][y] = np.mean(A[x-1:x+2, y-1:y+2])
            A = B
            
        # make 10 different altitute levels
        A = A - np.min(A)
        A = 10*A/np.max(A)
        A = np.round(A)
        
        return A"""
    
    def _defined_altitude(self, shape):
        A = np.zeros(self.shape)
        #A[0] = [8, 7, 5, 4]
        #A[1] = [9, 6, 4, 2]
        #A[2] = [8, 5, 4, 1]
        #A[3] = [8, 4, 2, 0]
        A[0] = [8.00, 8.00, 8.00, 8.00] #[8, 7, 5, 4]
        A[1] = [6.00, 6.00, 6.00, 8.00] #[9, 6, 4, 2]
        A[2] = [4.00, 4.00, 6.00, 8.00] #[8, 5, 4, 1]
        A[3] = [2.00, 4.00, 6.00, 8.00] #[8, 4, 2, 0]
        
        """A[0] = [8.00, 8.25, 8.50, 8.75] #[8, 7, 5, 4]
        A[1] = [6.00, 6.00, 6.00, 9.00] #[9, 6, 4, 2]
        A[2] = [5.00, 5.00, 5.00, 9.25] #[8, 5, 4, 1]
        A[3] = [4.00, 3.00, 2.00, 1.00] """#[8, 4, 2, 0]
        
        return A
    
    
    def _calculate_dynamics(self, shape, nA, nS, A):
        # Base probability of struggling uphill
        Bstrug = 0.9
        # Base probability of stilding downhill
        Bslide = 0.2
        
        wsize = shape[0]
        
        T = np.zeros((nA, nS, nS))
        for a in range(nA):
            T[a] = np.eye(nS)
        
        for row in range(0, wsize):
            for col in range(0, wsize):
                
                # -- current state --
                s1 = row*wsize + col

                # -- check north --
                if row > 0:
                    s2 = (row-1)*wsize + col
                    s3 = (row-2)*wsize + col

                    # slope
                    diff = A[row-1][col] - A[row][col]

                    if row > 1:
                        slip = Bslide/(1+math.exp(2+2*diff))
                    else:
                        slip = 0
                    stay = Bstrug/(1+math.exp(2-2*diff))
                    move = 1 - slip - stay

                    T[0][s1][s1] = stay
                    T[0][s1][s2] = move
                    if row > 1:
                        T[0][s1][s3] = slip

                # -- check east --
                if col < wsize-1 :
                    s2 = row*wsize + col + 1
                    s3 = row*wsize + col + 2

                    # slope
                    diff = A[row][col+1] - A[row][col]

                    if col < wsize-2 :
                        slip = Bslide/(1+math.exp(2+2*diff))
                    else:
                        slip = 0
                    stay = Bstrug/(1+math.exp(2-2*diff))
                    move = 1 - slip - stay

                    T[1][s1][s1] = stay
                    T[1][s1][s2] = move
                    if col < wsize-2:
                        T[1][s1][s3] = slip
        
                # -- check south --
                if row < wsize-1:
                    s2 = (row+1)*wsize + col
                    s3 = (row+2)*wsize + col

                    # slope
                    diff = A[row+1][col] - A[row][col]

                    if (row+2) < wsize:
                        slip = Bslide/(1+math.exp(2+2*diff))
                    else:
                        slip = 0

                    stay = Bstrug/(1+math.exp(2-2*diff))
                    move = 1 - slip - stay

                    T[2][s1][s1] = stay
                    T[2][s1][s2] = move
                    if row+2 < wsize:
                        T[2][s1][s3] = slip
                        
                # -- check west --
                if col>0:
                    s2 = row*wsize + col - 1
                    s3 = row*wsize + col - 2

                    # slope
                    diff = A[row][col-1] - A[row][col]

                    if col > 1:
                        slip = Bslide/(1+math.exp(2+2*diff))
                    else:
                        slip = 0

                    stay = Bstrug/(1+math.exp(2-2*diff))
                    move = 1 - slip - stay

                    T[3][s1][s1] = stay
                    T[3][s1][s2] = move
                    if col > 1:
                        T[3][s1][s3] = slip
                        
        small = 0.000001
        for a in range(nA):
            T[a] = (T[a]+small)/(1+nS*small)
            
        return T
    
    def _calculate_transition_status(self, cur_s, action, T):
        
        prob = T[action][cur_s]
        next_s = np.random.choice(np.arange(len(prob)), p=prob)
        prob_next_s = prob[next_s]
        
        #next_position = np.unravel_index(next_s, self.shape)
        is_done = next_s == 12 #is_done = tuple(next_position)==(3,3) #Change_Target_Behavior #E-(3,2), M-(3,3), Mp-(3,0), H-(3,0)
        return prob_next_s, next_s, -1.0, is_done
    
    
    """def _calculate_P(self, a, nS, T):
        P = {}
        for s in range(nS):
            P[s] = {a : [] for a in range(nA)}
            P[s][NORTH] = self._calculate_transition_status(s, NORTH, T)
            P[s][EAST] = self._calculate_transition_status(s, EAST, T)
            P[s][SOUTH] = self._calculate_transition_status(s, SOUTH, T)
            P[s][WEST] = self._calculate_transition_status(s, WEST, T)
        
        return P"""
        

    def __init__(self):
        self.shape = (4, 4)
        
        self.nS = np.prod(self.shape) #16
        self.nA = 4
        
        # define attack action space -- for physical settings
        self.Attack_ActionSpace = spaces.Box(low=-1.0, high=+1.0, shape=(self.nS,), dtype=np.float64)
        
        # generate altitute
        self.altitude_default = self._defined_altitude(self.shape)
        
        self.altitude = self.altitude_default.copy()
        
        # environment transition
        self.T = self._calculate_dynamics(self.shape, self.nA, self.nS, self.altitude)
            
        # always start in state (0,0)
        #self.isd = np.zeros(self.nS)
        #self.isd[0] = 1.0 #[np.ravel_multi_index((0,0), self.shape)] = 1.0
        # target state
        #self.goal_s = 12 # Victim Goal State               #self.target_s = 12 #np.ravel_multi_index((3,3), self.shape) #Change_Target_Behavior #E-(3,2), M-(3,3), Mp-(3,0), H-(3,0)
        
        self.lastaction = None # for rendering
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        
        self.seed()
        self.s = 0 #categorical_sample(self.isd, self.np_random)

        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = 0 #categorical_sample(self.isd, self.np_random) #start state
        self.lastaction = None
        return self.s

    def reset_altitude(self):
        self.altitude = self.altitude_default.copy()
        self.T = self._calculate_dynamics(self.shape, self.nA, self.nS, self.altitude)
#         return self.altitude

    def step(self, action):
        prob, next_state, reward, done = self._calculate_transition_status(self.s, action, self.T) #P[self.s][action]
        #i = categorical_sample([t[0] for t in transitions], self.np_random)
        #p, s, r, d = transition[0]
        self.s = next_state
        self.lastaction = action
        return (next_state, reward, done, {"prob" : prob})
        
    """def render(self, mode='human', close=False):
        self._render(mode, close)
        
    def _render(self, mode='human', close=False):
        if close:
            return
        #outfile = StringIO() if mode == 'ansi' else sys.stdout
        
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position == (3,3): #Change_Target_Behavior #E-(3,2), M-(3,3), Mp-(3,0), H-(3,0)
                output = " T "
            else:
                output = " o "
                
            if position[1]==0:
                output = output.lstrip()
            if position[1]==self.shape[1]-1:
                output = output.rstrip()
                output += "\n"
                
            #outfile.write(output)
            
        #outfile.write("\n")"""

    def Attack_Env(self, U):
        A = np.clip(self.altitude + U.reshape(self.shape), 0.0, 10.0)

        self.altitude = A
        self.T = self._calculate_dynamics(self.shape, self.nA, self.nS, A)
            
        


