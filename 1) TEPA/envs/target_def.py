from collections import defaultdict
import numpy as np

# Environment object
from .env3D_4x4 import GridWorld_3D_env
env = GridWorld_3D_env()

TARGET = defaultdict(lambda: np.zeros(env.action_space.n))

#TARGET[0] = np.array([0, 0, 1, 0])
#TARGET[4] = np.array([0, 0, 1, 0])
#TARGET[8] = np.array([0, 0, 1, 0])
#TARGET[12] = np.array([0, 1, 0, 0])
#TARGET[13] = np.array([0, 1, 0, 0])

#00,01,02,03
#04,05,06,07
#08,09,10,11
#12,13,14,15

# Mp
TARGET[0] = np.array([0, 1, 0, 0])
TARGET[1] = np.array([0, 1, 0, 0])
TARGET[2] = np.array([0, 1, 0, 0])
TARGET[3] = np.array([0, 0, 1, 0])
TARGET[7] = np.array([0, 0, 1, 0])
TARGET[11] = np.array([0, 0, 1, 0])
TARGET[15] = np.array([0, 0, 0, 1])
TARGET[14] = np.array([0, 0, 0, 1])
TARGET[13] = np.array([0, 0, 0, 1])

# M and H paths
"""TARGET[0] = np.array([0, 1, 0, 0])
TARGET[1] = np.array([0, 1, 0, 0])
TARGET[2] = np.array([0, 1, 0, 0])
TARGET[3] = np.array([0, 0, 1, 0])
TARGET[7] = np.array([0, 0, 1, 0])
TARGET[11] = np.array([0, 0, 1, 0])"""