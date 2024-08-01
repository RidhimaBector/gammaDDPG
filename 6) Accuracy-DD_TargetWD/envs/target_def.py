from collections import defaultdict
import numpy as np

# Environment object
from .env3D_4x4 import GridWorld_3D_env
env = GridWorld_3D_env()

#TARGET = defaultdict(lambda: np.zeros(env.action_space.n))

#E target path
#TARGET[0] = np.array([0, 0, 1, 0])
#TARGET[4] = np.array([0, 0, 1, 0])
#TARGET[8] = np.array([0, 0, 1, 0])
#TARGET[12] = np.array([0, 1, 0, 0])
#TARGET[13] = np.array([0, 1, 0, 0])

#TARGET[0] = np.array([0, 1, 0, 0])
#TARGET[1] = np.array([0, 1, 0, 0])
#TARGET[2] = np.array([0, 1, 0, 0])
#TARGET[3] = np.array([0, 0, 1, 0])
#TARGET[7] = np.array([0, 0, 1, 0])
#TARGET[11] = np.array([0, 0, 1, 0])

# M and H target paths
"""TARGET = np.zeros((16,4))
TARGET[0][1] = 1 #np.array([0, 0, 1, 0])
TARGET[1][1] = 1 #np.array([0, 0, 1, 0])
TARGET[2][1] = 1 #np.array([0, 0, 1, 0])
TARGET[3][2] = 1 #np.array([0, 1, 0, 0])
TARGET[7][2] = 1 #np.array([0, 1, 0, 0])
TARGET[11][2] = 1 #np.array([0, 1, 0, 0])"""

#Mp target path
TARGET = np.zeros((16,4))
TARGET[0][1] = 1 #np.array([0, 0, 1, 0])
TARGET[1][1] = 1 #np.array([0, 0, 1, 0])
TARGET[2][1] = 1 #np.array([0, 0, 1, 0])
TARGET[3][2] = 1 #np.array([0, 1, 0, 0])
TARGET[7][2] = 1 #np.array([0, 1, 0, 0])
TARGET[11][2] = 1 #np.array([0, 1, 0, 0])
TARGET[15][3] = 1 #np.array([0, 1, 0, 0])
TARGET[14][3] = 1 #np.array([0, 1, 0, 0])
TARGET[13][3] = 1 #np.array([0, 1, 0, 0])