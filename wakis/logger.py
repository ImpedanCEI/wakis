# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

from tqdm import tqdm

import numpy as np
import time
import h5py

from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0
from scipy.sparse import csc_matrix as sparse_mat
from scipy.sparse import diags, hstack, vstack



class Logger():

    def __init__(self):
        self.grid_logs = None
        self.solver_logs = None
        self.wakeSolver_logs = None

    def assign_grid_logs(self, grid_logs):
        self.grid_logs = grid_logs

    def assign_solver_logs(self, solver_logs):
        self.solver_logs = solver_logs

    def assign_wakeSolver_logs(self, wakeSolver_logs):
        self.wakeSolver_logs = wakeSolver_logs

    def save_logs(self, filename):
        print('Here I would save the logs to file:' + self.grid_logs + self.solver_logs + self.wakeSolver_logs)