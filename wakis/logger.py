# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2025.                   #
# ########################################### #

from tqdm import tqdm

import numpy as np
import time
import h5py
import os
import json

from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0
from scipy.sparse import csc_matrix as sparse_mat
from scipy.sparse import diags, hstack, vstack



class Logger():

    def __init__(self):
        self.grid = {}
        self.solver = {}
        self.wakeSolver = {}

    def save_logs(self):
        """
        Save all logs (grid, solver, wakeSolver) into log-file inside the results folder.
        """
        logfile = os.path.join(self.wakeSolver["results_folder"], "wakis.log")

        # Write sections
        if not os.path.exists(self.wakeSolver["results_folder"]): 
            os.mkdir(self.wakeSolver["results_folder"])
        
        with open(logfile, "w", encoding="utf-8") as fh:
            fh.write("Simulation Parameters\n")
            fh.write("""=====================\n\n""")

            sections = [
                ("WakeSolver Logs", self.wakeSolver),
                ("Solver Logs", self.solver),
                ("Grid Logs", self.grid),
            ]

            for title, data in sections:
                fh.write(f"\n## {title} ##\n")
                if not data:
                    fh.write("(empty)\n")
                    continue

                # convert non-serializable values to strings recursively
                def _convert(obj):
                    if isinstance(obj, dict):
                        return {k: _convert(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_convert(v) for v in obj]
                    try:
                        json.dumps(obj)
                        return obj
                    except Exception:
                        return str(obj)

                clean = _convert(data)
                fh.write(json.dumps(clean, indent=2, ensure_ascii=False))
                fh.write("\n")
