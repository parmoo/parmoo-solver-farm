""" Apply ParMOO to solve a chemical design problem, based on a "fake" CFR
simulation that was created using data gathered via NMR spectroscopy from
real world CFR experiments.

Exploit the fact that one objective is just a design variable that can be
directly controlled.

Execute via the following command (where the optional int M is a random seed
in [0, 2^32-1]):

```
python3 parmoo_cfr_structured_solver.py [--iseed M]
```

"""

import numpy as np
import csv
import sys
from time import time_ns
from parmoo.moop import MOOP
from parmoo.optimizers import LBFGSB
from parmoo.surrogates import GaussRBF
from parmoo.searches import LatinHypercube
from parmoo.acquisitions import RandomConstraint
from parmoo.objectives import single_sim_out
import cfr_model

# Set the problem dimensions
n = 5
m = 2
o = 3

# Read the random seed from the command line
iseed = time_ns() % (2 ** 32) # default is to set from system clock
for i, opt in enumerate(sys.argv[1:]):
    if opt == "--iseed":
        try:
            iseed = int(sys.argv[i+2])
        except IndexError:
            raise ValueError("iseed requires an integer value")
        except ValueError:
            raise ValueError("iseed requires an integer value")

# Create a MOOP object
cfr_moop = MOOP(LBFGSB, hyperparams={})
# Add 3 continuous design variables from cfr_model meta data
for i in range(3):
    cfr_moop.addDesign({'name': cfr_model.DES_NAMES[i],
                        'lb': cfr_model.lb[i],
                        'ub': cfr_model.ub[i],
                        'des_tol': 1.0e-2})
# Add 2 categorical variables with 2 levels each
# (ParMOO will automatically embed these into a 2*2-1 = 3-dimensional
# latent space. So, combined with the 3 continuous variables, the
# effective dimension of the problem will be 6.
cfr_moop.addDesign({'name': "Solvent",
                    'des_type': "categorical",
                    'levels': ["S1", "S2"]})
cfr_moop.addDesign({'name': "Base",
                    'des_type': "categorical",
                    'levels': ["B1", "B2"]})
# Add 1 simulation
cfr_moop.addSimulation({'name': "CFR out",
                        'm': m,
                        'sim_func': cfr_model.cfr_sim_model,
                        'hyperparams': {'search_budget': 50},
                        'search': LatinHypercube,
                        'surrogate': GaussRBF})
# Add 3 objectives
cfr_moop.addObjective({'name': "-TFMC product",
                       'obj_func': single_sim_out(cfr_moop.getDesignType(),
                                                  cfr_moop.getSimulationType(),
                                                  ("CFR out", 0),
                                                  goal="max")})
cfr_moop.addObjective({'name': "unwanted byproduct",
                       'obj_func': single_sim_out(cfr_moop.getDesignType(),
                                                  cfr_moop.getSimulationType(),
                                                  ("CFR out", 1))})
cfr_moop.addObjective({'name': "reaction time",
                       'obj_func': cfr_model.reaction_time})
# Add 3 random acquisition functions
for i in range(3):
    cfr_moop.addAcquisition({'acquisition': RandomConstraint,
                             'hyperparams': {}})
# Fix the random seed for reproducability
np.random.seed(iseed)
# Solve the Fayans EDF callibration moop with 30 iterations
# (30 * 3 + 50 = 140 sim budget)
cfr_moop.solve(30)
full_data = cfr_moop.getObjectiveData()

# Dump full data set to a CSV file
with open("cfr_structured_results_seed_" + str(iseed) + ".csv", "w") as fp:
    csv_writer = csv.writer(fp, delimiter=",")
    # Define the header
    header = cfr_model.DES_NAMES.copy()
    header.append("-TFMC product")
    header.append("unwanted byproduct")
    header.append("reaction time")
    # Dump header to first row
    csv_writer.writerow(header)
    # Add each data point as another row
    for xs in full_data:
        csv_writer.writerow([xs[name] for name in header])
