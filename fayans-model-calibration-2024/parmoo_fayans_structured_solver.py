""" Apply ParMOO with libE plugin to solve the Fayans EDF callibration MOOP,
exploiting sum-of-squares structure in how simulation outputs are used.

Execute via one of the following commands (where N is the number of threads,
and the optional int M in [0, 2^32-1] is a random seed):

```
mpiexec -np N python3 parmoo_fayans_structured_solver.py [--iseed M]
python3 parmoo_fayans_structured_solver.py --nworkers N --comms local [--iseed M]
python3 parmoo_fayans_structured_solver.py --nworkers N --comms tcp [--iseed M]
```


The number of concurrent evaluations of the Fayans EDF model will be N-1,
since the first thread is reserved for the ParMOO generator.

"""

import numpy as np
import csv
import sys
from time import time_ns
from parmoo.extras.libe import libE_MOOP
from parmoo.optimizers import LocalSurrogate_BFGS
from parmoo.surrogates import GaussRBF
from parmoo.searches import LatinHypercube
from parmoo.acquisitions import RandomConstraint, FixedWeights
import fayans_model as fm

# Set the problem dimensions
n = 13
m = 198
o = 3

# Read the random seed from the command line
iseed = time_ns() % (2 ** 32) # default is to set from system clock
for i, opt in enumerate(sys.argv[1:]):
    if opt == "--iseed":
        try:
            iseed = int(sys.argv[i+2])
            del sys.argv[i+1]
            del sys.argv[i+1]
        except IndexError:
            raise ValueError("iseed requires an integer value")
        except ValueError:
            raise ValueError("iseed requires an integer value")

if __name__ == "__main__":
    """ For a libE_MOOP to be run on certain OS (such as MacOS) it must be
    enclosed within an ``if __name__ == '__main__'`` clause. """

    # Create a libE_MOOP
    fayans_moop = libE_MOOP(LocalSurrogate_BFGS,
                            hyperparams={'np_random_gen': iseed})
    # Add 13 design variables from fayans_model meta data
    for i in range(n):
        fayans_moop.addDesign({'name': fm.DES_NAMES[i],
                               'lb': fm.lb[i],
                               'ub': fm.ub[i],
                               })
    # Add 1 simulation
    fayans_moop.addSimulation({'name': "residuals",
                               'm': m,
                               'sim_func': fm.fayans_model_sim,
                               'hyperparams': {'search_budget': 2000},
                               'search': LatinHypercube,
                               'surrogate': GaussRBF,
                               })
    # Add 3 objectives
    fayans_moop.addObjective({'name': "binding energy",
                              'obj_func': fm.binding_energy_obj_func,
                              'obj_grad': fm.binding_energy_obj_grad,
                              })
    fayans_moop.addObjective({'name': "std radii",
                              'obj_func': fm.std_radii_obj_func,
                              'obj_grad': fm.std_radii_obj_grad,
                              })
    fayans_moop.addObjective({'name': "other quantities",
                              'obj_func': fm.other_quantities_obj_func,
                              'obj_grad': fm.other_quantities_obj_grad,
                              })
    # Add 3 constraints
    fayans_moop.addConstraint({'name': "binding energy max",
                               'con_func': fm.binding_energy_con_func,
                               'con_grad': fm.binding_energy_con_grad,
                               })
    fayans_moop.addConstraint({'name': "std radii max",
                               'con_func': fm.std_radii_con_func,
                               'con_grad': fm.std_radii_con_grad,
                               })
    fayans_moop.addConstraint({'name': "other quantities max",
                               'con_func': fm.other_quantities_con_func,
                               'con_grad': fm.other_quantities_con_grad,
                               })
    # Add 9 random acquisition functions
    for i in range(9):
        fayans_moop.addAcquisition({'acquisition': RandomConstraint,
                                    'hyperparams': {}})
    # Add a 10th acquisition function, which is the Chi^2 loss
    fayans_moop.addAcquisition({'acquisition': FixedWeights,
                                'hyperparams': {'weights': np.ones(o) / o}})
    # Solve the Fayans EDF callibration moop with a 10K sim budget
    fayans_moop.solve(sim_max=10000, wt_max=172800)
    full_data = fayans_moop.getObjectiveData()
    soln = fayans_moop.getPF()

    # Dump full data set to a CSV file
    with open(f"fayans_structured_results_seed_{iseed}.csv", "w") as fp:
        csv_writer = csv.writer(fp, delimiter=",")
        # Define the header
        header = fm.DES_NAMES.copy()
        header.append("binding energy")
        header.append("std radii")
        header.append("other quantities")
        # Dump header to first row
        csv_writer.writerow(header)
        # Add each data point as another row
        for xs in full_data:
            csv_writer.writerow([xs[name] for name in header])
