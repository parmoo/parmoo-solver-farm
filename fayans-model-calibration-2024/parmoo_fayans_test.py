""" A quick test-run for the Fayans problem, to make sure that
ParMOO, libEnsemble, and tensorflow are all running correctly.

Execute via one of the following commands (where N is the number of threads):

```
mpiexec -np N python3 parmoo_fayans_test.py
python3 parmoo_fayans_test.py --nworkers N --comms local
python3 parmoo_fayans_test.py --nworkers N --comms tcp
```


The number of concurrent evaluations of the Fayans EDF model will be N-1,
since the first thread is reserved for the ParMOO generator.

"""

import numpy as np
import csv
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

# Fix the random seed to 0 for the tests
iseed = 0

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
                               'ub': fm.ub[i]})
    # Add 1 simulation
    fayans_moop.addSimulation({'name': "residuals",
                               'm': m,
                               'sim_func': fm.fayans_model_sim,
                               'hyperparams': {'search_budget': 100},
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
    # Solve the Fayans EDF callibration moop with a 200 sim / 10 min budget
    fayans_moop.solve(sim_max=200, wt_max=600)
    full_data = fayans_moop.getObjectiveData()
    soln = fayans_moop.getPF()

    # Dump full data set to a CSV file
    with open("fayans_test_results.csv", "w") as fp:
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

    # Plot results
    from parmoo.viz import scatter
    # Use output="dash" to start an interactive dashboard in browser
    scatter(fayans_moop, output="png")
