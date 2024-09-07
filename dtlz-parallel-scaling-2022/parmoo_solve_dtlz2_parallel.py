
import numpy as np
import pandas as pd
from parmoo.acquisitions import RandomConstraint, FixedWeights
from parmoo.extras.libe import libE_MOOP
from parmoo.objectives.obj_lib import single_sim_out
from parmoo.optimizers import TR_LBFGSB
from parmoo.searches import LatinHypercube
from parmoo.surrogates import LocalGaussRBF
import sys
import time

### Problem dimensions ###
NUM_DES = 10
NUM_OBJ = 3

### Define the simulation evaluation

def sim_func(x):
    """ Evaluate DTLZ2 as a 1 second simulation in ParMOO """

    xx = np.asarray([x[f"x{i+1}"] for i in range(NUM_DES)])
    g2 = np.sum((xx[NUM_OBJ-1:NUM_DES] - 0.5) ** 2)
    fx = np.zeros(NUM_OBJ)
    fx[:] = (1.0 + g2)
    for i in range(NUM_OBJ):
        for j in range(NUM_OBJ - 1 - i):
            fx[i] *= np.cos(np.pi * xx[j] / 2)
        if i > 0:
            fx[i] *= np.sin(np.pi * xx[NUM_OBJ - 1 - i] / 2)
    t1 = 1. + (2. * np.random.sample()) # 2 seconds +/- 1
    time.sleep(t1)  # Add a ~2 second pause to simulation evaluation
    return fx


if __name__ == "__main__":

    ### Set the batch size and random seed from command line, if present ###
    if len(sys.argv) > 1:
        SIZE = int(sys.argv[1])
    else:
        SIZE = 8
    if len(sys.argv) > 2:
        SEED = int(sys.argv[2])
    else:
        from datetime import datetime
        SEED = int(datetime.now().timestamp())
    np.random.seed(SEED)
    
    ### Budget variables -- total budget is 1000 sims ###
    n_per_batch = SIZE                         # batch size
    iters_limit = int(800 / SIZE)              # run approx 800 sims iteration
    n_search_sz = 1000 - (SIZE * iters_limit)  # approx 200 pt DOE
    
    ### Setup the problem following the ParMOO docs recommendation
    #   For more details, see:
    #      https://parmoo.readthedocs.io/en/latest/tutorials/local_method.html
    ###
    moop_tr = libE_MOOP(TR_LBFGSB)
    for i in range(NUM_DES):
        moop_tr.addDesign({'name': f"x{i+1}",
                           'des_type': "continuous",
                           'lb': 0.0, 'ub': 1.0})
    moop_tr.addSimulation({'name': "DTLZ_out",
                           'm': NUM_OBJ,
                           'sim_func': sim_func,
                           'search': LatinHypercube,
                           'surrogate': LocalGaussRBF,
                           'hyperparams': {'search_budget': n_search_sz}})
    for i in range(NUM_OBJ):
        moop_tr.addObjective({'name': f"f{i+1}",
                              'obj_func': single_sim_out(moop_tr.getDesignType(),
                                                         moop_tr.getSimulationType(),
                                                         ("DTLZ_out", i))})
    for i in range(n_per_batch - NUM_OBJ - 1):
       moop_tr.addAcquisition({'acquisition': RandomConstraint,
                               'hyperparams': {}})
    for i in range(NUM_OBJ):
       moop_tr.addAcquisition({'acquisition': FixedWeights,
                               'hyperparams': {'weights': np.eye(NUM_OBJ)[i]}})
    moop_tr.addAcquisition({'acquisition': FixedWeights,
                            'hyperparams': {'weights': np.ones(NUM_OBJ) / NUM_OBJ}})
    
    ### Time the solve ###
    tick = time.time()
    moop_tr.solve(sim_max=1000)
    tock = time.time()
    
    ### Get solution and calculate hypervolumes ###
    from pymoo.indicators.hv import HV
    pts = moop_tr.getPF()
    pts1 = np.zeros((pts.shape[0], 3))
    pts1[:, 0] = pts["f1"]
    pts1[:, 1] = pts["f2"]
    pts1[:, 2] = pts["f3"]
    hv = HV(ref_point=np.ones(3))
    hypervol = hv(pts1)

    ### Dump the results to a file ###
    FILENAME = f"parmoo-dtlz2/size{SIZE}_seed{SEED}.csv"
    with open(FILENAME, 'a') as fp:
        fp.write(f"{tock - tick:.2f},{hypervol:.2f}\n")
