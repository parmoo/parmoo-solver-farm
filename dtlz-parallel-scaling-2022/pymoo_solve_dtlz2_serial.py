import csv
import numpy as np
import os
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.problems import get_problem
import sys

# Set default problem parameters
BB_BUDGET = 1000 # 10K eval budget
POP_SIZE = 50 # Population size
NDIMS = 10 # 10 vars
NOBJS = 3  # 3 objs

if __name__ == "__main__":

    # Set the random seed from CL or system clock
    if len(sys.argv) > 1:
        SEED = int(sys.argv[1])
    else:
        from datetime import datetime
        SEED = int(datetime.now().timestamp())
    FILENAME = f"pymoo-dtlz2/results_seed{SEED}.csv"
    
    # Solve DTLZ2 problem w/ NSGA-II (pop size 50) in pymoo
    problem = get_problem(f"dtlz2", n_var=NDIMS, n_obj=NOBJS)
    algorithm = NSGA2(pop_size=POP_SIZE)
    res = minimize(problem,
                   algorithm,
                   ("n_gen", BB_BUDGET // POP_SIZE),
                   save_history=True,
                   seed=SEED,
                   verbose=False)

    ### Get solution and calculate hypervolumes ###
    from pymoo.indicators.hv import HV
    pts = []
    for i, row in enumerate(res.history):
        for fi in row.result().F:
            pts.append(fi)
    hv = HV(ref_point=np.ones(3))
    hypervol = hv(np.asarray(pts))

    ### Dump the results to a file ###
    FILENAME = f"pymoo-dtlz2/hv.csv"
    with open(FILENAME, 'a') as fp:
        fp.write(f"{hypervol:.2f}\n")
