""" Evaluate a generalized least-squares model (trained with sklearn).

Defines the DES_NAMES environment variable (list of design variable names),
define the lower and upper bounds, the nonlinear transformation kernels, and
regression weights,, and defines the following module functions that are ready
for usage by ParMOO:
 - the ParMOO compatible simulation function cfr_sim_model(x)
 - the ParMOO compatible simulation function cfr_blackbox_model(x)
 - the ParMOO compatible objective function reaction_time(x, s, der=0)

"""

import numpy as np

# List of expected design variable names
DES_NAMES = ["temp", "RT", "EQR", "Solvent", "Base"]

# Problem dims
lb = np.array([ 35.0,  60.0, 0.8, 0.0, 0.0])
ub = np.array([150.0, 300.0, 1.5, 1.0, 1.0])

# Init shifts and scales
__shifts__ = np.array([124.,  0., 1., 0., 0.])
__scales__ = np.array([100., 20., 1., 1., 1.])

# Pre-calculated feature weights for our model
__A1__ = np.array([7.49107726, -2.58646041, -1.03283051,
                   0.79047118, -4.41711488, -3.43695858])
__A2__ = np.array([4.53028992,  2.64319176,  1.14540860,
                   -0.36471219,  3.19852515,  3.90731791])
__b1__ = -0.7862417771324974
__b2__ = -2.341494492573002

# Define functions

def __feature_map__(x):
    """ Embed x, rescale the embedding, and map through a feature kernel.

    Args:
        x (numpy structured array): Containins the following fiels:
         - "temp" (float): in the range 35 to 150 
         - "RT" (float): in the range 60 to 300
         - "EQR" (float): in the range 0.8 to 1.5
         - "Solvent" (str-compatible): either "S1" or "S2"
         - "Base" (str-compatible): either "B1" or "B2"

    Returns:
        numpy.ndarray: the feature-map embedding of the problem

    """

    # Initialize an embedding vector
    xx = np.zeros(5)
    # Rescale the float inputs
    for i in range(3):
        xx[i] = (x[DES_NAMES[i]] - __shifts__[i]) / __scales__[i]
    # One-hot encoding for 2 solvents
    if str(x[DES_NAMES[3]]) == "S1":
        xx[3] = (0 - __shifts__[3]) / __scales__[3]
    elif str(x[DES_NAMES[3]]) == "S2":
        xx[3] = (1 - __shifts__[3]) / __scales__[3]
    else:
        raise ValueError(f"x[{DES_NAMES[3]}] contains an illegal value")
    # One-hot encoding for 2 bases
    if str(x[DES_NAMES[4]]) == "B1":
        xx[4] = (0 - __shifts__[4]) / __scales__[4]
    elif str(x[DES_NAMES[4]]) == "B2":
        xx[4] = (1 - __shifts__[4]) / __scales__[4]
    else:
        raise ValueError(f"x[{DES_NAMES[4]}] contains an illegal value")
    # Return a feature map embedding
    return np.array([1.0, xx[0] ** 2.0,
                     1.0 - xx[3], (1.0 - xx[3]) * xx[0],
                     xx[4], xx[4] * xx[0]]) * ((1.0 - np.exp(-xx[1])) *
                                               (1.0 - xx[2] ** 2.0))

def tfmc_int_model(x):
    """ Predict the TFMC integral for the reaction defined by input x.

    Args:
        x (numpy structured array): Containins the following fiels:
         - "temp" (float): in the range 35 to 150 
         - "RT" (float): in the range 60 to 300
         - "EQR" (float): in the range 0.8 to 1.5
         - "Solvent" (str-compatible): either "S1" or "S2"
         - "Base" (str-compatible): either "B1" or "B2"

    Returns:
        float: the predicted value of the TFMC integral, based
        on our nonlinear feature model

    """

    xf = __feature_map__(x)
    return np.dot(xf, __A1__) + __b1__

def byprod_int_model(x):
    """ Predict the byproduct integral for the reaction defined by input x.

    Args:
        x (numpy structured array): Containins the following fiels:
         - "temp" (float): in the range 35 to 150 
         - "RT" (float): in the range 60 to 300
         - "EQR" (float): in the range 0.8 to 1.5
         - "Solvent" (str-compatible): either "S1" or "S2"
         - "Base" (str-compatible): either "B1" or "B2"

    Returns:
        float: the predicted value of the byproduct integral, based
        on our nonlinear feature model

    """

    xf = __feature_map__(x)
    return np.dot(xf, __A2__) + __b2__

def cfr_sim_model(x):
    """ Predict both TFMC and byproduct integrals, mimicing the CFR output.

    Args:
        x (numpy structured array): Containins the following fiels:
         - "temp" (float): in the range 35 to 150 
         - "RT" (float): in the range 60 to 300
         - "EQR" (float): in the range 0.8 to 1.5
         - "Solvent" (str-compatible): either "S1" or "S2"
         - "Base" (str-compatible): either "B1" or "B2"

    Returns:
        numpy.ndarray: a float array containing 2 values, corresponding to
        our nonlinear feature model's predictions for 2 quantities of
        interest:
         - result[0] contains the predicted value of the TFMC integral
         - result[1] contains the predicted value of the byproduct integral

    """

    return np.array([tfmc_int_model(x), byprod_int_model(x)])

def cfr_blackbox_model(x):
    """ Predict both TFMC and byproduct integrals, and reaction time,
    mimicing a true blackbox objective.

    Args:
        x (numpy structured array): Containins the following fiels:
         - "temp" (float): in the range 35 to 150 
         - "RT" (float): in the range 60 to 300
         - "EQR" (float): in the range 0.8 to 1.5
         - "Solvent" (str-compatible): either "S1" or "S2"
         - "Base" (str-compatible): either "B1" or "B2"

    Returns:
        numpy.ndarray: a float array containing 2 values, corresponding to
        our nonlinear feature model's predictions for 2 quantities of
        interest:
         - result[0] contains the predicted value of the TFMC integral
         - result[1] contains the predicted value of the byproduct integral

    """

    return np.array([tfmc_int_model(x), byprod_int_model(x), x["RT"]])

def reaction_time(x, s, der=0):
    """ A ParMOO objective that passes the reaction time as an objective.

    Args:
        x (numpy structured array): contains 5 fields (defined in DES_NAMES)

        s (numpy structured array): not used here

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        result = np.zeros(1, dtype=x.dtype)[0]
        result["RT"] = 1.0
        return result
    elif der == 2:
        return np.zeros(1, dtype=s.dtype)[0]
    else:
        return x["RT"]


if __name__ == "__main__":
    # Check that the following test point is closely approximated
    des_type = [("temp", float), ("RT", float), ("EQR", float),
                ("Solvent", "U2"), ("Base", "U2")]
    x1 = np.zeros(1, dtype=des_type)[0]
    x1["temp"] = 150.
    x1["RT"] = 300.
    x1["EQR"] = 1.
    x1["Solvent"] = "S1"
    x1["Base"] = "B1"
    assert(np.abs(np.sum(cfr_sim_model(x1)) - 9.12) < 1.0e-1)
