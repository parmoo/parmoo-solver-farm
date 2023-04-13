""" Load a pre-trained (in keras) model in pytorch for approximating the
residual function for the Fayans EDF model, based on a 13-dimensional input
vector.

Defines the DES_NAMES environment variable (list of design variable names),
loads the rescaled model that predicts the residuals for a 13-dim
design x, and defines the following module functions that are ready for
usage by ParMOO:
 - the ParMOO compatible simulation function fayans_sim_model(x)
 - the ParMOO compatible simulation function fayans_blackbox_model(x)
 - the ParMOO compatible objective function binding_energy(x, s, der=0)
 - the ParMOO compatible objective function charge_radius(x, s, der=0)
 - the ParMOO compatible objective function diffraction_radius(x, s, der=0)
 - the ParMOO compatible objective function std_radii(x, s, der=0)
 - the ParMOO compatible objective function surface_thickness(x, s, der=0)
 - the ParMOO compatible objective function neutron_single_level(x, s, der=0)
 - the ParMOO compatible objective function proton_single_level(x, s, der=0)
 - the ParMOO compatible objective function isotopic_shifts_radii(x, s, der=0)
 - the ParMOO compatible objective function neutron_pairing_gap(x, s, der=0)
 - the ParMOO compatible objective function proton_pairing_gap(x, s, der=0)
 - the ParMOO compatible objective function other_quantities(x, s, der=0)
 - the ParMOO compatible constraint constraint_binding_energy(x, s, der=0)
 - the ParMOO compatible constraint constraint_std_radii(x, s, der=0)
 - the ParMOO compatible constraint constraint_other_quantities(x, s, der=0)

"""

import numpy as np
import torch
import torch.nn as nn

# Define the parmoo input keys for later use
DES_NAMES = ["rho eq", "E/A", "K", "J", "L", "h v 2-", "a s +",
             "h s nabla", "kappa", "kappa'", "f xi ex", "h xi +",
             "h xi nabla"]

# Define input bounds and scale/shift terms
lb = np.array([0.146, -16.21, 137.2, 19.5, 2.20, 0.0,
               0.418, 0.0, 0.076, -0.892, -4.62, 3.94,
               -0.96])
ub = np.array([0.167, -15.50, 234.4, 37.0, 69.6, 100.0,
               0.706, 0.516, 0.216, 0.982, -4.38, 4.27,
               3.66])
__shifts__ = lb.copy()
__scales__ = np.zeros(13)
__scales__ = ub - lb

# Largest value allowed
npmax = np.sqrt(np.finfo(np.float64).max) / 200.0

# Set the torch dtype
torch.set_default_dtype(torch.float64)

# Define the FayansNet PyTorch class
class FayansNet(nn.Module):
    """ PyTorch class for the Fayans EDF residual model's network arch """

    def __init__(self):
        """ Constructor """

        super(FayansNet, self).__init__()
        self.l_in  = nn.Linear(13, 256)
        self.l_h1  = nn.Linear(256, 256)
        self.l_h2  = nn.Linear(256, 256)
        self.l_out = nn.Linear(256, 198)

    def forward(self, x):
        """ Forward pass

        Args:
            x (np.ndarray): 13-dimensional input normalized to range [0,1]

        Returns:
            np.ndarray: 198-dimensional output normalized to range [0, 1]

        """

        x = torch.tanh(self.l_in(x))
        x = torch.tanh(self.l_h1(x))
        x = torch.tanh(self.l_h2(x))
        x = torch.tanh(self.l_out(x))
        return x

# Reload the trained weights and biases from a pytorch save file
fayans_trained_model = FayansNet()
fayans_trained_model.load_state_dict(torch.load("fayans_weights.pt"))
fayans_trained_model.double()
fayans_trained_model.eval();

def __descale__(fi):
    """ De-scale outputs from the model by arctanh of a custom transform

    Args:
        fi (numpy:ndarray): An array of 198 floats in the range (-1, 1),
            which have been predicted by the model.

    Returns:
        numpy.ndarray: An array of 198 floats.
         - if |arctanh(fi_j)| <= 1: fi_j' = arctanh(fi_j)
         - if 1 < |arctanh(fi_j)| <= ~10^5: arctanh(10^{|fi_j|-1})
         - if ~10^5 < |fi_j|: arctanh(10^{|fi_j|-1})

    """

    fi_new = fi.copy()
    fi_new = np.arctanh(fi_new)
    for j in range(fi_new.size):
        if np.abs(fi_new[j]) > 5.0:
            fi_new[j] = (np.sign(fi_new[j]) *
                         (10.0 ** (np.abs(fi_new[j]) - 5.0) + 4.0))
        if np.abs(fi_new[j]) > 1.0:
            fi_new[j] = (np.sign(fi_new[j]) *
                         (10.0 ** (np.abs(fi_new[j]) - 1.0)))
    return fi_new

def fayans_sim_model(x):
    """ Predict the fit residuals using a pre-trained model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

    Returns:
        numpy ndarray: contains 198 float-valued residuals, based
        on model's predictions

    """

    # Embed parmoo named-inputs into torch-compatible tensor in [0, 1]
    xx = torch.zeros(1, 13)
    for i, namei in enumerate(DES_NAMES):
        xx[0, i] = (x[namei] - __shifts__[i]) / __scales__[i]
    # Predict with the loaded torch model and descale output
    with torch.no_grad():
        y = fayans_trained_model(xx).detach().cpu().numpy().flatten()
    # Descale the prediction
    yy = __descale__(y)
    for i in range(len(yy)):
        yy[i] = min(yy[i], npmax)
        yy[i] = max(yy[i], -npmax)
    return yy

def fayans_blackbox_model(x):
    """ A blackbox simulation outputting sum-of-squared simulation residuals.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

    Returns:
        numpy ndarray: contains 3 float-valued squared-residuals, based
        on the model's predictions composed with objectives.

    """

    # Predict simulation output
    s = np.zeros(1, dtype=[("residuals", "f8", 198)])[0]
    s["residuals"][:] = fayans_sim_model(x)
    # Calculate the 3 output fields using objectives from below
    s_out = np.zeros(3)
    s_out[0] = binding_energy(x, s)
    s_out[1] = charge_radius(x, s) + diffraction_radius(x, s)
    s_out[2] = (surface_thickness(x, s) + neutron_single_level(x, s) +
                proton_single_level(x, s) + isotopic_shifts_radii(x, s) +
                neutron_pairing_gap(x, s) + proton_pairing_gap(x, s))
    return s_out

def binding_energy(x, s, der=0):
    """ A ParMOO objective that calculates the sum-of-squared residuals in the
    binding energy class based on the output of the fayans_sim_model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["residuals"][0:63] = 2.0 * s["residuals"][0:63]
        return res
    else:
        return np.dot(s["residuals"][0:63], s["residuals"][0:63])

def charge_radius(x, s, der=0):
    """ A ParMOO objective that calculates the sum-of-squared residuals in the
    charge radius class based on the output of the fayans_sim_model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["residuals"][117:169] = 2.0 * s["residuals"][117:169]
        return res
    else:
        return np.dot(s["residuals"][117:169], s["residuals"][117:169])

def diffraction_radius(x, s, der=0):
    """ A ParMOO objective that calculates the sum-of-squared residuals in the
    charge radius class based on the output of the fayans_sim_model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["residuals"][63:91] = 2.0 * s["residuals"][63:91]
        return res
    else:
        return np.dot(s["residuals"][63:91], s["residuals"][63:91])

def std_radii(x, s, der=0):
    """ A ParMOO objective that calculates the sum-of-squared residuals in the
    standard radii class based on the output of the fayans_sim_model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 0:
        return charge_radius(x, s) + diffraction_radius(x, s)
    else:
        r1 = charge_radius(x, s, der=der)
        r2 = diffraction_radius(x, s, der=der)
        result = np.zeros(1, dtype=r1.dtype)[0]
        for name in r1.dtype.names:
            result[name] = r1[name] + r2[name]
    return result

def surface_thickness(x, s, der=0):
    """ A ParMOO objective that calculates the sum-of-squared residuals in the
    surface thickness class based on the output of the fayans_sim_model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["residuals"][91:117] = 2.0 * s["residuals"][91:117]
        return res
    else:
        return np.dot(s["residuals"][91:117], s["residuals"][91:117])

def neutron_single_level(x, s, der=0):
    """ A ParMOO objective that calculates the sum-of-squared residuals in the
    neutron single level class based on the output of the fayans_sim_model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["residuals"][174:179] = 2.0 * s["residuals"][174:179]
        return res
    else:
        return np.dot(s["residuals"][174:179], s["residuals"][174:179])

def proton_single_level(x, s, der=0):
    """ A ParMOO objective that calculates the sum-of-squared residuals in the
    proton single level class based on the output of the fayans_sim_model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["residuals"][169:174] = 2.0 * s["residuals"][169:174]
        return res
    else:
        return np.dot(s["residuals"][169:174], s["residuals"][169:174])

def isotopic_shifts_radii(x, s, der=0):
    """ A ParMOO objective that calculates the sum-of-squared residuals in the
    isotopic shifts radii class based on the output of the fayans_sim_model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["residuals"][179:182] = 2.0 * s["residuals"][179:182]
        return res
    else:
        return np.dot(s["residuals"][179:182], s["residuals"][179:182])

def neutron_pairing_gap(x, s, der=0):
    """ A ParMOO objective that calculates the sum-of-squared residuals in the
    neutron pairing gap class based on the output of the fayans_sim_model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["residuals"][182:187] = 2.0 * s["residuals"][182:187]
        return res
    else:
        return np.dot(s["residuals"][182:187], s["residuals"][182:187])

def proton_pairing_gap(x, s, der=0):
    """ A ParMOO objective that calculates the sum-of-squared residuals in the
    proton pairing gap class based on the output of the fayans_sim_model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["residuals"][187:198] = 2.0 * s["residuals"][187:198]
        return res
    else:
        return np.dot(s["residuals"][187:198], s["residuals"][187:198])

def other_quantities(x, s, der=0):
    """ A ParMOO objective that calculates the sum-of-squared residuals across
    all nonstandard observation classes (surface thickness, neutron single level,
    proton single level, isotopic shifts radii, neutron pairing gap, and proton
    pairing gap) based on the output of the fayans_sim_model.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate f(x, s)),
             - 1 (calculated df/dx), or
             - 2 (calculate df/ds)

    Returns:
        float: the objective value to be minimized by ParMOO

    """

    if der == 0:
        return (surface_thickness(x, s, der=der)
                + neutron_single_level(x, s, der=der)
                + proton_single_level(x, s, der=der)
                + isotopic_shifts_radii(x, s, der=der)
                + neutron_pairing_gap(x, s, der=der)
                + proton_pairing_gap(x, s, der=der))
    else:
        r1 = surface_thickness(x, s, der=der)
        r2 = neutron_single_level(x, s, der=der)
        r3 = proton_single_level(x, s, der=der)
        r4 = isotopic_shifts_radii(x, s, der=der)
        r5 = neutron_pairing_gap(x, s, der=der)
        r6 = proton_pairing_gap(x, s, der=der)
        result = np.zeros(1, dtype=r1.dtype)[0]
        for name in r1.dtype.names:
            result[name] = (r1[name] + r2[name] + r3[name]
                            + r4[name] + r5[name] + r6[name])
        return result

def constraint_binding_energy(x, s, der=0):
    """ A ParMOO constraint that bounds the sum-of-squared residuals for
    binding energy below 804.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate c(x, s)),
             - 1 (calculated dc/dx), or
             - 2 (calculate dc/ds)

    Returns:
        float: the constraint violation score, to be used by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        res = np.zeros(1, dtype=s.dtype)[0]
        res["residuals"][0:63] = 2.0 * s["residuals"][0:63]
        return res
    else:
        return np.dot(s["residuals"][0:63], s["residuals"][0:63]) - 804

def constraint_std_radii(x, s, der=0):
    """ A ParMOO constraint that bounds the sum-of-squared residuals for std
    radii below 2090.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate c(x, s)),
             - 1 (calculated dc/dx), or
             - 2 (calculate dc/ds)

    Returns:
        float: the constraint violation score, to be used by ParMOO

    """

    if der == 1 or der == 2:
        return std_radii(x, s, der=der)
    else:
        return std_radii(x, s) - 2090

def constraint_other_quantities(x, s, der=0):
    """ A ParMOO constraint that bounds the sum-of-squared residuals for other
    quantities below 613.

    Args:
        x (numpy structured array): contains 13 float-valued fields,
            defined in DES_NAMES

        s (numpy structured array): contains a single field "residuals"
            with 198 float-valued fields standard residuals, as output by the
            fayans_sim_model function

        der (int, optional): defaults to 0, may take one of three values:
             - 0 (evaluate c(x, s)),
             - 1 (calculated dc/dx), or
             - 2 (calculate dc/ds)

    Returns:
        float: the constraint violation score, to be used by ParMOO

    """

    if der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        return other_quantities(x, s, der=2)
    else:
        return other_quantities(x, s) - 613


if __name__ == "__main__":
    """ Driver code: checks if the output is reasonable for a dummy input """

    # Rescale function is only used by driver code
    def __rescale__(fi):
        """ Re-scale Fayans EDF residuals to [0, 1] via custom transformation
    
        Args:
            fi (numpy:ndarray): An array of 198 Fayans EDF model residuals.
    
        Returns:
            numpy.ndarray: An array of 198 floats in range (-1, 1), i.e.,
            at the scale that the original keras model was trained.
    
        """
    
        fi_new = fi.copy()
        for j in range(fi_new.size):
            if np.abs(fi_new[j]) > 1.0:
                fi_new[j] = np.sign(fi_new[j]) * \
                            (np.log(np.abs(fi_new[j])) / np.log(10) + 1.0)
            if np.abs(fi_new[j]) > 5.0:
                fi_new[j] = np.sign(fi_new[j]) * \
                            (np.log(np.abs(fi_new[j]) - 4.0) / np.log(10.0)
                             + 5.0)
        return np.tanh(fi_new[:])

    # Create a dummy input/output pair
    xin = [0.57361597, 0.57859087, 0.5455427, 0.48610193, 0.33418274,
           0.49567327, 0.20770496, 0.5654273, 0.44474036, 0.53015137,
           0.46080855, 0.47403628, 0.57264185]
    xout = [ 4.76958096e-01, -1.87459394e-01,  3.99756044e-01,  5.84251404e-01,
             1.74756214e-01,  2.50667721e-01,  6.65053785e-01,  2.80404687e+00,
             2.61621737e+00,  2.04316163e+00,  5.22532463e-01,  4.30171251e-01,
             3.61464292e-01,  6.19013309e-01,  6.56691492e-01,  3.15324426e+00,
             7.54071951e-01,  1.15164757e+00,  1.28436780e+00,  2.10099673e+00,
             2.41746473e+00,  5.62005377e+00,  5.73144102e+00,  5.89360094e+00,
             5.51160276e-01,  5.82801163e-01,  6.03691399e-01,  2.92197061e+00,
             5.80027819e+00,  5.47740078e+00,  5.29456568e+00,  5.07482386e+00,
             4.93640041e+00,  6.06802821e-01,  7.18631387e-01,  8.72136891e-01,
             3.00361998e-02,  1.45457006e+00,  7.11130723e-02,  7.27187544e-02,
             2.34882641e+00,  4.01521206e+00,  2.44715309e+00,  1.81306529e+00,
             7.85663247e-01,  7.16356456e-01,  5.91596603e-01,  5.98083591e+00,
             6.43317223e+00,  5.90593624e+00,  5.38159513e+00,  2.34887099e+00,
             2.20275068e+00,  2.08535576e+00,  1.67803192e+00,  1.41655302e+00,
             1.33055949e+00,  8.25271797e+00,  5.40900612e+00,  5.22932529e+00,
             5.02158928e+00,  4.94617701e+00,  4.82397223e+00,  7.50368536e-01,
             5.86946607e-01,  1.25893831e+00,  1.25157201e+00,  3.99331361e-01,
            -4.47224602e-02, -6.01591468e-01,  4.43912625e-01,  9.14739490e-01,
             5.11332393e-01,  5.08704126e-01,  4.86678690e-01,  6.86461747e-01,
             1.45417833e+00,  1.07003021e+00,  1.26647854e+00,  5.12988091e-01,
             2.37447664e-01,  4.99279425e-02,  3.74263167e-01,  1.10319400e+00,
             3.92844081e-01,  2.02812970e-01,  4.39007245e-02, -1.64755091e-01,
            -1.20150298e-01,  2.55761594e-01,  8.58146071e-01,  1.41754463e-01,
            -2.67316341e-01, -5.95661640e-01, -2.17244670e-01,  7.10995615e-01,
            -1.05445698e-01, -1.45911217e-01, -4.80364971e-02, -1.73052903e-02,
             1.62952729e-02,  2.82324962e-02,  2.23663718e-01,  1.00715899e+00,
             4.32874650e-01,  4.21711296e-01,  1.81782454e-01,  2.95597315e-01,
            -9.91614312e-02, -4.22825217e-01, -1.17938623e-01, -9.10019875e-02,
             6.30317926e-01,  4.27730411e-01,  6.25361145e-01,  2.36729622e-01,
            -4.65673476e-01,  1.75203216e+00,  3.03319305e-01,  8.13974515e-02,
             5.61677292e-02,  9.83059406e-01,  2.13744223e-01,  1.23274988e-02,
            -2.93462034e-02, -1.85785308e-01, -1.29350200e-01, -1.97796106e-01,
            -5.02220213e-01, -1.72309279e-01,  4.39882204e-02,  8.24081805e-03,
             1.10672690e-01,  1.50674805e-01,  1.60809547e-01,  1.43889022e+00,
             1.47543204e+00,  1.48623359e+00,  1.39297307e+00,  9.33225214e-01,
             1.60669553e+00,  1.06523383e+00,  7.57983923e-01,  2.73237914e-01,
            -2.21476659e-01, -1.90207940e-02,  1.52076006e-01,  3.98420781e-01,
             1.02454650e+00,  1.47291923e+00,  3.42558503e-01,  9.21884775e-02,
            -3.32788490e-02,  6.95218369e-02,  1.19572096e-01, -1.53917655e-01,
             3.99634205e-02,  6.50007308e-01,  6.45680189e-01,  4.96141046e-01,
             4.55260038e-01, -4.72639166e-02,  4.41695184e-01, -4.74886894e-01,
            -3.92314404e-01,  1.15423184e-02,  2.07411230e-01,  9.54728067e-01,
             1.81652629e+00, -3.67189318e-01, -4.59430879e-03,  7.17058405e-03,
            -5.08971214e-01,  5.85015774e-01, -3.02936554e-01, -3.48901004e-02,
             1.47924829e+00, -1.18879721e-01,  9.37922478e-01,  9.79301691e-01,
            -2.09888363e+00, -5.83763659e-01,  1.32382286e+00,  2.17423618e-01,
             1.54183358e-01,  3.74596059e-01,  9.12698805e-02,  7.63320327e-02,
            -6.20965838e-01,  1.08513546e+00,  1.31074488e+00,  2.64834076e-01,
            -6.14579022e-01, -1.73587054e-01, -7.67316744e-02,  1.24840789e-01,
             3.18735726e-02,  6.74190670e-02]

    # Define as parmoo expects
    des_type = [(f"{name}", float) for name in DES_NAMES]
    x1 = np.zeros(1, dtype=des_type)[0]
    for i, xi in enumerate(xin):
        x1[f"{DES_NAMES[i]}"] = xi * __scales__[i] + __shifts__[i]
    # Calculate the absolute error at scale of model's training
    ae = np.abs(__rescale__(fayans_sim_model(x1)) -
                __rescale__(np.asarray(xout)))
    # Check whether the absolute errors are too large
    try:
        assert(np.all(ae < 1.0e-6))
        print("Finished successfully.")
    except AssertionError:
        raise ValueError("Error reloading weights -- some of the absolute " +
                         "errors are too large")
    finally:
        print(f"Max absolute error: {np.max(ae)}")
