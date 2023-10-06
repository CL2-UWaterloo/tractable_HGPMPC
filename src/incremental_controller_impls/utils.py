import numpy as np
from smpc.utils import setup_terminal_costs


def linearize_for_terminal(sym_dyn_model, x_desired, n_u, Q, R):
    partial_der_calc = sym_dyn_model.df_func(x=x_desired, u=np.zeros((n_u, 1)))
    A_xf, B_xf = partial_der_calc['dfdx'], partial_der_calc['dfdu']
    _, P_val = setup_terminal_costs(A_xf, B_xf, Q, R)
    return P_val

