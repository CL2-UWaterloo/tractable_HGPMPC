import casadi as cs


def covSEard(x,
             z,
             ell,
             sf2
             ):
    """GP squared exponential kernel.

    This function is based on the 2018 GP-MPC library by Helge-André Langåker

    Args:
        x (np.array or casadi.MX/SX): First vector.
        z (np.array or casadi.MX/SX): Second vector.
        ell (np.array or casadi.MX/SX): Length scales.
        sf2 (float or casadi.MX/SX): output scale parameter.

    Returns:
        SE kernel (casadi.MX/SX): SE kernel.

    """
    dist = cs.sum1((x - z)**2 / ell**2)
    return sf2 * cs.SX.exp(-.5 * dist)
