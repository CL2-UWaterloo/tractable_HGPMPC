"""
author: Leroy D'Souza
Credit: Functions/classes heavily borrowed from safe-control-gym with slight modifications
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import casadi as cs
import math
import os
import contextlib

from ds_utils import box_constraint
from smpc.utils import setup_terminal_costs
from smpc.controller_utils import fwdsim_w_pw_res
from mapping.util import Traj_DS
from common.plotting_utils import save_fig


class SymbolicModel:
    """Implements the dynamics model with symbolic variables.

    x_dot = f(x,u) serve as priors for the controllers.

    Notes:
        * naming convention on symbolic variable and functions.
            * for single-letter symbol, use {}_sym, otherwise use underscore for delimiter.
            * for symbolic functions to be exposed, use {}_func.

    """
    def __init__(self,
                 dynamics,
                 dt=1e-3,
                 integration_algo='cvodes',
                 lti=False):
        # Setup for dynamics.
        self.x_sym = dynamics["vars"]["X"]
        self.u_sym = dynamics["vars"]["U"]
        self.x_dot = dynamics["dyn_eqn"]
        if lti:
            self.Ac = dynamics["Ac"]
            self.Bc = dynamics["Bc"]
        # Sampling time.
        self.dt = dt
        # Integration algorithm.
        self.integration_algo = integration_algo
        # Variable dimensions.
        self.nx = self.x_sym.shape[0]
        self.nu = self.u_sym.shape[0]
        # Setup symbolic model.
        self.setup_model()
        # Setup Jacobian and Hessian of the dynamics function.
        self.setup_linearization()
        self.setup_rk4()

        self.lti = lti
        if lti:
            self.setup_exact_lin()
        # self.setup_linearization2()


    def setup_model(self):
        """Exposes functions to evaluate the model.

        """
        # Continuous time dynamics.
        self.fc_func = cs.Function('fc', [self.x_sym, self.u_sym], [self.x_dot], ['x', 'u'], ['f'])
        # Discrete time dynamics.
        self.fd_func = cs.integrator('fd', self.integration_algo, {'x': self.x_sym,
                                                                   'p': self.u_sym,
                                                                   'ode': self.x_dot}, {'tf': self.dt}
                                    )
        self.dfdx_disc = cs.jacobian(self.fd_func(self.x_sym, self.u_sym, 0, 0, 0, 0)[0], self.x_sym)
        self.dfdu_disc = cs.jacobian(self.fd_func(self.x_sym, self.u_sym, 0, 0, 0, 0)[0], self.u_sym)
        self.f_disc_linear_func = cs.Function('fd_linear', [self.x_sym, self.u_sym], [self.dfdx_disc, self.dfdu_disc], ['x', 'u'], ['dfdx_lin', 'dfdu_lin'])

    def setup_rk4(self):
        # ode_casadi = cs.Function("rk4_disc", [self.x_sym, self.u_sym], [self.x_dot])

        mu_x_sym, mu_u_sym = cs.MX.sym('mu_x_sym', self.nx, 1), cs.MX.sym('mu_u_sym', self.nu, 1)
        k1 = self.fc_func(mu_x_sym, mu_u_sym)
        k2 = self.fc_func(mu_x_sym + self.dt/2*k1, mu_u_sym)
        k3 = self.fc_func(mu_x_sym + self.dt/2*k2, mu_u_sym)
        k4 = self.fc_func(mu_x_sym + self.dt/k3, mu_u_sym)
        xrk4 = mu_x_sym + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
        self.rk4 = cs.Function("ode_rk4", [mu_x_sym, mu_u_sym], [xrk4], ['x', 'u'], ['f'])

    def setup_linearization(self):
        """Exposes functions for the linearized model.

        """
        # Jacobians w.r.t state & input.
        self.dfdx = cs.jacobian(self.x_dot, self.x_sym)
        self.dfdu = cs.jacobian(self.x_dot, self.u_sym)
        self.df_func = cs.Function('df', [self.x_sym, self.u_sym],
                                   [self.dfdx, self.dfdu], ['x', 'u'],
                                   ['dfdx', 'dfdu'])
        # Evaluation point for linearization.
        self.x_eval = cs.MX.sym('x_eval', self.nx, 1)
        self.u_eval = cs.MX.sym('u_eval', self.nu, 1)
        # Linearized dynamics model.
        self.x_dot_linear = self.x_dot + self.dfdx @ (
            self.x_eval - self.x_sym) + self.dfdu @ (self.u_eval - self.u_sym)
        self.fc_linear_func = cs.Function(
            'fc', [self.x_eval, self.u_eval, self.x_sym, self.u_sym],
            [self.x_dot_linear], ['x_eval', 'u_eval', 'x', 'u'], ['f_linear'])
        self.fd_linear_func = cs.integrator(
            'fd_linear', self.integration_algo, {
                'x': self.x_eval,
                'p': cs.vertcat(self.u_eval, self.x_sym, self.u_sym),
                'ode': self.x_dot_linear
            }, {'tf': self.dt})


    def setup_exact_lin(self):
        self.exact_flag = True
        dt = self.dt
        # dt = 1e-3
        try:
            Ainv = np.linalg.inv(self.Ac)
        except np.LinAlgError:
            self.exact_flag = False
        if self.exact_flag:
            Ad_exact_lti = scipy.linalg.expm(self.Ac * dt)
            Bd_exact_lti = np.linalg.inv(self.Ac) @ ((Ad_exact_lti - np.eye(self.Ac.shape[0])) @ self.Bc)
            exact_ns = Ad_exact_lti @ self.x_sym + Bd_exact_lti @ self.u_sym
            self.fd_linear_func_exact = cs.Function('fd_linear', [self.x_sym, self.u_sym], [exact_ns], ['x', 'u'], ['xf'])

            fine_disc = 1e-3
            Ad_exact_lti_fine = scipy.linalg.expm(self.Ac * fine_disc)
            Bd_exact_lti_fine = np.linalg.inv(self.Ac) @ ((Ad_exact_lti_fine - np.eye(self.Ac.shape[0])) @ self.Bc)
            exact_ns_fine = Ad_exact_lti_fine @ self.x_sym + Bd_exact_lti_fine @ self.u_sym
            self.fd_linear_func_exact_1ms = cs.Function('fd_linear_fine', [self.x_sym, self.u_sym], [exact_ns_fine], ['x', 'u'], ['xf'])

    def setup_linearization2(self):
        # Jacobians w.r.t state & input.
        self.dfdx = cs.jacobian(self.x_dot, self.x_sym)
        self.dfdu = cs.jacobian(self.x_dot, self.u_sym)
        self.df_func = cs.Function('df', [self.x_sym, self.u_sym],
                                   [self.dfdx, self.dfdu], ['x', 'u'],
                                   ['dfdx', 'dfdu'])
        dfdx = self.df_func['dfdx'].toarray()
        dfdu = self.df_func['dfdu'].toarray()
        # delta_x's are symbolic vars for x-xd across whole trajectory.
        delta_x = cs.MX.sym('delta_x', self.nx, 1)
        delta_u = cs.MX.sym('delta_u', self.nu, 1)
        # Find next state using linearized dynamics at every timestep
        x_dot_lin_vec = dfdx @ delta_x + dfdu @ delta_u
        # Linearized dynamics model.
        self.linear_dynamics_func = cs.integrator(
            'linear_discrete_dynamics', self.integration_algo,
            {
                'x': delta_x,
                'p': delta_u,
                'ode': x_dot_lin_vec
            }, {'tf': self.dt}
        )



class SystemDynamics:
    def __init__(self):
        self.dynamics = None
        self.state_space, self.action_space = None, None

    def setup_symbolic_dynamics(self):
        raise NotImplementedError

    def setup_input_constraint_set(self):
        raise NotImplementedError

    def setup_state_constraint_set(self):
        raise NotImplementedError

    def get_dyn_info_dict(self):
        raise NotImplementedError

    def setup_constraint_sets(self):
        self.setup_state_constraint_set()
        self.setup_input_constraint_set()



class quad_2D_dyn(SystemDynamics):
    nx, nu = 6, 2
    ctrl_freq = 5
    lti = False
    # dt = 20 * 1e-3

    def __init__(self, velocity_limit_override=None, x_threshold=7, z_threshold=7, dt=20*10e-3, x_min_threshold=None):
        # Parameters obtained from cf2x.urdf in gym_pybullet_drones dir using the indexing in _parse_urdf_parameters of the base_aviary.py file
        super().__init__()

        self.state_vec_verbose = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
        self.state_plot_idxs = [0, 2]

        self.x_threshold = x_threshold
        self.x_min_threshold = x_min_threshold
        if x_min_threshold is None:
            self.x_min_threshold = -x_threshold
        self.z_threshold = z_threshold
        self.dt = dt

        velocity_default_limit = 100
        self.velocity_limit = velocity_default_limit if velocity_limit_override is None else velocity_limit_override

        # Setup symbolic model instance
        self.setup_symbolic_dynamics()
        # print(self.dt)
        self.symbolic = SymbolicModel(dynamics=self.dynamics, dt=self.dt)

        # Setup constraint sets
        self.setup_constraint_sets()

    def setup_symbolic_dynamics(self):
        m, g, l = 0.027, cs.DM([9.8]), 0.0397
        Iyy = 1.4e-5
        # Define states.
        x = cs.MX.sym('x')
        z = cs.MX.sym('z')
        z_dot = cs.MX.sym('z_dot')
        x_dot = cs.MX.sym('x_dot')
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        X = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
        # Define input thrusts.
        T1 = cs.MX.sym('T1')
        T2 = cs.MX.sym('T2')
        # input vector is concatenation of both thrusts.
        U = cs.vertcat(T1, T2)
        # Define dynamics equations.
        X_dot = cs.vertcat(x_dot,
                           cs.sin(theta) * (T1 + T2) / m,
                           z_dot,
                           (cs.cos(theta) * (T1 + T2) / m) - g,
                           theta_dot,
                           l * (T2 - T1) / Iyy / np.sqrt(2))
        # Define dynamics dictionary to be used by the SymbolicModel class.
        self.dynamics = {"dyn_eqn": X_dot, "vars": {"X": X, "U": U}}

    def setup_state_constraint_set(self):
        x_threshold = self.x_threshold
        z_threshold = self.z_threshold
        theta_threshold_radians = 85 * math.pi / 180
        GROUND_PLANE_Z = -0.05
        # GROUND_PLANE_Z = 0

        low = np.array([[
            self.x_min_threshold, -self.velocity_limit,
            # 0, -self.velocity_limit,
            GROUND_PLANE_Z, -self.velocity_limit,
            -theta_threshold_radians, -5
        ]]).T
        high = np.array([[
            x_threshold, self.velocity_limit,
            z_threshold, self.velocity_limit,
            theta_threshold_radians, 5
        ]]).T
        self.state_space = box_constraint(lb=low, ub=high)

    def setup_input_constraint_set(self):
        KF = 3.16e-10
        PWM2RPM_SCALE = 0.2685
        PWM2RPM_CONST = 4070.3
        MIN_PWM = 20000.0
        MAX_PWM = 65535.0
        input_scale = 1

        # Direct thrust control.
        n_motors = 4 / self.nu
        a_low = KF * n_motors * (PWM2RPM_SCALE * MIN_PWM + PWM2RPM_CONST)**2
        a_high = KF * n_motors * (PWM2RPM_SCALE * MAX_PWM + PWM2RPM_CONST)**2
        # print([np.full(nu, a_low, np.float32), np.full(nu, a_high, np.float32)])
        # self.action_space = box_constraint(lb=np.full(nu, a_low, np.float32),
        #                                    ub=np.full(nu, a_high, np.float32))
        # Defining input constraint set.
        # a_low = -10
        # a_high = 10
        self.action_space = box_constraint(lb=np.ones((self.nu, 1))*a_low,
                                           ub=np.ones((self.nu, 1))*a_high*input_scale)
        # print(self.action_space)

    def get_dyn_info_dict(self):
        return {"n_x": self.nx, "n_u": self.nu, "sys_dyn": self.symbolic, "X": self.state_space, "U": self.action_space,
                "state_vec_verbose": self.state_vec_verbose, "state_plot_idxs": self.state_plot_idxs,
                "x_threshold": self.x_threshold, "z_threshold": self.z_threshold, "x_min_threshold": self.x_min_threshold, "dt": self.dt}


class planar_nl_sys(SystemDynamics):
    nx, nu = 2, 2
    ctrl_freq = 50
    lti = False
    # dt = 1/ctrl_freq

    def __init__(self, x1_threshold=7, x2_threshold=7, dt=20*10e-3):
        # Parameters obtained from cf2x.urdf in gym_pybullet_drones dir using the indexing in _parse_urdf_parameters of the base_aviary.py file
        super().__init__()

        self.state_vec_verbose = ['x1', 'x2']
        self.state_plot_idxs = [0, 1]

        self.x1_threshold = x1_threshold
        self.x2_threshold = x2_threshold
        self.dt = dt

        # Setup symbolic model instance
        self.setup_symbolic_dynamics()
        self.symbolic = SymbolicModel(dynamics=self.dynamics, dt=self.dt)

        # Setup constraint sets
        self.setup_constraint_sets()

    def setup_symbolic_dynamics(self):
        k = 1
        m = 1
        input_scale = 0

        x1 = cs.MX.sym('x1')
        x2 = cs.MX.sym('x2')
        X = cs.vertcat(x1, x2)

        # Define inputs
        u1 = cs.MX.sym('u1')
        u2 = cs.MX.sym('u2')
        # input vector is concatenation of both thrusts.
        U = cs.vertcat(u1, u2)
        # Define dynamics equations. Example from Slide 20 of https://www.egr.msu.edu/~khalil/NonlinearControl/Slides-Full/Lect_2.pdf
        # X_dot = cs.vertcat(-x2 - mu*x1*(x1**2 + x2**2) + input_scale*u1,
        #                    x1 - mu*x2*(x1**2 + x2**2) + input_scale*u2)
        X_dot = cs.vertcat(x2  + input_scale*u1,
                           -(k/m)*x1 + input_scale*u2)
        # Define dynamics dictionary to be used by the SymbolicModel class.
        self.dynamics = {"dyn_eqn": X_dot, "vars": {"X": X, "U": U}}

    def setup_state_constraint_set(self):
        x1_threshold = 7
        x2_threshold = 7
        GROUND_PLANE_Z = -0.05

        low = np.array([[
            -x1_threshold, GROUND_PLANE_Z
        ]]).T
        high = np.array([[
            x1_threshold, x2_threshold
        ]]).T
        self.state_space = box_constraint(lb=low, ub=high)

    def setup_input_constraint_set(self):
        a_low = -5
        a_high = 5
        self.action_space = box_constraint(lb=np.ones((self.nu, 1))*a_low,
                                           ub=np.ones((self.nu, 1))*a_high)

    def get_dyn_info_dict(self):
        return {"n_x": self.nx, "n_u": self.nu, "sys_dyn": self.symbolic, "X": self.state_space, "U": self.action_space,
                "state_vec_verbose": self.state_vec_verbose, "state_plot_idxs": self.state_plot_idxs}


class planar_lti_sys(SystemDynamics):
    nx, nu = 2, 2
    ctrl_freq = 50
    lti = True
    # dt = 1/ctrl_freq

    def __init__(self, x1_threshold=7, x2_threshold=7, u_max=5, dt=20*10e-3, A=None, B=None, x1_min_threshold=None):
        # Parameters obtained from cf2x.urdf in gym_pybullet_drones dir using the indexing in _parse_urdf_parameters of the base_aviary.py file
        super().__init__()

        self.A = A
        if A is None:
            self.A = np.array([[1, 0], [0, -1]])
        if B is None:
            self.B = np.eye(self.nu)

        self.state_vec_verbose = ['x1', 'x2']
        self.state_plot_idxs = [0, 1]

        self.x1_threshold = x1_threshold
        self.x1_min_threshold = x1_min_threshold
        if x1_min_threshold is None:
            self.x1_min_threshold = -x1_threshold
        self.x2_threshold = x2_threshold
        self.u_max = u_max
        self.dt = dt

        # Setup symbolic model instance
        self.setup_symbolic_dynamics()
        self.symbolic = SymbolicModel(dynamics=self.dynamics, dt=self.dt, lti=self.lti)

        # Setup constraint sets
        self.setup_constraint_sets()

    def setup_symbolic_dynamics(self):
        input_scale = 1

        x1 = cs.MX.sym('x1', 1, 1)
        x2 = cs.MX.sym('x2', 1, 1)
        X = cs.vertcat(x1, x2)

        # Define inputs
        u1 = cs.MX.sym('u1', 1, 1)
        u2 = cs.MX.sym('u2', 1, 1)
        # input vector is concatenation of both thrusts.
        U = cs.vertcat(u1, u2)
        X_dot = self.A @ cs.vertcat(x1, x2) + (input_scale * self.B @ cs.vertcat(u1, u2))
        # Define dynamics dictionary to be used by the SymbolicModel class.
        self.dynamics = {"dyn_eqn": X_dot, "vars": {"X": X, "U": U}, "Ac": self.A, "Bc": self.B}

    def setup_state_constraint_set(self):
        x1_threshold = self.x1_threshold
        x2_threshold = self.x2_threshold
        GROUND_PLANE_Z = -0.05

        low = np.array([[
            self.x1_min_threshold, GROUND_PLANE_Z
        ]]).T
        high = np.array([[
            x1_threshold, x2_threshold
        ]]).T
        self.state_space = box_constraint(lb=low, ub=high)

    def setup_input_constraint_set(self):
        a_low = -self.u_max
        a_high = self.u_max
        self.action_space = box_constraint(lb=np.ones((self.nu, 1))*a_low,
                                           ub=np.ones((self.nu, 1))*a_high)

    def get_dyn_info_dict(self):
        return {"n_x": self.nx, "n_u": self.nu, "sys_dyn": self.symbolic, "X": self.state_space, "U": self.action_space,
                "state_vec_verbose": self.state_vec_verbose, "state_plot_idxs": self.state_plot_idxs,
                "A": self.A, "B": self.B, "x1_threshold": self.x1_threshold, "x2_threshold": self.x2_threshold,
                "u_max": self.u_max, "x1_min_threshold": self.x1_min_threshold, "dt": self.dt}


def discretize_linear_system(A,
                             B,
                             dt,
                             exact=False
                             ):
    """
    Discretize a linear system.

    dx/dt = A x + B u
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A: np.array, system transition matrix.
        B: np.array, input matrix.
        dt: scalar, step time interval.
        exact: bool, if to use exact discretization.

    Returns:
        Discretized matrices Ad, Bd.

    """
    state_dim, input_dim = A.shape[1], B.shape[1]
    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B
        Md = scipy.linalg.expm(M * dt)
        Ad = Md[:state_dim, :state_dim]
        Bd = Md[:state_dim, state_dim:]
    else:
        I = np.eye(state_dim)
        Ad = I + A * dt
        Bd = B * dt
    return Ad, Bd


def rk_discrete(f, n, m, dt):
    """
    Runge Kutta discretization for the function.
    Args:
        f (casadi function): Function to discretize.
        n (int): state dimensions.
        m (int): input dimension.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    """
    X = cs.SX.sym('X', n)
    U = cs.SX.sym('U', m)
    # Runge-Kutta 4 integration
    k1 = f(X,         U)
    k2 = f(X+dt/2*k1, U)
    k3 = f(X+dt/2*k2, U)
    k4 = f(X+dt*k3,   U)
    x_next = X + dt/6*(k1+2*k2+2*k3+k4)
    rk_dyn = cs.Function('rk_f', [X, U], [x_next], ['x0', 'p'], ['xf'])

    return rk_dyn


def test_quad_2d(test_jit=False, velocity_limit_override=100):
    opti = cs.Opti()
    N = 30  # Discrete time horizon
    dt = 20*10e-3  # 20ms sampling time
    T = N*dt  # prediction horizon
    quad_2d_inst = quad_2D_dyn(velocity_limit_override=velocity_limit_override)
    quad_2d_symbolic, U, X = quad_2d_inst.symbolic, quad_2d_inst.action_space, quad_2d_inst.state_space
    ct_dyn = quad_2d_symbolic.fc_func
    x = cs.horzcat(opti.parameter(quad_2d_symbolic.nx, 1), opti.variable(quad_2d_symbolic.nx, N))
    u = opti.variable(quad_2d_symbolic.nu, N)
    Q = np.diag(np.ones(quad_2d_symbolic.nx)*1)
    # Tracking in x and z assigned high priority
    Q[0, 0] = 5
    Q[2, 2] = 5
    R = np.diag(np.ones(quad_2d_symbolic.nu)*0.1)
    track_state = np.array([[2.5, 0, 2.5, 0, 0, 0]]).T
    print(X)
    print(U)
    # track_input = np.array(np.zeros(quad_2d_symbolic.nu, 1))
    cost = 0
    for i in range(N):
        # state and input constraints
        opti.subject_to(U.H_np @ u[:, i] <= U.b_np)
        if i>0:
            opti.subject_to(X.H_np @ x[:, i] <= X.b_np)

        # Dynamics constraints
        # f_xkt_ukt = ct_dyn(x[:, i], u[:, i])['f']
        # Euler approximation of CT dynamics integration over [kdt: (k+1)dt)
        opti.subject_to(x[:, i+1] == (x[:, i] + dt*(ct_dyn(x=x[:, i], u=u[:, i])['f'])))

        # Stage costs
        x_dev = (x[:, [i]] - track_state)
        cost += (x_dev.T @ Q @ x_dev + u[:, [i]].T @ R @ u[:, [i]])

    x_dev_N = (x[:, [N]] - track_state)
    P = opti.parameter(quad_2d_symbolic.nx, quad_2d_symbolic.nx)
    # Terminal cost term
    cost += x_dev_N.T @ P @ x_dev_N
    opti.minimize(cost)

    acceptable_dual_inf_tol = 1e11
    acceptable_compl_inf_tol = 1e-3
    acceptable_iter = 10
    acceptable_constr_viol_tol = 1e-3
    acceptable_tol = 1e4
    additional_opts = {"ipopt.acceptable_tol": acceptable_tol,
                       "ipopt.acceptable_constr_viol_tol": acceptable_constr_viol_tol,
                       "ipopt.acceptable_dual_inf_tol": acceptable_dual_inf_tol,
                       "ipopt.acceptable_iter": acceptable_iter,
                       "ipopt.acceptable_compl_inf_tol": acceptable_compl_inf_tol,
                       "ipopt.hessian_approximation": "limited-memory",
                       "ipopt.print_level": 5}
    if test_jit:
        jit_options = {"flags": ["-O3"], "verbose": True}
        options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        additional_opts.update(options)
    opti.solver('ipopt', additional_opts)

    # For penultimate cost, use Casadi to linearize nominal dynamics about penultimate state and calculate cost that way.
    partial_der_calc = quad_2d_symbolic.df_func(x=track_state, u=np.zeros(quad_2d_symbolic.nu))
    A_xf, B_xf = partial_der_calc['dfdx'], partial_der_calc['dfdu']
    _, P_val = setup_terminal_costs(A_xf, B_xf, Q, R)
    opti.set_value(P, P_val)
    opti.set_value(x[:, 0], np.zeros((quad_2d_symbolic.nx, 1)))

    sol = opti.solve()
    fig, ax = plt.subplots(1, 1)
    ax.plot(opti.debug.value(x)[0], opti.debug.value(x)[2])
    # print(opti.debug.value(x))

    # opti.set_value(P, P_val)
    # opti.set_value(x[:, 0], np.zeros((quad_2d_symbolic.nx, 1)))
    # sol = opti.solve()


def verify_kwargs_for_pw_fwd_sim(forward_sim_kwargs):
    reqd_keys = ["true_func_obj", "Bd", "gp_input_mask", "delta_input_mask"]
    for key in reqd_keys:
        assert key in forward_sim_kwargs.keys(), "Missing key {} in forward_sim_kwargs".format(key)


def test_quad_2d_track(test_jit=False, num_discrete=120, sim_steps=100, viz=True, verbose=True, N=30,
                       velocity_override=5, ret_inputs=False, sampling_time=20*10e-3, waypoints_arr=None,
                       ax_xlim_override=None, ax_ylim_override=None, plot_wps=True,
                       forward_sim="nominal_dyn", forward_sim_kwargs={}, x_init=None, expl_cost_fn=None, ax=None,
                       regions_2_viz=None, inst_override=None, integration='euler', Q_override=None, R_override=None):
    opti = cs.Opti()
    # dt = sampling_time  # 20ms sampling time
    # T = N*dt  # prediction horizon
    if inst_override is None:
        quad_2d_inst = quad_2D_dyn(velocity_limit_override=velocity_override)
    else:
        quad_2d_inst = inst_override
    U, X = quad_2d_inst.action_space, quad_2d_inst.state_space
    quad_2d_symbolic = quad_2d_inst.symbolic
    if x_init is None:
        x_init = np.zeros((quad_2d_symbolic.nx, 1))
    ct_dyn = quad_2d_symbolic.fc_func
    tracking_matrix = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    x_desired, u_desired = test_track_sim(opti, ct_dyn, symbolic_inst=quad_2d_symbolic, U=U, X=X, tracking_matrix=tracking_matrix, plot_idxs=quad_2d_inst.state_plot_idxs,
                                          test_jit=test_jit, num_discrete=num_discrete, sim_steps=sim_steps, viz=viz, verbose=verbose, N=N,
                                          velocity_override=velocity_override, ret_inputs=ret_inputs, sampling_time=sampling_time,
                                          waypoints_arr=waypoints_arr, ax_xlim_override=ax_xlim_override,
                                          ax_ylim_override=ax_ylim_override, plot_wps=plot_wps, forward_sim=forward_sim,
                                          forward_sim_kwargs=forward_sim_kwargs, x_init=x_init, expl_cost_fn=expl_cost_fn, ax=ax,
                                          regions_2_viz=regions_2_viz, Q_override=Q_override, R_override=R_override)
    return x_desired, u_desired


def test_planar_sys_track(test_jit=False, num_discrete=120, sim_steps=100, viz=True, verbose=True, N=30,
                          velocity_override=5, ret_inputs=False, sampling_time=20*10e-3, waypoints_arr=None,
                          ax_xlim_override=None, ax_ylim_override=None, plot_wps=True,
                          forward_sim="nominal_dyn", forward_sim_kwargs={}, x_init=None, expl_cost_fn=None, ax=None,
                          regions_2_viz=None):
    opti = cs.Opti()
    planar_sys_inst = planar_nl_sys()
    U, X = planar_sys_inst.action_space, planar_sys_inst.state_space
    planar_sys_sym = planar_sys_inst.symbolic
    if x_init is None:
        x_init = np.zeros((planar_sys_sym.nx, 1))
    ct_dyn = planar_sys_sym.fc_func
    tracking_matrix = np.array([[1, 0], [0, 1]])
    x_desired, u_desired = test_track_sim(opti, ct_dyn, symbolic_inst=planar_sys_sym, U=U, X=X, tracking_matrix=tracking_matrix, plot_idxs=planar_sys_inst.state_plot_idxs,
                                          test_jit=test_jit, num_discrete=num_discrete, sim_steps=sim_steps, viz=viz, verbose=verbose, N=N,
                                          velocity_override=velocity_override, ret_inputs=ret_inputs, sampling_time=sampling_time,
                                          waypoints_arr=waypoints_arr, ax_xlim_override=ax_xlim_override,
                                          ax_ylim_override=ax_ylim_override, plot_wps=plot_wps, forward_sim=forward_sim,
                                          forward_sim_kwargs=forward_sim_kwargs, x_init=x_init, expl_cost_fn=expl_cost_fn, ax=ax,
                                          regions_2_viz=regions_2_viz)
    return x_desired, u_desired


def test_planar_lti_track(test_jit=False, num_discrete=120, sim_steps=100, viz=True, verbose=True, N=30,
                          velocity_override=5, ret_inputs=False, sampling_time=20*10e-3, waypoints_arr=None,
                          plot_wps=True, forward_sim="nominal_dyn", forward_sim_kwargs={}, x_init=None, expl_cost_fn=None, ax=None,
                          regions_2_viz=None, inst_override=None, integration='exact', Q_override=None, R_override=None):
    opti = cs.Opti()
    if inst_override is None:
        planar_lti_sys_inst = planar_lti_sys()
    else:
        planar_lti_sys_inst = inst_override
    if integration == 'exact':
        if planar_lti_sys_inst.symbolic.exact_flag is False:
            print("CANNOT PERFORM EXACT INTEGRATION. A matrix is not invertible. Reverting to euler integration.")
            integration = 'euler'
    U, X = planar_lti_sys_inst.action_space, planar_lti_sys_inst.state_space
    planar_sys_sym = planar_lti_sys_inst.symbolic
    if x_init is None:
        x_init = np.zeros((planar_sys_sym.nx, 1))
    ct_dyn = planar_sys_sym.fc_func
    tracking_matrix = np.array([[1, 0], [0, 1]])
    x_desired, u_desired = test_track_sim(opti, ct_dyn, symbolic_inst=planar_sys_sym, U=U, X=X, tracking_matrix=tracking_matrix, plot_idxs=planar_lti_sys_inst.state_plot_idxs,
                                          test_jit=test_jit, num_discrete=num_discrete, sim_steps=sim_steps, viz=viz, verbose=verbose, N=N,
                                          ret_inputs=ret_inputs, sampling_time=sampling_time,
                                          waypoints_arr=waypoints_arr, ax_xlim_override=planar_lti_sys_inst.x1_threshold,
                                          ax_ylim_override=planar_lti_sys_inst.x2_threshold, plot_wps=plot_wps, forward_sim=forward_sim,
                                          forward_sim_kwargs=forward_sim_kwargs, x_init=x_init, expl_cost_fn=expl_cost_fn, ax=ax,
                                          regions_2_viz=regions_2_viz, integration=integration, Q_override=Q_override, R_override=R_override)
    return x_desired, u_desired


def test_track_sim(opti, ct_dyn, symbolic_inst, U, X, tracking_matrix, plot_idxs,
                   test_jit=False, num_discrete=120, sim_steps=100, viz=True, verbose=True, N=30,
                   velocity_override=5, ret_inputs=False, sampling_time=20*10e-3, waypoints_arr=None,
                   ax_xlim_override=None, ax_ylim_override=None, plot_wps=True,
                   forward_sim="nominal_dyn", forward_sim_kwargs={}, x_init=None, expl_cost_fn=None, ax=None,
                   regions_2_viz=None, integration='euler', Q_override=None, R_override=None):
    dt = sampling_time
    if forward_sim == "w_res_pw":
        verify_kwargs_for_pw_fwd_sim(forward_sim_kwargs)
        forward_sim_kwargs.update({"ct_dyn_nom": ct_dyn, "dt": dt})
        gp_inp_dim = np.linalg.matrix_rank(forward_sim_kwargs["gp_input_mask"])
        n_d = np.linalg.matrix_rank(forward_sim_kwargs["Bd"])
        delta_ctrl_dim = np.linalg.matrix_rank(forward_sim_kwargs["delta_input_mask"])
        assert n_d == 1, "Currently only support 1D disturbance. Code needs to be modified for higher dimensions"
        created_ds = {"pw_gp_inp": np.zeros([gp_inp_dim, sim_steps]),
                      "delta_control_vec": np.zeros([delta_ctrl_dim, sim_steps]),
                      "measured_res": np.zeros([n_d, sim_steps])}

    x = cs.horzcat(opti.parameter(symbolic_inst.nx, 1), opti.variable(symbolic_inst.nx, N))
    u = opti.variable(symbolic_inst.nu, N)
    Q = Q_override
    R = R_override
    if Q_override is None:
        Q = np.diag(np.ones(symbolic_inst.nx)*1)
    if R_override is None:
        R = np.diag(np.ones(symbolic_inst.nu)*0.1)
    if waypoints_arr is None:
        waypoints_arr = [(0.5, 0), (0.5, 0.5), (0, 0.5), (0, 0)]
    waypoints = np.hstack([np.array([[x, y]]).T for (x, y) in waypoints_arr])
    # print("PRINTING INFO FROM TRACKING SIM")
    # print(waypoints)
    num_waypoints = waypoints.shape[-1]
    # Divide number of discrete points in the trajectory to track equally among all waypoints.
    total_discrete = num_discrete
    discrete_per_waypoint, rem_discrete = total_discrete // num_waypoints, (total_discrete % num_waypoints + max(0, (sim_steps+N) - total_discrete))
    setpoint_track_arr = np.hstack([np.hstack([waypoints[:, [i]]]*discrete_per_waypoint) for i in range(num_waypoints)])
    # If num discrete not exactly divisible by num waypoints, append last waypoint to discrete traj to track till desired length is reached.
    if rem_discrete != 0:
        setpoint_track_arr = np.hstack([setpoint_track_arr, np.hstack([waypoints[:, [-1]]]*rem_discrete)])
    # print(setpoint_track_arr.shape)
    # print(setpoint_track_arr)

    track_idxs = [np.nonzero(tracking_matrix[k, :])[0].item() for k in range(tracking_matrix.shape[0])]
    x_desired = np.zeros([symbolic_inst.nx, setpoint_track_arr.shape[-1]])
    x_desired[track_idxs, :] = setpoint_track_arr
    print(X)
    print(U)
    # track_input = np.array(np.zeros(quad_2d_symbolic.nu, 1))
    cost = 0
    track_arr = opti.parameter(symbolic_inst.nx, N+1)
    for i in range(N):
        # state and input constraints
        opti.subject_to(U.H_np @ u[:, i] <= U.b_np)
        if i > 0:
            opti.subject_to(X.H_np @ x[:, i] <= X.b_np)

        # Dynamics constraints
        # f_xkt_ukt = ct_dyn(x[:, i], u[:, i])['f']
        # Euler approximation of CT dynamics integration over [kdt: (k+1)dt)
        if integration == 'exact':
            # print(symbolic_inst.fd_linear_func_exact(x=x[:, i], u=u[:, i])['xf'])
            opti.subject_to(x[:, i+1] == symbolic_inst.fd_linear_func_exact(x=x[:, i], u=u[:, i])['xf'])
        if integration == 'euler':
            opti.subject_to(x[:, i+1] == (x[:, i] + dt*(ct_dyn(x=x[:, i], u=u[:, i])['f'])))
            # opti.subject_to(x[:, i+1] == ct_dyn(x=x[:, i], u=u[:, i])['f'])

        # Stage costs
        x_dev = (x[:, [i]] - track_arr[:, i])
        cost += (x_dev.T @ Q @ x_dev + u[:, [i]].T @ R @ u[:, [i]])
        if expl_cost_fn is not None:
            cost += expl_cost_fn(x[:, [i]])

    x_dev_N = (x[:, [N]] - track_arr[:, N])
    P = opti.parameter(symbolic_inst.nx, symbolic_inst.nx)
    # Terminal cost term
    cost += x_dev_N.T @ P @ x_dev_N
    opti.minimize(cost)

    acceptable_dual_inf_tol = 1e11
    acceptable_compl_inf_tol = 1e-3
    acceptable_iter = 5
    acceptable_constr_viol_tol = 1e-3
    acceptable_tol = 1e4
    additional_opts = {"ipopt.acceptable_tol": acceptable_tol,
                       "ipopt.acceptable_constr_viol_tol": acceptable_constr_viol_tol,
                       "ipopt.acceptable_dual_inf_tol": acceptable_dual_inf_tol,
                       "ipopt.acceptable_iter": acceptable_iter,
                       "ipopt.acceptable_compl_inf_tol": acceptable_compl_inf_tol,
                       "ipopt.hessian_approximation": "limited-memory",
                       "ipopt.print_level": 5}
    if not verbose:
        additional_opts["ipopt.print_level"] = 0
    if test_jit:
        jit_options = {"flags": ["-O3"], "verbose": True}
        options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        additional_opts.update(options)
    opti.solver('ipopt', additional_opts)

    cl_traj = np.zeros([symbolic_inst.nx, sim_steps+N+1])
    cl_traj[:, [0]] = x_init
    if ret_inputs:
        cl_inputs = np.zeros([symbolic_inst.nu, sim_steps])
    # For penultimate cost, use Casadi to linearize nominal dynamics about penultimate state and calculate cost that way.

    for sim_step in range(sim_steps):
        track_subtraj = x_desired[:, sim_step: sim_step+N+1]
        # print(track_subtraj)
        opti.set_value(track_arr, track_subtraj)
        partial_der_calc = symbolic_inst.df_func(x=track_subtraj[:, [-1]], u=np.zeros(symbolic_inst.nu))
        A_xf, B_xf = partial_der_calc['dfdx'], partial_der_calc['dfdu']
        _, P_val = setup_terminal_costs(A_xf, B_xf, Q, R)
        opti.set_value(P, P_val)
        opti.set_value(x[:, 0], cl_traj[:, sim_step])

        try:
            if verbose is True:
                sol = opti.solve()
            else:
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    sol = opti.solve()
        except RuntimeError:
            pass

        # cl_traj[:, sim_step] = opti.debug.value(x)[:, 1]
        x_0, u_0 = cl_traj[:, [sim_step]], opti.debug.value(u)[:, [0]]
        if forward_sim == "nominal_dyn":
            # Just planning so don't need to actually properly forward simulate with CVODES. Just take next state generated
            # by optimization as the true next state.
            nom_dyn_ns = opti.debug.value(x)[:, 1]
            cl_traj[:, sim_step+1] = nom_dyn_ns
        elif forward_sim == "w_res_pw":
            forward_sim_kwargs.update({"x_0": x_0, "u_0": u_0})
            sampled_ns, sampled_residual = fwdsim_w_pw_res(**forward_sim_kwargs, ret_residual=True)
            created_ds["pw_gp_inp"][:, sim_step] = forward_sim_kwargs["gp_input_mask"] @ np.vstack((x_0, u_0))
            created_ds["delta_control_vec"][:, [sim_step]] = forward_sim_kwargs["delta_input_mask"] @ np.vstack((x_0, u_0))
            created_ds["measured_res"][:, sim_step] = sampled_residual
            cl_traj[:, [sim_step+1]] = sampled_ns
        if ret_inputs:
            cl_inputs[:, [sim_step]] = u_0


    print("Nominal MPC traj to warmstart and track solved.")

    cl_traj[:, sim_steps:] = opti.debug.value(x)[:, [1]]  # Broadcast next state from final O.L. opt across rest of traj to track

    # if ax is not None:
    #     print("TRAJECTORY PRINTING")
    #     t = np.linspace(0, sim_steps, sim_steps)
    #     print(t.shape)
    #     print(cl_traj[0, :sim_steps])
    #     print(cl_traj[2, :sim_steps])
    #     ax[0].plot(t, cl_traj[0, :sim_steps])
    #     ax[1].plot(t, cl_traj[2, :sim_steps])
    #     # plt.show()
    if viz:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if regions_2_viz is not None:
            colours = ['r', 'g', 'b']
            for region_idx, region in enumerate(regions_2_viz):
                region.plot_constraint_set(ax=ax, alpha=0.6, colour=colours[region_idx])
        ax.plot(cl_traj[plot_idxs[0], :], cl_traj[plot_idxs[1], :], color="blue", marker='o',
                linestyle='solid', linewidth=3, markersize=7, markerfacecolor='blue',
                label='OL output')
        if plot_wps:
            ax.scatter(np.array([waypoints_arr[i][0] for i in range(len(waypoints_arr))]),
                       np.array([waypoints_arr[i][1] for i in range(len(waypoints_arr))]), color="green", marker='x',
                       linestyle='solid', linewidth=2)
        if ax_xlim_override is not None:
            ax.set_xlim(ax_xlim_override)
        if ax_ylim_override is not None:
            ax.set_ylim(ax_ylim_override)
        # save_fig(axes=[ax], fig_name="traj_op_base_mapping", tick_sizes=16)
        plt.show()
    if forward_sim == "nominal_dyn":
        if not ret_inputs:
            return cl_traj
        else:
            return cl_traj, np.hstack([cl_inputs, opti.debug.value(u)])
    elif forward_sim == "w_res_pw":
        traj_ds_inst = Traj_DS(created_ds["pw_gp_inp"], created_ds["measured_res"], created_ds["delta_control_vec"])
        return traj_ds_inst
    else:
        raise NotImplementedError
        # print(opti.debug.value(x))

        # opti.set_value(P, P_val)
        # opti.set_value(x[:, 0], np.zeros((quad_2d_symbolic.nx, 1)))
        # sol = opti.solve()


def test_track_sim_timing(opti, ct_dyn, symbolic_inst, U, X, tracking_matrix, plot_idxs,
                           test_jit=False, num_discrete=120, sim_steps=100, viz=True, verbose=True, N=30,
                           velocity_override=5, ret_inputs=False, sampling_time=20*10e-3, waypoints_arr=None,
                           ax_xlim_override=None, ax_ylim_override=None, plot_wps=True,
                           forward_sim="nominal_dyn", forward_sim_kwargs={}, x_init=None, expl_cost_fn=None, ax=None,
                           regions_2_viz=None, integration='euler', Q_override=None, R_override=None, k_step=1):
    dt = sampling_time

    x = cs.horzcat(opti.parameter(symbolic_inst.nx, 1), opti.variable(symbolic_inst.nx, N))
    u = opti.variable(symbolic_inst.nu, N)
    Q = Q_override
    R = R_override
    if Q_override is None:
        Q = np.diag(np.ones(symbolic_inst.nx)*1)
    if R_override is None:
        R = np.diag(np.ones(symbolic_inst.nu)*0.1)
    if waypoints_arr is None:
        waypoints_arr = [(0.5, 0), (0.5, 0.5), (0, 0.5), (0, 0)]
    waypoints = np.hstack([np.array([[x, y]]).T for (x, y) in waypoints_arr])
    # print("PRINTING INFO FROM TRACKING SIM")
    # print(waypoints)
    num_waypoints = waypoints.shape[-1]
    # Divide number of discrete points in the trajectory to track equally among all waypoints.
    total_discrete = num_discrete
    discrete_per_waypoint, rem_discrete = total_discrete // num_waypoints, (total_discrete % num_waypoints + max(0, (sim_steps+N) - total_discrete))
    setpoint_track_arr = np.hstack([np.hstack([waypoints[:, [i]]]*discrete_per_waypoint) for i in range(num_waypoints)])
    # If num discrete not exactly divisible by num waypoints, append last waypoint to discrete traj to track till desired length is reached.
    if rem_discrete != 0:
        setpoint_track_arr = np.hstack([setpoint_track_arr, np.hstack([waypoints[:, [-1]]]*rem_discrete)])
    # print(setpoint_track_arr.shape)
    # print(setpoint_track_arr)

    track_idxs = [np.nonzero(tracking_matrix[k, :])[0].item() for k in range(tracking_matrix.shape[0])]
    x_desired = np.zeros([symbolic_inst.nx, setpoint_track_arr.shape[-1]])
    x_desired[track_idxs, :] = setpoint_track_arr
    print(X)
    print(U)
    # track_input = np.array(np.zeros(quad_2d_symbolic.nu, 1))
    cost = 0
    track_arr = opti.parameter(symbolic_inst.nx, N+1)
    for i in range(N):
        # state and input constraints
        opti.subject_to(U.H_np @ u[:, i] <= U.b_np)
        if i > 0:
            opti.subject_to(X.H_np @ x[:, i] <= X.b_np)

        # Dynamics constraints
        # f_xkt_ukt = ct_dyn(x[:, i], u[:, i])['f']
        # Euler approximation of CT dynamics integration over [kdt: (k+1)dt)
        if integration == 'exact':
            # print(symbolic_inst.fd_linear_func_exact(x=x[:, i], u=u[:, i])['xf'])
            opti.subject_to(x[:, i+1] == symbolic_inst.fd_linear_func_exact(x=x[:, i], u=u[:, i])['xf'])
        if integration == 'euler':
            opti.subject_to(x[:, i+1] == (x[:, i] + dt*(ct_dyn(x=x[:, i], u=u[:, i])['f'])))
            # opti.subject_to(x[:, i+1] == ct_dyn(x=x[:, i], u=u[:, i])['f'])

        # Stage costs
        x_dev = (x[:, [i]] - track_arr[:, i])
        cost += (x_dev.T @ Q @ x_dev + u[:, [i]].T @ R @ u[:, [i]])
        if expl_cost_fn is not None:
            cost += expl_cost_fn(x[:, [i]])

    x_dev_N = (x[:, [N]] - track_arr[:, N])
    P = opti.parameter(symbolic_inst.nx, symbolic_inst.nx)
    # Terminal cost term
    cost += x_dev_N.T @ P @ x_dev_N
    opti.minimize(cost)

    acceptable_dual_inf_tol = 1e11
    acceptable_compl_inf_tol = 1e-3
    acceptable_iter = 5
    acceptable_constr_viol_tol = 1e-3
    acceptable_tol = 1e4
    additional_opts = {"ipopt.acceptable_tol": acceptable_tol,
                       "ipopt.acceptable_constr_viol_tol": acceptable_constr_viol_tol,
                       "ipopt.acceptable_dual_inf_tol": acceptable_dual_inf_tol,
                       "ipopt.acceptable_iter": acceptable_iter,
                       "ipopt.acceptable_compl_inf_tol": acceptable_compl_inf_tol,
                       "ipopt.hessian_approximation": "limited-memory",
                       "ipopt.print_level": 5}
    if not verbose:
        additional_opts["ipopt.print_level"] = 0
    if test_jit:
        jit_options = {"flags": ["-O3"], "verbose": True}
        options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        additional_opts.update(options)
    opti.solver('ipopt', additional_opts)

    cl_traj = np.zeros([symbolic_inst.nx, sim_steps+N+1])
    cl_traj[:, [0]] = x_init
    if ret_inputs:
        cl_inputs = np.zeros([symbolic_inst.nu, sim_steps])
    # For penultimate cost, use Casadi to linearize nominal dynamics about penultimate state and calculate cost that way.

    for sim_step in range(0, sim_steps+1, k_step):
        if sim_step == sim_steps:
            sim_step -= 1
        track_subtraj = x_desired[:, sim_step: sim_step+N+1]
        # print(track_subtraj)
        opti.set_value(track_arr, track_subtraj)
        partial_der_calc = symbolic_inst.df_func(x=track_subtraj[:, [-1]], u=np.zeros(symbolic_inst.nu))
        A_xf, B_xf = partial_der_calc['dfdx'], partial_der_calc['dfdu']
        _, P_val = setup_terminal_costs(A_xf, B_xf, Q, R)
        opti.set_value(P, P_val)
        opti.set_value(x[:, 0], cl_traj[:, sim_step])

        try:
            if verbose is True:
                sol = opti.solve()
            else:
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    sol = opti.solve()
        except RuntimeError:
            pass

        if forward_sim == "nominal_dyn":
            # Just planning so don't need to actually properly forward simulate with CVODES. Just take next state generated
            # by optimization as the true next state.
            nom_dyn_ns = opti.debug.value(x)[:, 1:k_step+1]
            cl_traj[:, sim_step+k_step] = nom_dyn_ns


    print("Nominal MPC traj to warmstart and track solved.")

    # cl_traj[:, sim_steps:] = opti.debug.value(x)[:, [1]]  # Broadcast next state from final O.L. opt across rest of traj to track

    # if ax is not None:
    #     print("TRAJECTORY PRINTING")
    #     t = np.linspace(0, sim_steps, sim_steps)
    #     print(t.shape)
    #     print(cl_traj[0, :sim_steps])
    #     print(cl_traj[2, :sim_steps])
    #     ax[0].plot(t, cl_traj[0, :sim_steps])
    #     ax[1].plot(t, cl_traj[2, :sim_steps])
    #     # plt.show()
    if viz:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if regions_2_viz is not None:
            colours = ['r', 'g', 'b']
            for region_idx, region in enumerate(regions_2_viz):
                region.plot_constraint_set(ax=ax, alpha=0.6, colour=colours[region_idx])
        ax.plot(cl_traj[plot_idxs[0], :], cl_traj[plot_idxs[1], :], color="blue", marker='o',
                linestyle='solid', linewidth=3, markersize=7, markerfacecolor='blue',
                label='OL output')
        if plot_wps:
            ax.scatter(np.array([waypoints_arr[i][0] for i in range(len(waypoints_arr))]),
                       np.array([waypoints_arr[i][1] for i in range(len(waypoints_arr))]), color="green", marker='x',
                       linestyle='solid', linewidth=2)
        if ax_xlim_override is not None:
            ax.set_xlim(ax_xlim_override)
        if ax_ylim_override is not None:
            ax.set_ylim(ax_ylim_override)
        # save_fig(axes=[ax], fig_name="traj_op_base_mapping", tick_sizes=16)
        plt.show()
    if forward_sim == "nominal_dyn":
        if not ret_inputs:
            return cl_traj
        else:
            return cl_traj, np.hstack([cl_inputs, opti.debug.value(u)])


if __name__ == "__main__":
    test_quad_2d_track(test_jit=False, num_discrete=120, sim_steps=100, viz=True, verbose=True, N=30,
                       velocity_override=5)




