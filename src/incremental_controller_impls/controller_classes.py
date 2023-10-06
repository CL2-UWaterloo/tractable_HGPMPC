import os
import contextlib
import casadi as cs
import copy
import functools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import scipy

from sys_dyn.problem_setups import *
from sys_dyn.nlsys_utils import SymbolicModel, test_quad_2d_track
from .utils import linearize_for_terminal
from smpc.utils import delta_matrix_mult, get_inv_cdf, hybrid_res_covar, Sigma_u_Callback, computeSigmaglobal_meaneq, \
    computeSigma_nofeedback
from smpc.controller_utils import fwdsim_w_pw_res
from common.plotting_utils import plot_constraint_sets
from mapping.nn_map import SoftLabelNet
from mapping.util import Mapping_DS
from models.utils import GP_Model
from mapping.nn_map import train_gp_regionnet
from ds_utils import combine_box


# sanity check.
class BaseGPMPC_sanity:
    def __init__(self, sys_dyn, Q, R, horizon_length, X: box_constraint, U: box_constraint,
                 n_x, n_u, n_d, terminal_calc_fn="linearized_lqr",
                 solver='ipopt', addn_solver_opts=None,
                 ignore_init_constr_check=False, add_scaling=False, sampling_time=20 * 10e-3,
                 integration_method='euler',
                 ignore_callback=False, periodic_callback_freq=None, ignore_cost=False, lti=False,
                 use_prev_if_infeas=False, **kwargs):

        self.ignore_callback = ignore_callback

        # assert type(sys_dyn) == SymbolicModel, "symbolic_dynamics_model arg must be of type SymbolicModel (nlsys_utils.py)"
        self.sys_dyn: SymbolicModel = sys_dyn
        self.ct_dyn_nom = self.sys_dyn.fc_func
        self.dt = sampling_time
        self.integration_method = integration_method

        self.X, self.U = X, U
        self.N = horizon_length
        self.x_des_end_delim = self.N + 1
        self.n_x, self.n_u, self.n_d = n_x, n_u, n_d

        self.lti = lti
        if self.lti and self.integration_method == 'exact':
            self.fd_linear_func_exact = self.sys_dyn.fd_linear_func_exact
            self.fd_linear_func_exact_1ms = self.sys_dyn.fd_linear_func_exact_1ms

        self.Q, self.R = [np.array(x, ndmin=2) for x in [Q, R]]
        self.P = None   # Will be set based on linearization around terminal point.
                        # P gets defined as a parameter for the optimization problem that gets set at every timestep.

        if terminal_calc_fn == "linearized_lqr":
            partial_terminal_calc = functools.partial(linearize_for_terminal, sym_dyn_model=self.sys_dyn, n_u=self.n_u,
                                                      Q=self.Q, R=self.R)
            self.terminal_calc_fn = lambda lin_point_x: partial_terminal_calc(x_desired=lin_point_x)
        else:
            raise NotImplementedError

        self.add_scaling = add_scaling
        self.ignore_init_constr_check = ignore_init_constr_check

        self.ignore_cost = ignore_cost

        self.solver = solver
        self.periodic_callback_freq = periodic_callback_freq
        self.use_prev_if_infeas = use_prev_if_infeas

        base_opts = {"enable_fd": True}
        self.solver_opts = base_opts
        if addn_solver_opts is not None:
            self.solver_opts.update(addn_solver_opts)

        self.setup_OL_optimization()

    def set_traj_to_track(self, x_desired, u_desired=None):
        self.x_desired = x_desired
        if u_desired is not None:
            self.u_desired = u_desired

    def get_attrs_from_dict(self, vals):
        return [self.opti_dict[val] for val in vals]

    def get_info_for_traj_gen(self):
        return self.sys_dyn, self.Bd, self.gp_fns, self.gp_inputs, self.input_mask, self.N

    def cost_fn(self, mu_i_x, mu_i_u, x_desired, u_desired, idx=-1, terminal=False, Sigma_x=None, Sigma_u=None):
        if not terminal:
            x_des_dev, u_des_dev = (mu_i_x[:, idx] - x_desired[:, idx]), (mu_i_u[:, idx] - u_desired[:, idx])
            x_cost = x_des_dev.T @ self.Q @ x_des_dev
            u_cost = u_des_dev.T @ self.R @ u_des_dev
            return x_cost + u_cost
        else:
            x_des_dev = (mu_i_x[:, -1] - x_desired[:, -1])
            # Note self.P = self.opti_dict["P"] i.e. the parametrized terminal cost matrix that is set in closed loop.
            x_cost = x_des_dev.T @ self.P @ x_des_dev
            return x_cost

    def solve_optimization(self, ignore_initial=False, **kwargs):
        if not ignore_initial:
            self.set_initial(**kwargs)
        if kwargs.get('verbose', True):
            sol = self.opti.solve_limited()
        else:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                sol = self.opti.solve_limited()
        return sol

    def set_initial(self, **kwargs):
        # Set parameter values for initial state and desired state and input trajectory.
        self.opti.set_value(self.opti_dict["x_init"], kwargs.get('x_init'))
        self.opti.set_value(self.opti_dict["x_desired"], kwargs.get('x_desired', np.zeros([self.n_x, self.N + 1])))
        self.opti.set_value(self.opti_dict["u_desired"], kwargs.get('u_desired', np.zeros([self.n_u, self.N])))
        self.opti.set_value(self.opti_dict["P"], kwargs.get('P'))

        if kwargs.get('mu_x', None) is not None:
            mu_x_init = kwargs.get('mu_x')
            self.opti.set_initial(self.opti_dict["mu_x"], mu_x_init[:, :self.N + 1])
        if kwargs.get('mu_u', None) is not None:
            mu_u_init = kwargs.get('mu_u')
            self.opti.set_initial(self.opti_dict["mu_u"], mu_u_init[:, :self.N])

    def init_opti_means_n_params(self):
        opti = cs.Opti()
        self.opti = opti

        # Problem parameters
        x_init = opti.parameter(self.n_x, 1)
        x_desired = opti.parameter(self.n_x, self.N + 1)
        u_desired = opti.parameter(self.n_u, self.N)

        scale_x, scale_u = np.ones([self.n_x, 1]), np.ones([self.n_u, 1])
        # No need to replicate the scaling vector. Casadi implicitly takes care of this as seen by running
        # the stand-alone example below
        # opti = cs.Opti()
        # y = np.array([[4, 2, 1]]).T
        # x = y*opti.variable(3, 5)
        # print(x)
        if self.add_scaling:
            scale_x, scale_u = self.X.ub, self.U.ub

        # All mu_x, Sigma_x are optimization variables. mu_x array is nxN and mu_u array is mxN.
        mu_x = scale_x * opti.variable(self.n_x, self.N + 1)
        mu_u = scale_u * opti.variable(self.n_u, self.N)

        P = opti.parameter(self.n_x, self.n_x)

        return opti, mu_x, mu_u, x_init, x_desired, u_desired, P

    def run_ol_opt(self, initial_info_dict, x_desired=None):
        try:
            if x_desired is not None:
                self.set_traj_to_track(x_desired)
            self.init_warmstart(initial_info_dict)
            sol = self.solve_optimization(ignore_initial=True)
        except Exception as e:
            print(e)
        return sol

    def perform_forward_sim(self, mu_x_0, mu_u_0, mu_x_1):
        if self.integration_method == 'euler':
            sampled_ns = mu_x_0 + (self.dt * (self.ct_dyn_nom(x=mu_x_0, u=mu_u_0))['f'])
        elif self.integration_method == 'exact' and self.lti:
            sampled_ns = self.fd_linear_func_exact(x=mu_x_0, u=mu_u_0)['xf']
        try:
            assert np.isclose(np.array(sampled_ns).squeeze(), mu_x_1.squeeze(),
                              atol=1e-3).all(), "Sampled state is not close to the next state in the nominal trajectory"
        except AssertionError as e:
            print(e)
            print("sampled_ns %s" % sampled_ns)
            print("mu_x[:, 1] %s" % mu_x_1)
            # print(mu_x[:, 0])
            # print(mu_u[:, 0])
            # print(self.ct_dyn_nom)
            # print(np.isclose(sampled_ns, mu_x[:, 1], atol=1e-3))
        return sampled_ns

    def cl_warmstart(self, data_dict, x_desired, sim_step_idx, verbose):
        initialize_dict = {}

        initialize_dict["x_desired"] = x_desired[:, sim_step_idx + 1:sim_step_idx + 1 + self.x_des_end_delim]
        initialize_dict["u_desired"] = np.zeros((self.n_u, self.N))

        # Casadi squeezes variables that can be squeeze so an (N+1, 1) vector gets squeezed to (N+1) and hence the ndmin is required for the scalar case
        # without affecting planar or higher dim cases
        mu_x = np.array(data_dict['mu_x'], ndmin=2)
        mu_u = np.array(data_dict['mu_u'], ndmin=2)
        # Can only warmstart till penultimate state and input hence pad with 0s.
        initialize_dict['mu_x'] = np.hstack([mu_x[:, 1:], np.zeros((self.n_x, 1))])
        initialize_dict['mu_u'] = np.hstack([mu_u[:, 1:], np.zeros((self.n_u, 1))])

        mu_u = mu_u[:, 0]
        if self.use_prev_if_infeas:
            if self.last_feasible_idx - self.sim_step != 1:     # If this condition is not met it means the current solve was infeasible.
                                                                # Apply inputs and warmstart states from last feasible solve.
                idx_diff = self.sim_step - (self.last_feasible_idx - 1)
                initialize_dict['mu_u'] = np.hstack([self.last_feasible_mu_u[:, 1+idx_diff:], np.zeros((self.n_u, 1+idx_diff))])
                initialize_dict['mu_x'] = np.hstack([self.last_feasible_mu_x[:, 1+idx_diff:], np.zeros((self.n_x, 1+idx_diff))])
                mu_u = self.last_feasible_mu_u[:, 0+idx_diff]

        sampled_ns = self.perform_forward_sim(self.curr_state, mu_u, mu_x[:, 1])
        self.curr_state = sampled_ns

        initialize_dict['mu_x'][:, 0] = np.array(sampled_ns).squeeze()
        initialize_dict['x_init'] = sampled_ns
        # print("sampled_ns %s" % sampled_ns)
        initialize_dict['verbose'] = verbose
        initialize_dict['P'] = self.terminal_calc_fn(mu_x[:, -1])

        reqd_keys = ["x_init", "x_desired", "u_desired", "P", "mu_x", "mu_u"]
        for key in reqd_keys:
            assert key in initialize_dict.keys(), "Required key {} not found in initialize_dict".format(key)

        self.set_initial(**initialize_dict)

    def init_warmstart(self, initial_info_dict):
        reqd_keys = ["u_warmstart", "x_init"]
        for key in reqd_keys:
            assert key in initial_info_dict.keys(), "Required key {} not found in initialize_dict".format(key)

        initialize_dict = {"mu_x": self.x_desired[:, :self.N + 1], "mu_u": initial_info_dict["u_warmstart"],
                           'x_init': initial_info_dict['x_init'],
                           "x_desired": self.x_desired[:, :self.N + 1], "u_desired": np.zeros((self.n_u, self.N)),
                           "P": self.terminal_calc_fn(self.x_desired[:, self.N])}
        self.set_initial(**initialize_dict)

    def opti_infeas_debug_cbfn(self, iter_num, assert_violation=False):
        raise NotImplementedError

    def periodic_cl_callback(self, iter_num, runtime_error=False):
        raise NotImplementedError

    def run_cl_opt(self, initial_info_dict, simulation_length, opt_verbose=False, x_desired=None,
                   infeas_debug_cb=False):
        self.simulation_length = simulation_length
        self.sampled_ns = initial_info_dict["x_init"]
        self.opt_verbose = opt_verbose
        data_dicts = []
        sol_stats = []
        if x_desired is not None:
            self.set_traj_to_track(x_desired)

        self.curr_state = initial_info_dict["x_init"]
        self.init_warmstart(initial_info_dict)

        if self.use_prev_if_infeas:
            self.last_feasible_mu_x = self.x_desired[:, :self.N + 1]
            self.last_feasible_idx = 0
            if self.u_desired is not None:
                self.last_feasible_mu_u = self.u_desired[:, :self.N]


        for i in range(simulation_length):
            self.sim_step = i
            runtime_error = False
            try:
                print("Solving for timestep: %s" % i)
                sol = self.solve_optimization(ignore_initial=True, verbose=self.opt_verbose)
            except Exception as e:
                # Error is thrown after finishing computing solution before generating the warmstarts for the next iteration.
                # So all the warmstarting code for the next iteration is put in this block after the exception has been caught.
                print(e)
                runtime_error = True
            finally:
                infeasible_prob = False
                if self.use_prev_if_infeas:
                    try:
                        if runtime_error:
                            infeasible_prob = True
                        if sol.stats()["return_status"] in ["Infeasible_Problem_Detected", "Maximum_Iterations_Exceeded"]:
                            infeasible_prob = True
                        if not infeasible_prob:
                            print("t: %s, Ret. stat.: %s, System inputs: %s" % (i, sol.stats()["return_status"], self.opti.debug.value(self.opti_dict["mu_u"][:, 0])), end=" ")
                            self.last_feasible_mu_u = self.opti.debug.value(self.opti_dict["mu_u"])
                            self.last_feasible_mu_x = self.opti.debug.value(self.opti_dict["mu_x"])
                            self.last_feasible_idx = i+1
                    except UnboundLocalError:
                        infeasible_prob = True
                    if infeasible_prob:
                        idx_diff = self.sim_step - (self.last_feasible_idx - 1)
                        mu_u = self.last_feasible_mu_u[:, 0+idx_diff]
                        print("t: %s, Ret. stat.: %s, System inputs (from previous OL soln): %s" % (i, "Infeasible_Problem_Detected", mu_u), end=" ")

                    if infeas_debug_cb and sol.stats()["return_status"] == "Infeasible_Problem_Detected":
                        self.opti_infeas_debug_cbfn(i)

                    if not runtime_error:
                        mu_x_OL = self.opti.debug.value(self.opti_dict["mu_x"])
                        mu_u_OL = self.opti.debug.value(self.opti_dict["mu_u"])
                    else:
                        mu_x_OL = np.zeros((self.n_x, self.N+1))
                        mu_u_OL = np.zeros((self.n_u, self.N))
                    # replace first state with true state value
                    mu_x_OL[:, 0] = self.curr_state.squeeze()
                    if self.use_prev_if_infeas and infeasible_prob:
                        # replace first input with overriden input
                        mu_u_OL[:, 0] = self.last_feasible_mu_u[:, 0+idx_diff].squeeze()

                    if not infeasible_prob:
                        sol_stats.append(sol.stats())
                    data_dicts.append({"mu_x": mu_x_OL, "mu_u": mu_u_OL, "infeasible_prob": infeasible_prob, "runtime_error": runtime_error})
                else:
                    infeasible_prob = False
                    try:
                        print("t: %s, Ret. stat.: %s, System inputs: %s" % (i, sol.stats()["return_status"], self.opti.debug.value(self.opti_dict["mu_u"][:, 0])), end=" ")
                    except UnboundLocalError:
                        print("t: %s, Ret. stat.: %s" % (i, "Infeasible_Problem_Detected"), end=" ")
                        infeasible_prob = True
                    if infeas_debug_cb and sol.stats()["return_status"] == "Infeasible_Problem_Detected":
                        self.opti_infeas_debug_cbfn(i)

                    mu_x_OL = self.opti.debug.value(self.opti_dict["mu_x"])
                    mu_u_OL = self.opti.debug.value(self.opti_dict["mu_u"])

                    if not infeasible_prob:
                        sol_stats.append(sol.stats())
                    data_dicts.append({"mu_x": mu_x_OL, "mu_u": mu_u_OL})

                # Periodic callback BEFORE warmstarting just in case any of the variables (ex: deltas for inherited controller)
                # need to be updated before warmstarting.
                if self.periodic_callback_freq is not None:
                    self.periodic_cl_callback(iter_num=i, runtime_error=runtime_error)

                self.cl_warmstart(data_dict=data_dicts[-1], x_desired=self.x_desired, sim_step_idx=i,
                                  verbose=self.opt_verbose)

        return data_dicts, sol_stats

    def respect_constraint_sets(self, mu_x, mu_u, k):
        self.opti.subject_to(self.X.H_np @ mu_x[:, k + 1] - self.X.b_np <= 0)
        self.opti.subject_to(self.U.H_np @ mu_u[:, k] - self.U.b_np <= 0)

    def setup_initial_state_constr(self, mu_x_0, x_init):
        self.opti.subject_to(mu_x_0 - x_init == 0)
        # We add a config option to neglect this constraint if working with a tracking case where we expect constraint violation
        if not self.ignore_init_constr_check:
            self.opti.subject_to(self.X.H_np @ mu_x_0 - self.X.b_np <= 0)

    def setup_cost_fn_and_shrinking(self):
        # Retrieve variables
        mu_x, mu_u, x_desired, u_desired, x_init = self.get_attrs_from_dict(
            ['mu_x', 'mu_u', 'x_desired', 'u_desired', 'x_init'])

        # Initial state constraints.
        self.setup_initial_state_constr(mu_x[:, 0], x_init)
        # Obeying constraint sets at all timesteps
        for k in range(self.N):
            # Restrict state and input vectors to either be within original specified constraints (if skip shrinking=True)
            # or within constraint sets shrunk according to CC-MPC results.
            self.respect_constraint_sets(mu_x, mu_u, k=k)

        self.setup_cost_metric()

    def setup_cost_metric(self):
        mu_x, mu_u, x_desired, u_desired, x_init = self.get_attrs_from_dict(
            ['mu_x', 'mu_u', 'x_desired', 'u_desired', 'x_init'])
        # Cost function stuff
        cost = 0
        # Stage cost
        if not self.ignore_cost:
            for k in range(self.N):
                cost += self.cost_fn(mu_i_x=mu_x, mu_i_u=mu_u,
                                     x_desired=x_desired, u_desired=u_desired, idx=k)
            # Terminal cost
            cost += self.cost_fn(mu_x[:, -1], None, x_desired=x_desired, u_desired=None,
                                 terminal=True)
        self.opti.minimize(cost)

    def setup_solver(self):
        acceptable_dual_inf_tol = 1e3
        acceptable_compl_inf_tol = 1e-1
        acceptable_iter = 5
        acceptable_constr_viol_tol = 5*1e-2
        acceptable_tol = 1e5
        max_iter = 100

        self.solver_opts["error_on_fail"] = False
        additional_opts = {"acceptable_tol": acceptable_tol, "acceptable_constr_viol_tol": acceptable_constr_viol_tol,
                           "acceptable_dual_inf_tol": acceptable_dual_inf_tol, "acceptable_iter": acceptable_iter,
                           "acceptable_compl_inf_tol": acceptable_compl_inf_tol, "max_iter": max_iter,
                           "hessian_approximation": "limited-memory"}
        self.opti.solver(self.solver, self.solver_opts, additional_opts)

    def setup_dynamics_constraints(self, k, mu_x, mu_u, mean_vec=None):
        if self.integration_method == 'euler':
            self.opti.subject_to(
                mu_x[:, k + 1] == (mu_x[:, k] + self.dt * (self.ct_dyn_nom(x=mu_x[:, k], u=mu_u[:, k])['f'])))
        elif self.integration_method == 'exact' and self.lti:
            self.opti.subject_to(
                mu_x[:, k + 1] == self.fd_linear_func_exact(x=mu_x[:, k], u=mu_u[:, k])['xf'])
        else:
            raise NotImplementedError

    def setup_OL_optimization(self):
        self.opti, mu_x, mu_u, x_init, x_desired, u_desired, self.P = self.init_opti_means_n_params()

        for k in range(self.N):
            # State mean dynamics
            self.setup_dynamics_constraints(k, mu_x, mu_u)

        # We need to have a way to set x_init, x_desired etc. at every iteration of the closed loop optimization. This dict will maintain
        # references to variables/parameters contained in the "opti" instance and set values for parameters/provide warm start solutions for variables
        self.opti_dict = {"mu_x": mu_x, "mu_u": mu_u, "x_init": x_init, "x_desired": x_desired, "u_desired": u_desired,
                          "P": self.P}

        # Setup cost function to minimize during optimization along with shrunk set constraints.
        self.setup_cost_fn_and_shrinking()
        self.setup_solver()


class GPMPC_BnC(BaseGPMPC_sanity):
    def __init__(self, gp_fns, regions, region_subset_lim_lb, region_subset_lim_ub, delta_input_mask, gp_input_mask, Bd,
                 sys_dyn, Q, R, horizon_length, X: box_constraint, U: box_constraint, n_x, n_u, n_d,
                 add_delta_constraints=True, terminal_calc_fn="linearized_lqr", solver='ipopt', addn_solver_opts=None,
                 ignore_init_constr_check=False, add_scaling=False, sampling_time=20 * 10e-3,
                 integration_method='euler', ignore_callback=False, fwd_sim="nom_dyn", infeas_debug_callback=False,
                 periodic_callback_freq=None, collision_check=False, **kwargs):
        self.gp_fns = gp_fns
        self.delta_input_mask = delta_input_mask
        self.gp_input_mask = gp_input_mask
        self.Bd = Bd
        self.add_delta_constraints = add_delta_constraints
        self.infeas_debug_callback = infeas_debug_callback

        self.collision_check = collision_check
        self.collision_counter = 0
        self.collision_counter_limited = 0
        self.collision_idxs = []

        self.regions = regions
        self.num_regions = len(self.regions)
        self.region_subset_lim_lb, self.region_subset_lim_ub = region_subset_lim_lb, region_subset_lim_ub
        self.region_Hs, self.region_bs = [region.H_np for region in self.regions], [region.b_np for region in
                                                                                    self.regions]
        self.fwd_sim = fwd_sim
        if self.fwd_sim == "w_pw_res":
            assert kwargs.get("true_ds_inst",
                              None) is not None, "Need to provide true ds inst (GP_DS) for w_pw_res type fwd_sim"
            assert kwargs.get("Bd_fwd_sim", None) is not None, "Need to provide Bd_fwd_sim for w_pw_res type fwd_sim"
            self.true_ds_inst = kwargs.get("true_ds_inst")
            self.Bd_fwd_sim = kwargs.get("Bd_fwd_sim")

        super().__init__(sys_dyn, Q, R, horizon_length, X, U, n_x, n_u, n_d, terminal_calc_fn, solver, addn_solver_opts,
                         ignore_init_constr_check, add_scaling, sampling_time, integration_method, ignore_callback, periodic_callback_freq,
                         **kwargs)

    def get_vals_from_opti_debug(self, var_name):
        assert var_name in self.opti_dict.keys(), "Controller's opti_dict has no key: %s . Add it to the dictionary within the O.L. setups" % var_name
        if type(self.opti_dict[var_name]) in [list, tuple]:
            return [self.opti.debug.value(var_k) for var_k in self.opti_dict[var_name]]
        else:
            return self.opti.debug.value(self.opti_dict[var_name])

    def setup_hybrid_means_manual_preds(self, gp_inp_vec_size, gp_input):
        models = self.gp_fns.models
        # Instead of computing hybrid means 1 timestep at a time, we compute hybrid means over the entire trajectory for a given
        # model at one shot. Allows us to make use of efficiency with batch computing to speed up run-time. mu_d_pw_r is thus
        # representative of the residual terms generated by region r over the entire O.L. horizon.
        mu_d_pw_r = [cs.MX.sym('mu_d_pw', self.n_d, self.N) for r in range(self.num_regions)]  # placeholder
        gp_inp_OL = cs.MX.sym('gp_inp_k', gp_inp_vec_size, self.N)
        # Store functions in list to prevent garbage collection. Casadi needs to have references to these functions for gradient
        # computation and if they get garbage collected these refs will be invalid.
        res_compute_fns = []
        for r in range(self.num_regions):
            get_hyb_res_mean = cs.Function('compute_resmean_region_r', [gp_inp_OL],
                                           [models[r].cs_predict_mean(gp_inp_OL)],
                                           ['gp_inp_OL'], ['mu_d_hyb'])
            res_compute_fns.append(get_hyb_res_mean)
            if r == 0:
                assert get_hyb_res_mean(gp_inp_OL).shape == mu_d_pw_r[0].shape, print(get_hyb_res_mean(gp_inp_OL).shape,
                                                                                      mu_d_pw_r[0].shape)
            mu_d_pw_r[r] = get_hyb_res_mean(gp_input)
        # Reshape hybrid means from all regions over OL horizon into the form expected by the get_mu_d cs Function.
        # Each element in the below array has shape (n_d, num_regions)
        mu_d_pw_arr = [cs.horzcat(*[mu_d_pw_r[r][:, k] for r in range(self.num_regions)]) for k in range(self.N)]
        return res_compute_fns, mu_d_pw_arr

    def compute_mu_d_k(self, k, region_deltas_k, mu_d_arr=None, mu_d_pw_arr=None):
        # hybrid_means = mu_d_pw_arr[k]
        # mean_vec = self.get_mu_d(hybrid_means, region_deltas_k)
        # # Defining mu_d as an opt var helps improve sparsity pattern and empirically seems to improve run-time
        # self.opti.subject_to(mean_vec - mu_d_arr[:, k] == 0)
        # return mean_vec

        # Alternate:
        self.opti.subject_to(self.get_mu_d(mu_d_pw_arr[k], region_deltas_k) - mu_d_arr[:, k] == 0)
        return None

    def setup_delta_control_inputs(self, mu_z):
        delta_inp_dim = np.linalg.matrix_rank(self.delta_input_mask)
        delta_controls = cs.MX.zeros(delta_inp_dim, self.N)
        temp_z = cs.MX.sym('temp_z', self.n_x + self.n_u, 1)

        # TODO: For loop unnecessary here. Set up as "batch" matrix mult and remove.
        for k in range(self.N):
            # Extract the variables of the state (or joint state and input) vector to be used for checking the delta conditions
            f = cs.Function('select_delta_controllers', [temp_z], [self.delta_input_mask @ temp_z],
                            {"enable_fd": True})
            delta_controls[:, k] = f(mu_z[:, k])
        return delta_controls

    def create_gp_inp_sym(self, mu_x, mu_u):
        mu_z = cs.vertcat(mu_x[:, :-1], mu_u)
        # Mean function of GP takes (x, u) as input and outputs the mean and cov of the residual term for the current timestep.
        gp_inp_vec_size = np.linalg.matrix_rank(self.gp_input_mask)
        gp_input = cs.MX.zeros(gp_inp_vec_size, self.N)
        temp_z = cs.MX.sym('temp_z', self.n_x + self.n_u, 1)
        try:
            gp_input_fn = cs.Function('gp_input', [temp_z], [self.gp_input_mask @ temp_z], {"enable_fd": True})
            for k in range(self.N):
                gp_input[:, k] = gp_input_fn(mu_z[:, k])
        except TypeError:
            print(temp_z.shape, self.gp_input_mask.shape)

        return mu_z, gp_inp_vec_size, gp_input

    def define_delta_vars(self):
        region_deltas = self.opti.parameter(len(self.regions), self.N + 1)
        return region_deltas

    def define_pw_cb_fns(self, test_softplus=False):
        # This is the region_deltas array. Since we're going to be trying to track an RRT* generated path, the deltas will be set in advance
        # and we'll be trying to maintain the same deltas if a feasible path exists. As a result, they are parameters and not variables.
        # TODO: For now we only assume that we work with regions based on state and hence can take in deltas up to timestep N+1. In the future when using
        # nominal MPC to generate nominal control inputs for warmstarting, we'll need to change this array to be N dimensional and allow the final
        # state to lie in any region.
        region_deltas = self.define_delta_vars()

        # Piecewise functions to extract the right mean and covariance based on the delta vector at the current timestep
        hybrid_means_sym = cs.MX.sym('hybrid_means_sym', self.n_d, self.num_regions)
        delta_k_sym = cs.MX.sym('delta_k_sym', self.num_regions, 1)

        # No need for any delta tolerances here because we're given 1, 0 deltas and the covariances are abstracted away since
        # using parametrized shrunk vectors.
        self.get_mu_d = cs.Function('get_mu_d', [hybrid_means_sym, delta_k_sym],
                                    [(delta_k_sym.T @ hybrid_means_sym.T).T],
                                    {'enable_fd': True})

        # Select the H and b matrix of the right region at the current timestep to dynamically change what the constraints on the
        # joint state-input vector are that realize the delta assignments we generate using RRT*.
        H_mat_shape, b_vec_shape = self.regions[0].H_np.shape, self.regions[0].b_np.shape
        n_in_rows, n_in_cols = H_mat_shape[0], H_mat_shape[1]
        self.select_H = delta_matrix_mult('H_mat_sel', n_in_rows, n_in_cols, self.num_regions,
                                          self.N, opts={'enable_fd': True})
        n_in_rows, n_in_cols = b_vec_shape[0], b_vec_shape[1]
        self.select_b = delta_matrix_mult('b_vec_sel', n_in_rows, n_in_cols, self.num_regions,
                                          self.N, opts={'enable_fd': True})

        return region_deltas

    def setup_paramd_delta_satisfaction(self, k, region_deltas, delta_controls, ret_H_b=False):
        seld_region_H, seld_region_b = self.select_H(region_deltas[:, [k]], *self.region_Hs), self.select_b(
            region_deltas[:, [k]], *self.region_bs)
        # print(delta_controls[:, [k]].shape)
        self.opti.subject_to((seld_region_H @ delta_controls[:, [k]]) - seld_region_b <= 0)
        if ret_H_b:
            return seld_region_H, seld_region_b

    def setup_dynamics_constraints(self, k, mu_x, mu_u, mu_d):
        if self.integration_method == 'euler':
            self.opti.subject_to(mu_x[:, k + 1] - (mu_x[:, k] + self.dt * (
                        self.ct_dyn_nom(x=mu_x[:, k], u=mu_u[:, k])['f'] + self.Bd @ mu_d[:, k])) == 0)
        elif self.integration_method == 'exact':
            self.opti.subject_to(mu_x[:, k + 1] -
                                 (self.fd_linear_func_exact(x=mu_x[:, k], u=mu_u[:, k])['xf'] + self.dt * self.Bd @ mu_d[:, k]) == 0)
        else:
            raise NotImplementedError

    def res_mean_constraints(self, mu_x, mu_u, x_init, x_desired, u_desired, test_softplus=False):
        mu_d = self.opti.variable(self.n_d, self.N)
        mean_vecs = []
        # Extract variables from joint state-input vector that must be passed to the GP to compute the residual means
        mu_z, gp_inp_vec_size, gp_input = self.create_gp_inp_sym(mu_x, mu_u)
        # Compute means for all GPs in the hybrid model which will henceforth be used by get_mu_d to extract the right means
        # by multiplying with the delta vector
        res_compute_fns, mu_d_pw_arr = self.setup_hybrid_means_manual_preds(gp_inp_vec_size, gp_input)

        # Extract variables from joint state-input vector that determine which mode of the hybrid GP model is active.
        delta_controls = self.setup_delta_control_inputs(mu_z)

        # Initial condition constraints. Assumption: x0 is known with complete certainty.
        self.setup_initial_state_constr(mu_x[:, 0], x_init)

        # Setup callback functions to select right regions based on RRT* delta assignment and also compute the piecewise
        # residual mean value.
        self.nx_before_deltas = self.opti.nx
        region_deltas = self.define_pw_cb_fns(test_softplus=test_softplus)

        delta_debug_test = False
        if delta_debug_test:
            seld_H_arr, seld_b_arr = [], []
            self.opti.callback(lambda i: self.delta_debug_callback(i))

        testing_mu_d = False
        for k in range(self.N):
            mean_vec = self.compute_mu_d_k(k, region_deltas[:, [k]], mu_d, mu_d_pw_arr=mu_d_pw_arr)
            mean_vecs.append(mean_vec)

            # Set up constraints to make sure that the deltas passed in from RRT* are obeyed.
            if self.add_delta_constraints:
                ops = self.setup_paramd_delta_satisfaction(k, region_deltas, delta_controls, ret_H_b=delta_debug_test)
                if delta_debug_test:
                    seld_H_arr.append(ops[0])
                    seld_b_arr.append(ops[1])

            # State mean dynamics
            self.setup_dynamics_constraints(k, mu_x, mu_u, mu_d)

        # We need to have a way to set x_init, x_desired etc. at every iteration of the closed loop optimization. This dict will maintain
        # references to variables/parameters contained in the "opti" instance and set values for parameters/provide warm start solutions for variables
        self.opti_dict = {
            "mu_x": mu_x,
            "mu_u": mu_u,
            "mu_z": mu_z,
            "delta_controls": delta_controls,  # Function of mu_z which is a function of mu_x, mu_u and hence it contains the right dependencies.
            "x_init": x_init,
            "x_desired": x_desired,
            "u_desired": u_desired,
            "gp_input": gp_input,
            "mu_d": mu_d,
            "mean_vecs": mean_vecs,
            "mu_d_pw_arr": mu_d_pw_arr
        }
        self.opti_dict.update({"hld": region_deltas})
        # Add terminal cost matrix to opti dict to allow for setting of the parametrized values obtained after linearizing about the last point
        # of the trajectory to track over the given horizon.
        self.opti_dict["P"] = self.P
        if delta_debug_test:
            self.opti_dict.update({"seld_H": seld_H_arr, "seld_b": seld_b_arr})

    def setup_OL_optimization(self):
        opti, mu_x, mu_u, x_init, x_desired, u_desired, self.P = self.init_opti_means_n_params()

        self.res_mean_constraints(mu_x, mu_u, x_init, x_desired, u_desired)

        # Setup cost function to minimize during optimization along with shrunk set constraints.
        self.setup_cost_fn_and_shrinking()
        self.setup_solver()

        if self.infeas_debug_callback:
            self.opti.callback(lambda i: self.opti_infeas_debug_cbfn(i))

    def delta_debug_callback(self, i):
        if i == 5:
            mu_x_curr = self.opti.debug.value(self.opti_dict["mu_x"])
            seld_H_arr = [self.opti.debug.value(self.opti_dict["seld_H"][i]) for i in
                          range(len(self.opti_dict["seld_H"]))]
            seld_b_arr = [self.opti.debug.value(self.opti_dict["seld_b"][i]) for i in
                          range(len(self.opti_dict["seld_b"]))]

            steps = mu_x_curr.shape[-1]
            temp_hld_arr = np.zeros([self.num_regions, steps])
            for sim_step in range(steps):
                # 0 pad the inputs since we have already checked the deltas don't depend on them using the assertion above.
                joint_vec = np.vstack([mu_x_curr[:, [sim_step]], np.zeros([self.n_u, 1])]).squeeze()
                for region_idx, region in enumerate(self.regions):
                    temp_hld_arr[region_idx, sim_step] = 1 if region.check_satisfaction(
                        (self.delta_input_mask @ joint_vec).T).item() is True else 0
                    try:
                        assert temp_hld_arr[region_idx, sim_step] == (1 if (
                                    self.region_Hs[region_idx] @ np.array(self.delta_input_mask @ joint_vec,
                                                                          ndmin=2).T <= self.region_bs[
                                        region_idx]).all() else 0)
                    except AssertionError as e:
                        print("Sim step with error: %s" % sim_step)
                        print("Region idx: %s" % region_idx)
                        print(temp_hld_arr[region_idx, sim_step])
                        print("H comparison")
                        print(self.region_Hs[region_idx], seld_H_arr[sim_step])
                        print("b comparison")
                        print(self.region_bs[region_idx], seld_b_arr[sim_step])
                        print("control vec")
                        print(self.delta_input_mask @ joint_vec)
                        print(self.region_Hs[region_idx] @ (self.delta_input_mask @ joint_vec).squeeze())
                        print(self.region_Hs[region_idx] @ (self.delta_input_mask @ joint_vec) <= self.region_bs[
                            region_idx])
                try:
                    print("step: ", sim_step)
                    print(seld_H_arr[sim_step], seld_b_arr[sim_step])
                except IndexError:
                    pass
            print(mu_x_curr[[0, 2], :])
            print(temp_hld_arr)

    def cb_debug_order_checking(self, mu_x_size, mu_u_size, mu_d_size):
        print(self.opti_dict["mu_x"])
        print(self.opti.x[0])
        print(self.opti.x[mu_x_size - 1])
        print(self.opti_dict["mu_u"])
        print(self.opti.x[mu_x_size])
        print(self.opti.x[mu_x_size + mu_u_size - 1])
        print(self.opti_dict["mu_d"])
        print(self.opti.x[mu_x_size + mu_u_size])
        print(self.opti.x[mu_x_size + mu_d_size + mu_u_size - 2])

    def gen_x_p_for_debug(self, x_init, opt_x_infeas, opt_u_infeas, opt_d_infeas, order_check=False):
        # Note: All arrays need to be flattened in a certain order for them to match the order in which they have been added to the opti inst.
        mu_x_size, mu_u_size, mu_d_size = self.n_x * (self.N + 1), self.n_u * self.N, self.n_d * (self.N + 1)
        x_des_shape, u_des_shape = np.zeros(mu_x_size), np.zeros(mu_u_size)
        if order_check:
            self.cb_debug_order_checking(mu_x_size, mu_u_size, mu_d_size)
        P_mat_shape = np.zeros((self.n_x * self.n_x))
        hld_mat_shape = np.zeros((self.num_regions * (self.N + 1)))
        # parametric vectors are just placeholders. We are not checking parametric sensitivities here but we do need to pass them to the
        # function f created in self.opti_infeas_debug_cbfn and so the size needs to be correct.
        p = np.concatenate((x_init.flatten(order='F'), x_des_shape, u_des_shape, P_mat_shape, hld_mat_shape))
        x = np.hstack(
            [opt_x_infeas.flatten(order='F'), opt_u_infeas.flatten(order='F'), opt_d_infeas.flatten(order='F')])
        # Proving correct order to flatten
        assert (np.array(cs.vec(opt_d_infeas)).squeeze() == opt_d_infeas.flatten(
            order='F')).all(), "opt_d_infeas is not flattened in the correct order"
        # print(cs.vec(opt_u_infeas).shape, cs.vec(opt_x_infeas).shape, cs.vec(opt_d_infeas).shape)
        x = cs.vertcat(cs.vec(opt_x_infeas), cs.vec(opt_u_infeas), cs.vec(opt_d_infeas))
        # print(x.shape)

        assert x.shape[
                   0] == self.opti.nx, "x vector for debugging is not the correct size. x.shape[0]: %s, opti.nx: %s" % (
        x.shape[0], self.opti.nx)
        assert p.shape[
                   0] == self.opti.np, "p vector for debugging is not the correct size. p.shape[0]: %s, opti.np: %s" % (
        p.shape[0], self.opti.np)
        return x, p

    def opti_infeas_debug_cbfn(self, iter_num, assert_violation=False):
        cb_freq = 5
        order_check = True
        if iter_num // cb_freq == 0:
            return
        print("Checking violation")
        x_init = self.opti.debug.value(self.opti_dict["x_init"])
        opt_x_infeas = self.opti.debug.value(self.opti_dict["mu_x"])
        opt_u_infeas = self.opti.debug.value(self.opti_dict["mu_u"])
        opt_d_infeas = self.opti.debug.value(self.opti_dict["mu_d"])
        opt_mean_vec = self.get_vals_from_opti_debug("mean_vecs")
        mu_d_pw_arr = self.get_vals_from_opti_debug("mu_d_pw_arr")
        print(opt_d_infeas)
        print(opt_mean_vec)
        print(opt_d_infeas - opt_mean_vec)
        # print(mu_d_pw_arr)
        x, p = self.gen_x_p_for_debug(x_init, opt_x_infeas, opt_u_infeas, opt_d_infeas, order_check=order_check)
        print(x, p)
        f = cs.Function('f', [self.opti.x, self.opti.p], [self.opti.g])
        constr_viol_split = cs.vertsplit(f(x, p), 1)
        print(constr_viol_split)
        eq_constraint_idxs = []
        ineq_constraint_idxs = []
        constr_viol = 0
        num_viols = 0
        for i in range(self.opti.ng):
            if '==' in self.opti.debug.g_describe(i):
                if np.abs(constr_viol_split[i]) >= 1e-5:
                    print("Error at iteration %s" % iter_num)
                    print(i, self.opti.debug.g_describe(i))
                    print(constr_viol_split[i])
                    num_viols += 1
                if assert_violation:
                    assert np.abs(constr_viol_split[i]) <= 1e-5, "%s %s %s" % (
                    i, self.opti.debug.g_describe(i), constr_viol_split[i])
                eq_constraint_idxs.append(i)
                constr_viol += np.abs(constr_viol_split[i])
            else:
                ineq_constraint_idxs.append(i)
                if assert_violation:
                    assert constr_viol_split[i] <= 0
        if constr_viol <= 1e-4:
            print("All constraints passed with total violation less than 1e-4")
        print("Total constraint violation: %s" % constr_viol)
        print("Number of violations: %s" % num_viols)

    def clip_to_region_limits(self, inp_arr):
        inp_arr = np.array(inp_arr, ndmin=2).reshape(-1, 1)
        delta_var_idxs = [np.nonzero(self.delta_input_mask[k, :])[0][0] for k in range(self.delta_input_mask.shape[0])]
        inp_arr_clipped = copy.deepcopy(inp_arr)
        inp_arr_clipped[delta_var_idxs, :] = np.clip(
            self.delta_input_mask @ np.vstack([inp_arr, np.zeros((self.n_u, inp_arr.shape[-1]))]),
            self.region_subset_lim_lb, self.region_subset_lim_ub)
        return inp_arr_clipped

    def clip_x_des(self):
        delta_var_idxs = [np.nonzero(self.delta_input_mask[k, :])[0][0] for k in range(self.delta_input_mask.shape[0])]
        x_desired_clipped = copy.deepcopy(self.x_desired)
        x_desired_clipped[delta_var_idxs, :] = np.clip(
            self.delta_input_mask @ np.vstack([self.x_desired, np.zeros((self.n_u, self.x_desired.shape[-1]))]),
            self.region_subset_lim_lb, self.region_subset_lim_ub)
        return x_desired_clipped

    def generate_hld_for_warmstart(self):
        num_steps = self.x_desired.shape[-1]
        x_desired_clipped = self.clip_x_des()
        self.hld_arr = np.zeros([self.num_regions, num_steps])
        for sim_step in range(num_steps):
            # 0 pad the inputs since we have already checked the deltas don't depend on them using the assertion above.
            joint_vec = np.vstack([x_desired_clipped[:, [sim_step]], np.zeros([self.n_u, 1])])
            for region_idx, region in enumerate(self.regions):
                self.hld_arr[region_idx, sim_step] = 1 if region.check_satisfaction(
                    (self.delta_input_mask @ joint_vec).T).item() is True else 0
            assert np.sum(self.hld_arr[:,
                          sim_step]) == 1, "Delta assignment %s is not unique for the given input. Please check the input and try again." % self.hld_arr[:,sim_step]

    def run_collision_update(self):
        print("Collision detected")
        self.collision_counter += 1
        self.collision_idxs.append(self.sim_step)
        ## TODO: Generalize. Currently limited to 4 waypoint boundary tracking case.
        if self.sim_step in list(range(24, 50)) or self.sim_step in list(range(74, 100)):
            self.collision_counter_limited += 1

    def perform_forward_sim(self, mu_x_0, mu_u_0, mu_x_1, no_noise=False):
        if self.fwd_sim == "nom_dyn":
            sampled_ns = super().perform_forward_sim(mu_x_0, mu_u_0, mu_x_1)
        elif self.fwd_sim == "w_pw_res":
            dt_dyn_nom = None if (self.integration_method != "exact" or not self.lti) else self.fd_linear_func_exact
            fine_simulate = True
            if fine_simulate:
                if self.integration_method == "exact" and self.lti:
                    dt_dyn_nom = self.fd_linear_func_exact_1ms
                collision_bool = False
                start_state = self.curr_state
                fine_disc_time = 1e-3
                num_integrations = int(self.dt // fine_disc_time)
                for i in range(num_integrations):
                    # Necessary otherwise slight error tolerance in collision detector clipping function prevents collisions
                    # from being yielded when there was an actual collision.
                    # if i == num_integrations - 1:
                    #     clip_to_region_fn = lambda x: self.clip_to_region_limits(x)
                    # else:
                    #     clip_to_region_fn = None
                    clip_to_region_fn = lambda x: self.clip_to_region_limits(x)
                    if i == num_integrations - 1:
                        clip_state = True
                    else:
                        clip_state = False
                    ops = fwdsim_w_pw_res(clip_to_region_fn=clip_to_region_fn, clip_state=clip_state,
                                          true_func_obj=self.true_ds_inst,
                                          ct_dyn_nom=self.ct_dyn_nom, dt=fine_disc_time,
                                          Bd=self.Bd_fwd_sim, gp_input_mask=self.gp_input_mask,
                                          delta_input_mask=self.delta_input_mask,
                                          x_0=start_state, u_0=mu_u_0, ret_residual=False,
                                          no_noise=no_noise, ret_collision_bool=self.collision_check,
                                          integration_method=self.integration_method, dt_dyn_nom=dt_dyn_nom)  # Don't include noise for this increment since not shrinking based on variance outputs.
                    start_state, collision_bool_temp = ops
                    # print(start_state, end=" ")
                    if collision_bool_temp is True:
                        collision_bool = True
                sampled_ns = start_state
                if self.collision_check:
                    if collision_bool:
                        self.run_collision_update()
            else:
                ops = fwdsim_w_pw_res(clip_to_region_fn=lambda x: self.clip_to_region_limits(x),
                                      true_func_obj=self.true_ds_inst,
                                      ct_dyn_nom=self.ct_dyn_nom, dt=self.dt,
                                      Bd=self.Bd_fwd_sim, gp_input_mask=self.gp_input_mask,
                                      delta_input_mask=self.delta_input_mask,
                                      x_0=self.sampled_ns, u_0=mu_u_0, ret_residual=False,
                                      no_noise=no_noise, ret_collision_bool=self.collision_check,
                                      integration_method=self.integration_method, dt_dyn_nom=dt_dyn_nom)  # Don't include noise for this increment since not shrinking based on variance outputs.
                if self.collision_check:
                    sampled_ns, collision_bool = ops
                    if collision_bool:
                        self.run_collision_update()
                else:
                    sampled_ns = ops[0]
                # temp_compare = super().perform_forward_sim(mu_x_0, mu_u_0, mu_x_1)
                # print("perform_forward_sim_test")
                # print(temp_compare, sampled_ns)
            self.sampled_ns = sampled_ns  # Initial value set in run_cl_opt in Sanity class
        else:
            raise NotImplementedError
        return sampled_ns

    def cl_warmstart(self, data_dict, x_desired, sim_step_idx, verbose):
        super().cl_warmstart(data_dict, x_desired, sim_step_idx, verbose)
        self.hld_init(sim_step=sim_step_idx)
        # self.opti.set_value(self.opti_dict["hld"],
        #                     self.hld_arr[:, sim_step_idx + 1:sim_step_idx + 1 + self.x_des_end_delim])
        self.mu_d_init(sim_step=sim_step_idx)
        # self.opti.set_initial(self.opti_dict["mu_d"], self.resmean_warmstart[:, sim_step_idx + 1: sim_step_idx + 1 + self.N])

    def init_warmstart(self, initial_info_dict):
        super().init_warmstart(initial_info_dict)
        self.hld_init(sim_step=None)
        # self.opti.set_value(self.opti_dict["hld"], self.hld_arr[:, :self.N + 1])
        self.mu_d_init(sim_step=None)
        # self.opti.set_initial(self.opti_dict["mu_d"], self.resmean_warmstart[:, :self.N])

    def hld_init(self, sim_step=None):
        if sim_step is None:
            self.opti.set_value(self.opti_dict["hld"], self.hld_arr[:, :self.N + 1])
        else:
            self.opti.set_value(self.opti_dict["hld"], self.hld_arr[:, sim_step + 1:sim_step + 1 + self.N + 1])

    def mu_d_init(self, sim_step=None):
        if sim_step is None:
            self.opti.set_initial(self.opti_dict["mu_d"], self.resmean_warmstart[:, :self.N])
        else:
            self.opti.set_initial(self.opti_dict["mu_d"], self.resmean_warmstart[:, sim_step + 1: sim_step + 1 + self.N])

    def setup_resmean_warmstart_vec(self):
        self.resmean_warmstart = np.zeros((self.n_d, self.u_desired.shape[1]))
        for k in range(self.u_desired.shape[1]):
            # 0 pad the inputs since we have already checked the deltas don't depend on them using the assertion above.
            joint_vec = np.vstack([self.x_desired[:, [k]], np.zeros([self.n_u, 1])]).squeeze()
            mu_x, mu_u = self.opti_dict["mu_x"], self.opti_dict["mu_u"]
            _, gp_inp_vec_size, gp_input = self.create_gp_inp_sym(mu_x, mu_u)
            mean_res_compute_fns, _ = self.setup_hybrid_means_manual_preds(gp_inp_vec_size, gp_input)
            gp_inp_vec = np.array(self.gp_input_mask @ joint_vec, ndmin=2).reshape((-1, 1))
            for region_idx, region in enumerate(self.regions):
                mu_d_r_k = mean_res_compute_fns[region_idx](gp_inp_vec)[0]
                self.resmean_warmstart[:, [k]] = self.resmean_warmstart[:, [k]] + self.hld_arr[region_idx, k] * mu_d_r_k

    def run_cl_opt(self, initial_info_dict, simulation_length, opt_verbose=False, x_desired=None, u_desired=None,
                   infeas_debug_callback=False):
        if x_desired is not None:
            self.set_traj_to_track(x_desired, u_desired=u_desired)
        self.generate_hld_for_warmstart()
        self.setup_resmean_warmstart_vec()

        data_dicts, sol_stats = super().run_cl_opt(initial_info_dict, simulation_length, opt_verbose,
                                                   infeas_debug_cb=infeas_debug_callback)
        return data_dicts, sol_stats

    def run_ol_opt(self, initial_info_dict, x_desired=None, u_desired=None):
        if x_desired is not None:
            self.set_traj_to_track(x_desired, u_desired=u_desired)
        self.generate_hld_for_warmstart()
        # print("X DESIRED (x-z)")
        # print(self.x_desired[[0, 2], :])
        # print("GENERATED HLD ARR")
        # print(self.hld_arr)
        return super().run_ol_opt(initial_info_dict, x_desired=x_desired)


class GPMPC_D(GPMPC_BnC):
    def __init__(self, satisfaction_prob, gp_fns, regions, region_subset_lim_lb, region_subset_lim_ub, delta_input_mask,
                 gp_input_mask, Bd, sys_dyn, Q, R, horizon_length, X: box_constraint, U: box_constraint, n_x, n_u, n_d,
                 add_delta_constraints=True, terminal_calc_fn="linearized_lqr", solver='ipopt', addn_solver_opts=None,
                 ignore_init_constr_check=False, add_scaling=False, sampling_time=20 * 10e-3, state_plot_idxs=None,
                 integration_method='euler', ignore_callback=False, fwd_sim="nom_dyn", infeas_debug_callback=False,
                 skip_shrinking=False, skip_feedback=True, shrinking_approach="shrink_by_desired", K=None,
                 periodic_callback_freq=None, boole_deno_override=False, **kwargs):
        self.K = K
        self.n_x, self.n_u = n_x, n_u
        self.satisfaction_prob = satisfaction_prob
        self.skip_shrinking = skip_shrinking
        self.skip_feedback = skip_feedback
        self.state_plot_idxs = state_plot_idxs
        assert skip_feedback is True, "Currently not implemented feedback"
        if self.skip_feedback:
            self.K = np.zeros([self.n_u, self.n_x])
        else:
            assert K is not None, "Please provide a feedback gain matrix K"
            raise NotImplementedError

        self.boole_deno_override = boole_deno_override
        self.inverse_cdf_x = self.get_inv_cdf(self.n_x)

        self.shrinking_approach = shrinking_approach
        if self.shrinking_approach != "shrink_by_desired":
            raise NotImplementedError

        super().__init__(gp_fns, regions, region_subset_lim_lb, region_subset_lim_ub, delta_input_mask, gp_input_mask,
                         Bd, sys_dyn, Q, R, horizon_length, X, U, n_x, n_u, n_d, add_delta_constraints,
                         terminal_calc_fn, solver, addn_solver_opts, ignore_init_constr_check, add_scaling,
                         sampling_time, integration_method, ignore_callback, fwd_sim, infeas_debug_callback,
                         periodic_callback_freq, **kwargs)

    def get_inv_cdf(self, n_i):
        inverse_cdf_i = get_inv_cdf(n_i, self.satisfaction_prob, boole_deno_override=self.boole_deno_override)
        return inverse_cdf_i

    def init_opti_means_n_params(self):
        opti, mu_x, mu_u, x_init, x_desired, u_desired, P = super().init_opti_means_n_params()
        b_shrunk_x = cs.horzcat(cs.DM(self.X.b_np), self.opti.parameter(2 * self.n_x, self.N))
        b_shrunk_u = None
        if not self.skip_feedback:
            raise NotImplementedError("Need to add in b_shrunk_u vectors when there is feedback involved")
            # b_shrunk_u = cs.horzcat(cs.DM(self.U.b_np), self.opti.parameter(2*self.n_u, self.N-1))

        return opti, mu_x, mu_u, x_init, x_desired, u_desired, b_shrunk_x, b_shrunk_u, P

    def shrink_by_desired(self, simulation_length, cutoff_idx=19):
        print("Shrinking by desired. MAKE SURE YOU HAVE PASSED IN THE MOST UP TO DATE X_DESIRED. This approach is to be used AFTER"
            " hlds have been generated since the hlds are used to forward")
        affine_transform_list, Sigma_d_list = self.shrunk_gen_commons(simulation_length=simulation_length)
        print(len(Sigma_d_list))
        self.shrunk_vecs_x = np.zeros((simulation_length,
                                       2*self.n_x, self.N + 1))  # First timestep no shrinking. Use self.X.b_np
        if not self.skip_feedback:
            raise NotImplementedError(
                "Feedback not implemented yet but once it is, need to also shrink for the u vectors")
            # shrunk_vecs_u = np.zeros((simulation_length, self.n_u, self.N))     # First timestep no shrinking. Use self.U.b_np
        for wp_idx in range(simulation_length):
            Sigma_x_0_to_N = [np.zeros((self.n_x, self.n_x))]
            for k in range(self.N):
                curr_idx = wp_idx + k
                Sigma_d_curr = Sigma_d_list[curr_idx]
                Sigma_x_k = Sigma_x_0_to_N[k]
                if self.skip_feedback:
                    # Sigma_i = self.computesigma_wrapped(Sigma_x_k, Sigma_d_curr)
                    # Sigma_x_k_plus_1 = self.dt * (affine_transform_list[curr_idx] @ Sigma_i @ affine_transform_list[curr_idx].T)
                    Sigma_i = self.computesigma_wrapped(Sigma_x_k, Sigma_d_curr * (self.dt ** 2))
                    Sigma_x_k_plus_1 = affine_transform_list[curr_idx] @ Sigma_i @ affine_transform_list[curr_idx].T
                    Sigma_x_0_to_N.append(Sigma_x_k_plus_1)
                else:
                    raise NotImplementedError(
                        "Feedback not implemented yet. Need to write segment for shrunk u vecs")
                # cutoff_idx necessary to prevent problem from being infeasible due to shrunk sets being null sets towards the end of the horizon.
                if k == 0 or k > cutoff_idx:
                    # print(self.shrunk_vecs_x[wp_idx, :, [0]].shape, self.X.b_np.shape)
                    self.shrunk_vecs_x[wp_idx, :, [k]] = self.X.b_np.T
                else:
                    self.shrunk_vecs_x[wp_idx, :, [k]] = (self.X.b_np - (cs.fabs(self.X.H_np) @ (
                                cs.sqrt(cs.diag(Sigma_x_k)) * self.inverse_cdf_x))).T
            self.shrunk_vecs_x[wp_idx, :, -1] = self.shrunk_vecs_x[wp_idx, :, -2]
        # print(self.shrunk_vecs_x[0])

    def init_warmstart(self, initial_info_dict):
        super().init_warmstart(initial_info_dict)
        self.opti.set_value(self.opti_dict["b_shrunk_x"][:, 1:], self.shrunk_vecs_x[0, :, 1:])
        if not self.skip_feedback:
            raise NotImplementedError("Need to add in b_shrunk_u vectors when there is feedback involved")
            # self.opti.set_value(self.opti_dict["b_shrunk_u"][:, 1:], initial_info_dict["b_shrunk_u"])

    def define_pw_cb_fns(self, test_softplus=False):
        region_deltas = super().define_pw_cb_fns(test_softplus=test_softplus)
        self.get_Sigma_d = hybrid_res_covar('hybrid_res_cov', self.n_d, self.num_regions,
                                            self.N, opts={'enable_fd': True}, test_softplus=test_softplus)
        return region_deltas

    def respect_constraint_sets(self, mu_x, mu_u, k):
        # Note we cannot use numpy functions here because we're dealing with symbolic variables. Hence we need to make use of equivalent Casadi functions that are
        # capable of handling this.
        # Add constant before taking square root so that derivative includes the constant term to stop derivative blowing up to NaN
        if self.skip_shrinking:
            super().respect_constraint_sets(mu_x, mu_u, k)
        else:
            if k == 0:
                print("Not skipping shrinking.".upper())
            b_shrunk_x = self.opti_dict['b_shrunk_x'][:, k + 1]
            self.opti.subject_to(self.X.H_np @ mu_x[:, k + 1] - b_shrunk_x <= 0)
            if not self.skip_feedback:
                raise NotImplementedError("Need to add in b_shrunk_u vectors when there is feedback involved")
                # b_shrunk_u = self.opti_dict['b_shrunk_u'][:, k]
                # self.opti.subject_to(self.U.H_np @ mu_u[:, k] - b_shrunk_u <= 0)
            else:
                self.opti.subject_to(self.U.H_np @ mu_u[:, k] - self.U.b_np <= 0)

    def setup_cov_cb_fns(self):
        if self.skip_feedback:
            self.computesigma_wrapped = computeSigma_nofeedback('Sigma',
                                                                self.n_x, self.n_u, self.n_d,
                                                                opts={"enable_fd": True})
        else:
            raise NotImplementedError
            # self.compute_sigma_u = Sigma_u_Callback('Sigma_u', self.K,
            #                                         opts={"enable_fd": True})
            # self.computesigma_wrapped = computeSigmaglobal_meaneq('Sigma',
            #                                                       feedback_mat=self.K,
            #                                                       residual_dim=self.n_d,
            #                                                       opts={"enable_fd": True})

    # def create_affine_transform(self, x_lin, u_lin, cs_equivalent=False):
    #     partial_der_calc = self.sys_dyn.df_func(x=x_lin, u=u_lin)
    #     A_k, B_k = partial_der_calc['dfdx'], partial_der_calc['dfdu']
    #     if cs_equivalent:
    #         affine_transform = cs.horzcat(*(A_k, B_k, self.Bd))
    #     else:
    #         affine_transform = np.concatenate((A_k, B_k, self.Bd), axis=1)
    #     return affine_transform

    def linearize_and_discretize(self, x_lin, u_lin, cs_equivalent=False, method='zoh'):
        partial_der_calc = self.sys_dyn.df_func(x=x_lin, u=u_lin)
        A_c, B_c = partial_der_calc['dfdx'], partial_der_calc['dfdu']
        if not cs_equivalent:  # Note if cs_equivalent = True => doing shrinking within optimization and can't compute det with symbolic mat here.
            if np.linalg.det(A_c) == 0:
                A_k = np.eye(self.n_x) + self.dt * A_c
                B_k = self.dt * B_c
            else:
                A_k = scipy.linalg.expm(self.dt * np.array(A_c))
                B_k = np.linalg.inv(A_c) @ (A_k - np.eye(self.n_x)) @ B_c
        else:
            if method == "zoh":
                A_k = np.eye(self.n_x) + self.dt * A_c
                B_k = self.dt * B_c
            else:
                raise NotImplementedError
        return A_k, B_k

    def create_affine_transform(self, x_lin, u_lin, cs_equivalent=False, method='zoh'):
        A_k, B_k = self.linearize_and_discretize(x_lin, u_lin, cs_equivalent=cs_equivalent, method=method)

        if cs_equivalent:
            affine_transform = cs.horzcat(*(A_k, B_k, self.Bd))
        else:
            affine_transform = np.concatenate((A_k, B_k, self.Bd), axis=1)
        return affine_transform

    def shrunk_gen_commons(self, simulation_length):
        # Lists for affine transform for covariance dynamics propagation
        affine_transform_list = []
        # List for covariance matrices computed across x_desired array
        Sigma_d_list = []
        for wp_idx in range(self.N + simulation_length - 1):
            # Compute affine transform for current waypoint desired to be tracked.
            affine_transform = self.create_affine_transform(x_lin=self.x_desired[:, wp_idx], u_lin=self.u_desired[:, wp_idx])
            affine_transform_list.append(affine_transform)

            # Compute gp covariance for current waypoint for Sigma_d^k entry in Sigma_k matrix.
            joint_vec = np.vstack([self.x_desired[:, [wp_idx]], self.u_desired[:, [wp_idx]]])
            hybrid_means, *hybrid_covs = self.gp_fns(self.gp_input_mask @ joint_vec)
            Sigma_d = self.get_Sigma_d(self.hld_arr[:, [wp_idx]], *hybrid_covs)
            Sigma_d_list.append(Sigma_d)
        return affine_transform_list, Sigma_d_list

    def setup_OL_optimization(self):
        opti, mu_x, mu_u, x_init, x_desired, u_desired, b_shrunk_x, b_shrunk_u, self.P = self.init_opti_means_n_params()
        self.setup_cov_cb_fns()
        # Sets up ALL constraints except for the shrunk set constraints.
        super().res_mean_constraints(mu_x, mu_u, x_init, x_desired, u_desired)

        self.opti_dict.update({"b_shrunk_x": b_shrunk_x, "b_shrunk_u": b_shrunk_u})

        # Setup cost function to minimize during optimization along with shrunk set constraints.
        self.setup_cost_fn_and_shrinking()  # Use the overridden respect_constraint_sets function with shrinking added.
        self.setup_solver()

        if self.infeas_debug_callback:
            self.opti.callback(lambda i: self.opti_infeas_debug_cbfn(i))

    def cl_warmstart(self, data_dict, x_desired, sim_step_idx, verbose):
        super().cl_warmstart(data_dict, x_desired, sim_step_idx, verbose)
        self.opti.set_value(self.opti_dict['b_shrunk_x'][:, 1:], self.shrunk_vecs_x[sim_step_idx, :, 1:])

    def run_cl_opt(self, initial_info_dict, simulation_length, opt_verbose=False,
                   u_desired=None, x_desired=None, infeas_debug_callback=False):
        if x_desired is None and self.x_desired is None:
            raise ValueError(
                "x_desired must be passed to run_cl_opt if it is not already set in the controller instance.")
        if u_desired is None and self.u_desired is None:
            raise ValueError(
                "u_desired must be passed to run_cl_opt if it is not already set in the controller instance.")
        if x_desired is not None:
            self.set_traj_to_track(x_desired, u_desired=u_desired)

        # Note this needs to be the shrunk vector generation step since the deltas will be used to find the right GP model
        # to be used for covariance calculation.
        self.generate_hld_for_warmstart()
        self.setup_resmean_warmstart_vec()

        if self.shrinking_approach == "shrink_by_desired":
            self.shrink_by_desired(simulation_length)
        else:
            raise NotImplementedError

        data_dicts, sol_stats = super().run_cl_opt(initial_info_dict, simulation_length, opt_verbose)
        return data_dicts, sol_stats

    def plot_shrunk_sets(self, wp_idx, plot_shrunk_sep=False):
        assert self.state_plot_idxs is not None, "Need to set tracking matrix before plotting shrunk sets."
        shrunk_vecs_over_N = self.shrunk_vecs_x[wp_idx]
        if not plot_shrunk_sep:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        else:
            ax = None
        idx_range = np.linspace(0.3, 0.8, self.N + 1)
        alpha = idx_range
        # Define the colormap
        cmap = cm.get_cmap('viridis')
        colours = [cmap(i) for i in idx_range]
        for k in range(self.N + 1):
            plot_constraint_sets(plot_idxs=self.state_plot_idxs, inp_type="shrunk_vec", alpha=alpha[k], colour=colours[k],
                                 shrunk_vec=shrunk_vecs_over_N[:, k], ax=ax)

    def run_ol_opt(self, initial_info_dict, x_desired=None, u_desired=None, plot_shrunk_sets=False,
                   plot_shrunk_sep=False):
        if x_desired is None and self.x_desired is None:
            raise ValueError(
                "x_desired must be passed to run_cl_opt if it is not already set in the controller instance.")
        if u_desired is None and self.u_desired is None:
            raise ValueError(
                "u_desired must be passed to run_cl_opt if it is not already set in the controller instance.")
        if x_desired is not None:
            self.set_traj_to_track(x_desired, u_desired=u_desired)

        # Note this needs to be the shrunk vector generation step since the deltas will be used to find the right GP model
        # to be used for covariance calculation.
        self.generate_hld_for_warmstart()

        if self.shrinking_approach == "shrink_by_desired":
            self.shrink_by_desired(simulation_length=1)
        else:
            raise NotImplementedError

        sol = super().run_ol_opt(initial_info_dict, x_desired=x_desired)
        print(sol)

        if plot_shrunk_sets:
            self.plot_shrunk_sets(wp_idx=0, plot_shrunk_sep=plot_shrunk_sep)

        return sol


class GPMPC_QP(GPMPC_D):
    def __init__(self, satisfaction_prob, gp_fns, regions, region_subset_lim_lb, region_subset_lim_ub, delta_input_mask,
                 gp_input_mask, Bd, sys_dyn, Q, R, horizon_length, X: box_constraint, U: box_constraint, n_x, n_u, n_d,
                 add_delta_constraints=True, terminal_calc_fn="linearized_lqr", solver='ipopt', addn_solver_opts=None,
                 ignore_init_constr_check=False, add_scaling=False, sampling_time=20 * 10e-3, state_plot_idxs=None,
                 integration_method='euler', ignore_callback=False, fwd_sim="nom_dyn", infeas_debug_callback=False,
                 skip_shrinking=False, skip_feedback=True, shrinking_approach="shrink_by_desired", K=None,
                 periodic_callback_freq=None, boole_deno_override=False, **kwargs):

        super().__init__(satisfaction_prob, gp_fns, regions, region_subset_lim_lb, region_subset_lim_ub,
                         delta_input_mask, gp_input_mask, Bd, sys_dyn, Q, R, horizon_length, X, U, n_x, n_u, n_d,
                         add_delta_constraints, terminal_calc_fn, solver, addn_solver_opts, ignore_init_constr_check,
                         add_scaling, sampling_time, state_plot_idxs, integration_method, ignore_callback, fwd_sim,
                         infeas_debug_callback, skip_shrinking, skip_feedback, shrinking_approach, K,
                         periodic_callback_freq, boole_deno_override, **kwargs)

    def setup_OL_optimization(self):
        opti, mu_x, mu_u, x_init, x_desired, u_desired, b_shrunk_x, b_shrunk_u, self.P = self.init_opti_means_n_params()
        self.setup_cov_cb_fns()

        """
        THIS IS THE ONLY DIFFERENCE BETWEEN QP AND D. 
        Both nominal dynamics propagation and residual mean is modified when compared with D which inherits res_mean_constraints from BnC that
        defines mu_d as an opt var as opposed to a parameter as we do here.
        """
        # Sets up ALL constraints except for the shrunk set constraints.
        self.res_mean_constraints(mu_x, mu_u, x_init, x_desired, u_desired)

        self.opti_dict.update({"b_shrunk_x": b_shrunk_x, "b_shrunk_u": b_shrunk_u})

        # Setup cost function to minimize during optimization along with shrunk set constraints.
        self.setup_cost_fn_and_shrinking()  # Use the overridden respect_constraint_sets function with shrinking added.
        self.setup_solver()

        if self.infeas_debug_callback:
            self.opti.callback(lambda i: self.opti_infeas_debug_cbfn(i))

    def setup_dynamics_constraints(self, k, mu_x, mu_u, mu_d, x_desired, u_desired):
        if self.integration_method == "euler":
            super().setup_dynamics_constraints(k, mu_x, mu_u, mu_d)
        else:
            # Linearize about desired trajectory not opt since we need a static LTV system over the horizon, not one dependent
            # on the current assignments to the optimization variables.
            A_k, B_k = self.linearize_and_discretize(x_desired[:, k], u_desired[:, k], cs_equivalent=True, method='zoh')
            self.opti.subject_to(mu_x[:, k + 1] - (A_k @ mu_x[:, k] + B_k @ mu_u[:, k] + self.dt * (self.Bd @ mu_d[:, k])) == 0)

    def res_mean_constraints(self, mu_x, mu_u, x_init, x_desired, u_desired, test_softplus=False):
        mu_d = self.opti.parameter(self.n_d, self.N)

        # Defining get_mu_d and get_sigma_d for use with offline quantity generation step.
        _ = self.define_pw_cb_fns(test_softplus=test_softplus)

        # No need to set up symbolic mu_d computation dependent on region deltas. This is offloaded to
        # the setup_resmean_warmstart_vecs method which is called at the start of the run_cl_opt function.
        for k in range(self.N):
            # State mean dynamics
            self.setup_dynamics_constraints(k, mu_x, mu_u, mu_d, x_desired, u_desired)

        # We need to have a way to set x_init, x_desired etc. at every iteration of the closed loop optimization. This dict will maintain
        # references to variables/parameters contained in the "opti" instance and set values for parameters/provide warm start solutions for variables
        self.opti_dict = {"mu_x": mu_x, "mu_u": mu_u,
                          "x_init": x_init, "x_desired": x_desired, "u_desired": u_desired,
                          "mu_d": mu_d, "P": self.P}

    def mu_d_init(self, sim_step=None):
        # Change set_initial to set_value since mu_d is now a parameter
        if sim_step is None:
            self.opti.set_value(self.opti_dict["mu_d"], self.resmean_warmstart[:, :self.N])
        else:
            self.opti.set_value(self.opti_dict["mu_d"], self.resmean_warmstart[:, sim_step + 1: sim_step + 1 + self.N])

    def hld_init(self, sim_step=None):
        # No longer have hld parameters since already incorporated into the mu d parameter assignment step
        # using self.hld_arr
        pass


class GPMPC_That(GPMPC_D):
    def __init__(self, That_predictor: SoftLabelNet, satisfaction_prob, gp_fns, regions, region_subset_lim_lb, region_subset_lim_ub, delta_input_mask,
                 gp_input_mask, Bd, sys_dyn, Q, R, horizon_length, X: box_constraint, U: box_constraint, n_x, n_u, n_d,
                 mapping_type='scratch_full_ds', initial_mapping_ds: Mapping_DS=None,
                 add_delta_constraints=True, terminal_calc_fn="linearized_lqr", solver='ipopt', addn_solver_opts=None,
                 ignore_init_constr_check=False, add_scaling=False, sampling_time=20 * 10e-3, state_plot_idxs=None,
                 integration_method='euler', ignore_callback=False, fwd_sim="nom_dyn", infeas_debug_callback=False,
                 skip_shrinking=False, skip_feedback=True, shrinking_approach="shrink_by_desired", K=None, periodic_callback_freq=None, **kwargs):
        self.That_predictor = That_predictor
        self.mapping_type = mapping_type
        if self.mapping_type == 'scratch_full_ds':
            assert initial_mapping_ds is not None, "Need to pass initial_ds if mapping_type is scratch_full_ds."
            self.mapping_ds = initial_mapping_ds
            self.batch_size = self.mapping_ds.batch_size
            self.reset_ds_buffer()
            self.buffer_store_idx = 0
        else:
            raise NotImplementedError
        super().__init__(satisfaction_prob, gp_fns, regions, region_subset_lim_lb, region_subset_lim_ub,
                         delta_input_mask, gp_input_mask, Bd, sys_dyn, Q, R, horizon_length, X, U, n_x, n_u, n_d,
                         add_delta_constraints, terminal_calc_fn, solver, addn_solver_opts, ignore_init_constr_check,
                         add_scaling, sampling_time, state_plot_idxs, integration_method, ignore_callback, fwd_sim,
                         infeas_debug_callback, skip_shrinking, skip_feedback, shrinking_approach, K, periodic_callback_freq, **kwargs)

    def reset_ds_buffer(self):
        self.pw_gp_inp_buffer = np.zeros([self.mapping_ds.traj_ds_inst.pw_gp_inp.shape[0], self.batch_size])
        self.measured_res_buffer = np.zeros([self.mapping_ds.traj_ds_inst.measured_res.shape[0], self.batch_size])
        self.delta_control_vec = np.zeros([self.mapping_ds.traj_ds_inst.delta_control_vec.shape[0], self.batch_size])

    def dataset_buffer_update(self, pw_gp_inp, measured_res, delta_control_vec):
        self.pw_gp_inp_buffer[:, [self.buffer_store_idx]] = pw_gp_inp
        self.measured_res_buffer[:, [self.buffer_store_idx]] = measured_res
        self.delta_control_vec[:, [self.buffer_store_idx]] = delta_control_vec
        self.buffer_store_idx += 1
        if self.buffer_store_idx == self.batch_size:
            print("Updating mapping ds")
            self.mapping_ds.update_ds(self.pw_gp_inp_buffer, self.measured_res_buffer, self.delta_control_vec)
            self.buffer_store_idx = 0
            self.reset_ds_buffer()

    def perform_forward_sim(self, mu_x_0, mu_u_0, mu_x_1, no_noise=False):
        if self.fwd_sim == "nom_dyn":
            sampled_ns = super().perform_forward_sim(mu_x_0, mu_u_0, mu_x_1)
        elif self.fwd_sim == "w_pw_res":
            ops = fwdsim_w_pw_res(clip_to_region_fn=lambda x: self.clip_to_region_limits(x),
                                  true_func_obj=self.true_ds_inst,
                                  ct_dyn_nom=self.ct_dyn_nom, dt=self.dt,
                                  Bd=self.Bd_fwd_sim, gp_input_mask=self.gp_input_mask,
                                  delta_input_mask=self.delta_input_mask,
                                  x_0=mu_x_0, u_0=mu_u_0, ret_residual=True,
                                  no_noise=no_noise, ret_collision_bool=self.collision_check)  # Don't include noise for this increment since not shrinking based on variance outputs.
            if not self.collision_check:
                sampled_ns, sampled_residual = ops
            else:
                sampled_ns, sampled_residual, collision_bool = ops
            mu_x_0, mu_u_0 = np.array(mu_x_0, ndmin=2).reshape(-1, 1), np.array(mu_u_0, ndmin=2).reshape(-1, 1)
            pw_gp_inp = self.gp_input_mask @ np.vstack((mu_x_0, mu_u_0))
            delta_control_vec = self.delta_input_mask @ np.vstack((mu_x_0, mu_u_0))
            measured_res = sampled_residual
            self.dataset_buffer_update(pw_gp_inp, measured_res, delta_control_vec)
        else:
            raise NotImplementedError
        return sampled_ns

    def generate_hld_for_warmstart(self):
        num_steps = self.x_desired.shape[-1]
        x_desired_clipped = self.clip_x_des()
        inputs = torch.from_numpy((self.delta_input_mask @ np.vstack([x_desired_clipped, np.zeros([self.n_u, num_steps])])).astype(np.float32))
        outputs = self.That_predictor(inputs.T)
        pred_logits = outputs.data
        preds = torch.zeros_like(pred_logits)
        preds[torch.arange(pred_logits.shape[0]), torch.argmax(pred_logits, dim=1)] = 1
        self.hld_arr = preds.numpy().T
        assert self.hld_arr.shape[1] == num_steps, "Mismatch in shape of hld_arr"

        for sim_step in range(num_steps):
            assert np.sum(self.hld_arr[:, sim_step]) == 1, "Delta assignment %s is not unique for the given input. Please check the input and try again." % self.hld_arr[:, sim_step]


class GPMPC_MINLP(GPMPC_D):
    def __init__(self, satisfaction_prob, gp_fns, regions, region_subset_lim_lb, region_subset_lim_ub, delta_input_mask,
                 gp_input_mask, Bd, sys_dyn, Q, R, horizon_length, X: box_constraint, U: box_constraint, n_x, n_u, n_d,
                 ignore_variance_costs=True, shrink_by_param=False, add_delta_tol=False, enable_softplus=True, **kwargs):
        self.ignore_variance_costs = ignore_variance_costs
        self.shrink_by_param = shrink_by_param
        self.ignore_variance_compute = self.ignore_variance_costs and (kwargs.get('skip_shrinking', False) or self.shrink_by_param)
        self.add_delta_tol = add_delta_tol
        self.enable_softplus = enable_softplus
        assert not (self.add_delta_tol and self.enable_softplus), "Cannot enable both softplus and delta tolerance. They are different methods to prevent" \
                                                                  "negative covariances from numerical errors in setting the deltas while" \
                                                                  "solving the MINLP."
        super().__init__(satisfaction_prob, gp_fns, regions, region_subset_lim_lb, region_subset_lim_ub,
                         delta_input_mask, gp_input_mask, Bd, sys_dyn, Q, R, horizon_length, X, U, n_x, n_u, n_d,
                         **kwargs)

    def cl_warmstart(self, data_dict, x_desired, sim_step_idx, verbose):
        # Handles mu_d warmstarting as well
        BaseGPMPC_sanity.cl_warmstart(self, data_dict, x_desired, sim_step_idx, verbose)
        self.opti.set_initial(self.opti_dict["hld"],
                              self.hld_arr[:, sim_step_idx + 1:sim_step_idx + 1 + self.x_des_end_delim])
        self.opti.set_initial(self.opti_dict["mu_d"], self.resmean_warmstart[:, sim_step_idx + 1: sim_step_idx + 1 + self.N])

        if not self.skip_shrinking:
            if not self.shrink_by_param:
                self.opti.set_initial(self.opti_dict["b_shrunk_x"][:, 1:], self.shrunk_vecs_x[sim_step_idx, :, 1:])
            else:
                self.opti.set_value(self.opti_dict['b_shrunk_x'][:, 1:], self.shrunk_vecs_x[sim_step_idx, :, 1:])
            if not self.skip_feedback:
                raise NotImplementedError
            # self.opti.set_initial(self.opti_dict["b_shrunk_u"][:, 1:], self.shrunk_vecs_x[sim_step_idx, :, 1:])

    def init_warmstart(self, initial_info_dict):
        # Handles mu_d warmstarting as well
        BaseGPMPC_sanity.init_warmstart(self, initial_info_dict)
        self.opti.set_initial(self.opti_dict["hld"], self.hld_arr[:, :self.N + 1])
        self.opti.set_initial(self.opti_dict["mu_d"], self.resmean_warmstart[:, :self.N])
        if not self.skip_shrinking:
            if self.shrink_by_param:
                self.opti.set_value(self.opti_dict["b_shrunk_x"][:, 1:], self.shrunk_vecs_x[0, :, 1:])
            else:
                self.opti.set_initial(self.opti_dict["b_shrunk_x"][:, 1:], self.shrunk_vecs_x[0, :, 1:])
            if not self.skip_feedback:
                raise NotImplementedError("Need to add in b_shrunk_u vectors when there is feedback involved")
                # self.opti.set_value(self.opti_dict["b_shrunk_u"][:, 1:], initial_info_dict["b_shrunk_u"])
        # self.opti.set_initial(self.opti_dict["mu_d"], np.array([[0.2157, 0.0148, -0.0022]]))

    def define_delta_vars(self):
        region_deltas = self.opti.variable(len(self.regions), self.N + 1)
        return region_deltas

    def init_cov_arrays(self):
        # Because Sigma_x is not opt var.
        Sigma_x = [cs.MX.zeros((self.n_x, self.n_x)) for _ in range(self.N+1)]
        Sigma_x[0] = cs.MX.zeros((self.n_x, self.n_x))
        # Sigma_u needs to be an optimization variable because it is used to compute shrunk sets and casadi doesn't accept a raw MX variable in it's place.
        # Alternate would be to define Sigma_u as a callback function that takes in Sigma_x which is what we do here.
        Sigma_u = []  # Will be populated in for loop.
        if not self.skip_feedback:
            Sigma_u_0 = cs.MX.zeros((self.n_u, self.n_u))
            Sigma_u.append(Sigma_u_0)
        # Joint cov matrix
        Sigma = []
        return Sigma_x, Sigma_u, Sigma

    def construct_delta_constraint(self):
        constraint_obj = combine_box(self.X, self.U, verbose=False)
        masked_lb, masked_ub = self.delta_input_mask @ constraint_obj.lb, self.delta_input_mask @ constraint_obj.ub

        self.delta_constraint_obj = box_constraint(masked_lb, masked_ub)
        self.big_M = np.abs(self.delta_constraint_obj.b_np)*4

    def setup_delta_domain(self):
        # Domain specification for the delta variable. Combined with the fact that variable is discrete this makes the domain {0, 1}
        for k in range(self.N):
            for region_idx in range(self.num_regions):
                self.opti.subject_to(-self.hld_arr_opt[region_idx, k] <= 0)
                self.opti.subject_to(self.hld_arr_opt[region_idx, k] - 1 <= 0)

    def setup_delta_constraints(self, high_level_deltas, delta_controls):
        for k in range(self.N):
            # Specific to H and b matrix ordering generated by the box constraint class
            for region_idx, region in enumerate(self.regions):
                region_H, region_b = region.H_np, region.b_np  # For boxes, the number of rows will be 2*delta_inp_dim. Delta_inp_row = delta_controls[:, [k]].shape[0]
                self.opti.subject_to(region_H @ delta_controls[:, [k]] <= (region_b + ((-region_b + self.big_M)*(1-high_level_deltas[region_idx, k]))))
            # Enforce that all deltas at any time step must sum to 1.
            self.opti.subject_to(cs.DM.ones(1, self.num_regions) @ high_level_deltas[:, [k]] - 1 == 0)

    def respect_constraint_sets(self, mu_x, mu_u, k):
        # Note we cannot use numpy functions here because we're dealing with symbolic variables. Hence we need to make use of equivalent Casadi functions that are
        # capable of handling this.
        # Add constant before taking square root so that derivative includes the constant term to stop derivative blowing up to NaN
        if self.skip_shrinking or self.shrink_by_param:
            super().respect_constraint_sets(mu_x, mu_u, k)
        else:
            b_shrunk_x_opt, Sigma_x = self.get_attrs_from_dict(["b_shrunk_x", "Sigma_x"])
            delta_tol = 1e-8
            sqrt_arr = cs.sqrt(cs.diag(Sigma_x[k+1]) + delta_tol)
            # sqrt_arr = cs.diag(Sigma_x[k+1])
            # tol_arr = cs.DM.ones(self.n_x, 1)*delta_tol
            # tol_sqrt_arr = cs.vertcat(*[cs.mmax(cs.horzcat(tol_arr[i, 0], sqrt_arr[i, 0])) for i in range(self.n_x)])
            # tol_sqrt_arr = cs.mmax([sqrt_arr, cs.MX.ones(2*self.n_x, 1)*delta_tol])
            # tol_sqrt_arr = sqrt_arr + delta_tol
            tol_sqrt_arr = sqrt_arr
            self.opti.subject_to(b_shrunk_x_opt[:, k+1] - (self.X.b_np - (cs.fabs(self.X.H_np) @ (tol_sqrt_arr * self.inverse_cdf_x))) == 0)

            self.opti.subject_to(self.X.H_np @ mu_x[:, k + 1] - b_shrunk_x_opt[:, k+1] <= 0)
            if not self.skip_feedback:
                raise NotImplementedError("Need to add in b_shrunk_u vectors when there is feedback involved")
                # b_shrunk_u = self.opti_dict['b_shrunk_u'][:, k]
                # self.opti.subject_to(self.U.H_np @ mu_u[:, k] - b_shrunk_u <= 0)
            else:
                self.opti.subject_to(self.U.H_np @ mu_u[:, k] - self.U.b_np <= 0)

    def setup_cost_metric(self):
        mu_x, mu_u, x_desired, u_desired, x_init = self.get_attrs_from_dict(
            ['mu_x', 'mu_u', 'x_desired', 'u_desired', 'x_init'])
        if not self.ignore_variance_costs:
            Sigma_x = self.get_attrs_from_dict(['Sigma_x'])
        else:
            Sigma_x = None
        # Cost function stuff
        cost = 0
        # Stage cost
        if not self.ignore_cost:
            for k in range(self.N):
                cost += self.cost_fn(mu_i_x=mu_x, mu_i_u=mu_u,
                                     x_desired=x_desired, u_desired=u_desired, Sigma_x=Sigma_x, idx=k)
            # Terminal cost
            cost += self.cost_fn(mu_x[:, -1], None, x_desired=x_desired, u_desired=None, Sigma_x=Sigma_x,
                                 terminal=True)
        self.opti.minimize(cost)

    def cost_fn(self, mu_i_x, mu_i_u, x_desired, u_desired, idx=-1, terminal=False, Sigma_x=None, Sigma_u=None):
        if not terminal:
            # mu_i_u for now just assumes we want to drive the system to stable/equilibrium state.
            x_des_dev, u_des_dev = (mu_i_x[:, idx] - x_desired[:, idx]), (mu_i_u[:, idx] - u_desired[:, idx])
            # Mahalanobis/weighted 2 norm for x, u.
            mu_i_x_cost = x_des_dev.T @ self.Q @ x_des_dev
            mu_i_u_cost = u_des_dev.T @ self.R @ u_des_dev
            var_x_cost, var_u_cost = 0, 0
            if not self.ignore_variance_costs:
                var_x_cost = cs.trace(self.Q @ Sigma_x[idx])
                if not self.skip_feedback:
                    raise NotImplementedError
                    # var_u_cost = cs.trace(self.R @ Sigma_u[idx])
            return mu_i_x_cost + mu_i_u_cost + var_x_cost + var_u_cost
        else:
            x_des_dev = (mu_i_x[:, -1] - x_desired[:, -1])
            # Note self.P = self.opti_dict["P"] i.e. the parametrized terminal cost matrix that is set in closed loop.
            mu_i_x_cost = x_des_dev.T @ self.P @ x_des_dev
            var_x_cost = 0
            if not self.ignore_variance_costs:
                var_x_cost = cs.trace(self.P @ Sigma_x[-1])
            return mu_i_x_cost + var_x_cost

    def define_pw_cb_fns(self, test_softplus=False):
        region_deltas = super().define_pw_cb_fns(test_softplus=self.enable_softplus)
        self.get_Sigma_d = hybrid_res_covar('hybrid_res_cov', self.n_d, self.num_regions,
                                            self.N, opts={'enable_fd': True},
                                            delta_tol=(1e-2 if self.add_delta_tol else 0), test_softplus=self.enable_softplus)
        return region_deltas

    def init_opti_means_n_params(self):
        opti, mu_x, mu_u, x_init, x_desired, u_desired, P = GPMPC_BnC.init_opti_means_n_params(self)
        if self.skip_shrinking:
            b_shrunk_x = cs.horzcat(*([cs.DM(self.X.b_np)]*(self.N+1)))
        elif self.shrink_by_param:
            b_shrunk_x = cs.horzcat(cs.DM(self.X.b_np), self.opti.parameter(2 * self.n_x, self.N))
        else:
            b_shrunk_x = cs.horzcat(cs.DM(self.X.b_np), self.opti.variable(2 * self.n_x, self.N))

        b_shrunk_u = None
        if not self.skip_feedback:
            raise NotImplementedError("Need to add in b_shrunk_u vectors when there is feedback involved")
            # b_shrunk_u = cs.horzcat(cs.DM(self.U.b_np), self.opti.parameter(2*self.n_u, self.N-1))

        return opti, mu_x, mu_u, x_init, x_desired, u_desired, b_shrunk_x, b_shrunk_u, P

    def setup_hybrid_covs_manual_preds(self, gp_inp_vec_size, gp_input):
        models = self.gp_fns.models
        # Instead of computing hybrid means 1 timestep at a time, we compute hybrid means over the entire trajectory for a given
        # model at one shot. Allows us to make use of efficiency with batch computing to speed up run-time. mu_d_pw_r is thus
        # representative of the residual terms generated by region r over the entire O.L. horizon.
        assert self.n_d == 1, "Make sure the test example in models.utils runs when n_d is > 1 before removing this assert statement"
        Sigma_d_pw_r = [[cs.MX.sym('Sigma_d_pw', self.n_d, self.n_d) for r in range(self.num_regions)] for k in range(self.N)]  # placeholder
        assert self.n_d == 1, "The shape used for the Sigma_d array will work only assuming n_d == 1 at the moment."
        # Sigma_d_pw_r = [([cs.MX.sym('mu_d_pw', self.n_d, self.n_d) for k in range(self.N)])
        #                 for r in range(self.num_regions)]  # placeholder
        gp_inp_OL = cs.MX.sym('gp_inp_k', gp_inp_vec_size, self.N)
        # Store functions in list to prevent garbage collection. Casadi needs to have references to these functions for gradient
        # computation and if they get garbage collected these refs will be invalid.
        res_compute_fns = []
        for r in range(self.num_regions):
            models[r]: GP_Model
            get_hyb_res_cov = cs.Function('compute_resmean_region_r', [gp_inp_OL],
                                          [models[r].cs_predict_cov(gp_inp_OL)],
                                          ['gp_inp_OL'], ['mu_d_hyb'])
            res_compute_fns.append(get_hyb_res_cov)
            # if r == 0:
            #     assert get_hyb_res_cov(gp_inp_OL).shape == Sigma_d_pw_r[0].shape, print(get_hyb_res_cov(gp_inp_OL).shape,
            #                                                                             Sigma_d_pw_r[0].shape)
            Sigma_d_pw_r[r] = get_hyb_res_cov(gp_input)
            # print(Sigma_d_pw_r[r].shape)
        # Reshape hybrid means from all regions over OL horizon into the form expected by the get_mu_d cs Function.
        # Each element in the below array has shape (n_d, num_regions)
        # Sigma_d_pw_arr = [cs.horzcat(*[Sigma_d_pw_r[r][:, k] for r in range(self.num_regions)]) for k in range(self.N)]
        assert self.n_d == 1, "Check that the split up works for n_d > 1."
        Sigma_d_pw_arr = [[Sigma_d_pw_r[r][k] for r in range(self.num_regions)] for k in range(self.N)]
        assert len(Sigma_d_pw_arr) == self.N
        assert len(Sigma_d_pw_arr[0]) == self.num_regions

        return res_compute_fns, Sigma_d_pw_arr

    def compute_Sigma_d_k(self, k, region_deltas_k, Sigma_d_pw_arr=None):
        hybrid_covs = [Sigma_d_pw_arr[k][r] for r in range(self.num_regions)]
        cov_mat = self.get_Sigma_d(region_deltas_k, *hybrid_covs)
        # NOTE: Not having the below constraint unlike with the mean vector because the covariance mat is just an intermediate
        # for the set shrinking. The set shrinking vectors will be defined as opt vars. The cov mats will just be computed by
        # chaining on the state and input which ARE opt vars.
        # self.opti.subject_to(cov_mat - Sigma_d_arr[k] == 0)  # Sigma_d_arr[k] is of shape self.n_d x self.n_d.
        return cov_mat

    def shrunk_gen_opt_commons(self, Sigma_d_pw_arr):
        """Adapted from shrunk_gen_commons which does this for the relaxed version of the problem using information
        from x_desired. """
        # Lists for affine transform for covariance dynamics propagation
        affine_transform_sym_list = []
        # List for covariance matrices computed across x_desired array
        Sigma_d_sym_list = []
        mu_x_opt, mu_u_opt = self.opti_dict["mu_x"], self.opti_dict["mu_u"]
        hld_opt_arr = self.opti_dict["hld"]
        for k in range(self.N):
            affine_transform = self.create_affine_transform(x_lin=mu_x_opt[:, k], u_lin=mu_u_opt[:, k], cs_equivalent=True)
            affine_transform_sym_list.append(affine_transform)

            # Compute gp covariance for current waypoint for Sigma_d^k entry in Sigma_k matrix.
            Sigma_d_k_sym = self.compute_Sigma_d_k(k, hld_opt_arr[:, k], Sigma_d_pw_arr)
            Sigma_d_sym_list.append(Sigma_d_k_sym)
        return affine_transform_sym_list, Sigma_d_sym_list

    def setup_covariance_dynamics(self, Sigma_d_pw_arr):
        # This will always be needed if ignore_variance_compute is False. If just for shrinking it is required and even if just
        # for variance costs it's still required since the costs are computed based on the mu_x (and mu_u) opt vars for which the
        # symbolic equations are set up in shrunk_gen_opt_commons and the loop below that method call in this function.
        affine_transform_sym_list, Sigma_d_sym_list = self.shrunk_gen_opt_commons(Sigma_d_pw_arr)

        Sigma_x, Sigma_u, Sigma = self.init_cov_arrays()

        for k in range(self.N):
            Sigma_d_curr = Sigma_d_sym_list[k]
            Sigma_x_k = Sigma_x[k]
            if self.skip_feedback:
                Sigma_i = self.computesigma_wrapped(Sigma_x_k, Sigma_d_curr * (self.dt ** 2))
            else:
                raise NotImplementedError(
                    "Feedback not implemented yet. Need to write segment for shrunk u vecs")
            Sigma_x_k_plus_1 = affine_transform_sym_list[k] @ Sigma_i @ affine_transform_sym_list[k].T
            Sigma_x[k+1] = Sigma_x_k_plus_1

        self.opti_dict.update({"Sigma_x": Sigma_x, "Sigma": Sigma, "Sigma_d": Sigma_d_sym_list})
        if not self.skip_feedback:
            self.opti_dict.update({"Sigma_u": Sigma_u})

    def res_cov_opt_setup(self, mu_x, mu_u):
        # Extract variables from joint state-input vector that must be passed to the GP to compute the residual means
        mu_z, gp_inp_vec_size, gp_input = self.create_gp_inp_sym(mu_x, mu_u)

        res_compute_fns, Sigma_d_pw_arr = self.setup_hybrid_covs_manual_preds(gp_inp_vec_size, gp_input)
        self.opti_dict.update({"Sigma_d_pw_arr": Sigma_d_pw_arr})

        if not self.ignore_variance_compute:
            self.setup_covariance_dynamics(Sigma_d_pw_arr)

    def setup_solver(self, print_order=False):
        self.nx_after_deltas = self.opti.nx

        # for i in range(self.nx_before_deltas):
        #     print(self.opti.debug.x_describe(i))
        # print('after')
        # for i in range(self.nx_before_deltas, self.nx_after_deltas):
        #     print(self.opti.debug.x_describe(i))

        num_x = self.n_x * (self.N + 1)
        num_u = self.n_u * self.N
        num_mu_d = self.n_d * self.N
        num_shrunk = 0
        if not (self.shrink_by_param or self.skip_shrinking):  # If neither satisfied then shrunk vectors are opt vars.
            num_shrunk = 2 * self.n_x * self.N  # Not N+1 since 0th timestep has self.X.b_np constant
        nx_before_deltas = num_x + num_u + num_shrunk + num_mu_d  # Order in sum reflects ordering in self.opti.debug.x_describe

        num_hld = self.num_regions * (self.N + 1)

        if print_order:
            print('\n'.join([self.opti.debug.x_describe(idx) for idx in range(self.opti.nx)]))

        assert self.nx_after_deltas == nx_before_deltas + num_hld, "nx isn't created in the order expected. %s" % ('\n'.join([self.opti.debug.x_describe(idx) for idx in range(self.opti.nx)]))
        discrete_bool_vec = ([False] * nx_before_deltas) + ([True] * num_hld)
        self.discrete_bool_vec = discrete_bool_vec
        # opts = {"enable_fd": True, "bonmin.max_iter": 20, "bonmin.print_level": 4}
        new_opts = {}
        self.solver_opts['discrete'] = discrete_bool_vec
        for opt in self.solver_opts.keys():
            if 'ipopt' not in opt:
                new_opts[opt] = self.solver_opts[opt]
        self.solver_opts = new_opts

        opts = {"bonmin.print_level": 4, 'bonmin.expect_infeasible_problem': 'no'}
        # opts.update({"bonmin.allowable_gap": 2})

        # opts["monitor"] = ["nlp_g", "nlp_jac_g", "nlp_f", "nlp_grad_f"]
        # opts["monitor"] = ["nlp_f"]
        # opts["bonmin.solution_limit"] = 4
        # opts.update({"bonmin.allowable_gap": 5})
        opts["bonmin.integer_tolerance"] = 1e-2
        # if hsl_solver:
        #     opts['bonmin.linear_solver'] = 'ma27'
        # Attempt to fix grad_gamma_p error. Ref: https://groups.google.com/g/casadi-users/c/vDewPLPXLYA/m/96BP6Nt2BQAJ
        opts["calc_lam_p"] = False
        opts["bonmin.hessian_approximation"] = "limited-memory"

        opts["discrete"] = discrete_bool_vec
        self.solver_opts = opts

        self.opti.solver(self.solver, self.solver_opts)

    def setup_OL_optimization(self):
        opti, mu_x, mu_u, x_init, x_desired, u_desired, b_shrunk_x, b_shrunk_u, self.P = self.init_opti_means_n_params()

        # Sets up ALL constraints except for the shrunk set constraints. Also sets up the function
        # for getting residual covariances and creating the delta variables and defining the nx_before_deltas variable.
        super().res_mean_constraints(mu_x, mu_u, x_init, x_desired, u_desired, test_softplus=self.enable_softplus)

        self.opti_dict.update({"b_shrunk_x": b_shrunk_x, "b_shrunk_u": b_shrunk_u})

        self.hld_arr_opt = self.opti_dict["hld"]
        self.setup_delta_domain()
        self.setup_cov_cb_fns()

        self.res_cov_opt_setup(mu_x, mu_u)

        # Sets up constraints when the lld corresponding to the inequality row k of region r is 1 only if the delta_control input
        # satisfies it and 0 if it does not.
        delta_controls = self.opti_dict["delta_controls"]  # Setup by res_mean_constraints.
        self.construct_delta_constraint()  # Set up big M
        self.setup_delta_constraints(self.hld_arr_opt, delta_controls)

        # Now that Sigma_x and Sigma_u symbolic chaining has been setup, we can setup the cost function and shrinking.
        self.setup_cost_fn_and_shrinking()

        # for g_idx in range(self.opti.ng):
        #     print(self.opti.debug.g_describe(g_idx))
        # print(self.opti.debug.g_describe(142))
        self.setup_solver()

    def periodic_cl_callback(self, iter_num, runtime_error=False):
        if runtime_error:
            self.opt_verbose = True
            self.solver_opts.update({"monitor": "nlp_g"})
            self.opti.solver(self.solver, self.solver_opts)
            try:
                print("ERRORED OUT: Re-solving with monitor for debugging")
                sol = self.solve_optimization(ignore_initial=True, verbose=self.opt_verbose)
            except Exception as e:
                opti_inst = self.opti
                # test_gradient = cs.Function('grad_comp', [opti_inst.x, opti_inst.p],
                #                             [cs.jacobian(opti_inst.g, opti_inst.x)])
                # print(test_gradient(x, p)[6, :])
            self.opt_verbose = False
            del self.solver_opts["monitor"]
            self.opti.solver(self.solver, self.solver_opts)

        mu_x_opt = self.opti.debug.value(self.opti_dict["mu_x"])
        mu_u_opt = self.opti.debug.value(self.opti_dict["mu_u"])
        if not self.skip_shrinking:
            if self.shrink_by_param:
                b_shrunk_x = self.shrunk_vecs_x[iter_num]
            else:
                b_shrunk_x = self.opti.debug.value(self.opti_dict["b_shrunk_x"])
        else:
            print("Skipped shrinking")
        mu_d_opt = self.opti.debug.value(self.opti_dict["mu_d"])
        hld_opt = self.opti.debug.value(self.opti_dict["hld"])
        if not self.skip_shrinking:
            x = cs.vertcat(cs.vec(mu_x_opt), cs.vec(mu_u_opt), cs.vec(b_shrunk_x[:, 1:]), cs.vec(mu_d_opt), cs.vec(hld_opt))
            print(x)

        test_nans = False
        if test_nans is True and iter_num == 7:
            self.opt_verbose = True
            self.solver_opts.update({"monitor": "nlp_f"})
            self.opti.solver(self.solver, self.solver_opts)
