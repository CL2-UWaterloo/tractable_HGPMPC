import numpy as np
from common.box_constraint_utils import box_constraint_direct
from ds_utils import GP_DS
import torch
import casadi as cs
import sys
import traceback
import executing.executing
import copy
import matplotlib.pyplot as plt

from common.plotting_utils import save_fig


class OptiDebugger:
    def __init__(self, controller_inst):
        self.controller_inst = controller_inst

    def get_vals_from_opti_debug(self, var_name):
        assert var_name in self.controller_inst.opti_dict.keys(), "Controller's opti_dict has no key: %s . Add it to the dictionary within the O.L. setups" % var_name
        if type(self.controller_inst.opti_dict[var_name]) in [list, tuple]:
            return [self.controller_inst.opti.debug.value(var_k) for var_k in self.controller_inst.opti_dict[var_name]]
        else:
            return self.controller_inst.opti.debug.value(self.controller_inst.opti_dict[var_name])


def plot_OL_opt_soln(debugger_inst: OptiDebugger, state_plot_idxs, ax=None, waypoints_to_track=None,
                     ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center'):
    mu_x_ol = np.array(debugger_inst.get_vals_from_opti_debug('mu_x'), ndmin=2)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(mu_x_ol[state_plot_idxs[0], :].squeeze(), mu_x_ol[state_plot_idxs[1], :].squeeze(), color="r", marker='x',
            linestyle='dashed', linewidth=2, markersize=12, label='OL output')
    if waypoints_to_track is not None:
        ax.plot(waypoints_to_track[0, :], waypoints_to_track[1, :], color='g', marker='o', linestyle='solid',
                label='Trajectory to track')
    ax.set_xlim(ax_xlim)
    ax.set_ylim(ax_ylim)
    ax.legend(loc=legend_loc)


def plot_CL_opt_soln(waypoints_to_track, data_dict_cl, ret_mu_x_cl, state_plot_idxs, plot_ol_traj=False, axes=None,
                     ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center', regions=None, plot_idxs=(1, 3),
                     ignore_legend=False, itn_num=1, colour='r', label='CL Trajectory'):
    mu_x_cl = [data_dict_ol['mu_x'] for data_dict_ol in data_dict_cl]

    if plot_ol_traj:
        if axes is None:
            fig, axes = plt.subplots(2, 1)
        else:
            assert len(axes) == 2, "axes must be a list of length 2 if plot_ol_traj=True so OL traj can be plotted on separate graph"
    else:
        if axes is None:
            fig, axes = plt.subplots(1, 1)
            axes = [axes]
        else:
            assert len(axes) == 1, "axes must be a list of length 1 if plot_ol_traj=False"


    for ax in axes:
        if waypoints_to_track is not None:
            ax.plot(waypoints_to_track[0, :], waypoints_to_track[1, :], color='cyan',
                    label='Trajectory to track', marker='o', linewidth=3, markersize=10)
        if not ignore_legend:
            ax.legend(loc='upper center')
        colours = ['r', 'g', 'b']
        if regions is not None:
            for region_idx, region in enumerate(regions):
                # if type(region.plot_idxs) is not list:
                #     region.plot_idxs = list(plot_idxs)
                region.plot_constraint_set(ax, colour=colours[region_idx], alpha=0.6)
        ax.set_xlim(ax_xlim)
        ax.set_ylim(ax_ylim)

    mu_x_z_cl = [(mu_x_ol[state_plot_idxs[0], 0], mu_x_ol[state_plot_idxs[1], 0]) for mu_x_ol in mu_x_cl]
    axes[0].plot(np.array([mu_x_k[0] for mu_x_k in mu_x_z_cl]).squeeze(),
                 np.array([mu_x_k[1] for mu_x_k in mu_x_z_cl]),
                 color=colour, marker='o', linewidth=3, markersize=10, label=label)

    if plot_ol_traj:
        colours = ['cyan', 'r', 'g', 'b'] * (len(mu_x_cl) // 4 + 1)
        for timestep, mu_x_ol in enumerate(mu_x_cl):
            axes[1].plot(mu_x_ol[state_plot_idxs[0], :].squeeze(), mu_x_ol[state_plot_idxs[1], :].squeeze(), color=colours[timestep], marker='x',
                         linestyle='dashed', linewidth=3, markersize=10)

    for ax in axes:
        if not ignore_legend:
            ax.legend(loc=legend_loc, fontsize=15)

    # save_fig(axes=axes, fig_name="cl_traj_mapping_"+str(itn_num), tick_sizes=16)

    if ret_mu_x_cl:
        return mu_x_cl


def calc_cl_cost(mu_x_cl, mu_u_cl, x_desired, u_desired, Q, R):
    x_cost, u_cost = 0, 0
    for sim_step in range(len(mu_x_cl)-1):
        x_des_dev, u_des_dev = (mu_x_cl[sim_step] - x_desired[:, sim_step]), (mu_u_cl[sim_step] - u_desired[:, sim_step])
        x_cost += x_des_dev.T @ Q @ x_des_dev
        u_cost += u_des_dev.T @ R @ u_des_dev
    x_final_dev = (mu_x_cl[-1] - x_desired[:, len(mu_x_cl)-1])
    x_cost += x_final_dev.T @ Q @ x_final_dev
    return x_cost + u_cost


def retrieve_controller_results(controller_inst, X_test, U_test, ignore_covs=False, return_data_dict=False, verbose=True):
    debugger_inst = OptiDebugger(controller_inst)
    data_dict = {}
    try:
        data_dict["mu_x"] = debugger_inst.get_vals_from_opti_debug("mu_x")
    except RuntimeError:
        data_dict["mu_x"] = None
        data_dict["run_failed"] = True
        return False, data_dict
    data_dict["mu_u"] = debugger_inst.get_vals_from_opti_debug("mu_u")
    try:
        data_dict["mu_d"] = debugger_inst.get_vals_from_opti_debug("mu_d")
    except RuntimeError:
        print("COULDN'T GET mu_d")
    if not ignore_covs:
        data_dict["Sigma_x"] = debugger_inst.get_vals_from_opti_debug("Sigma_x")
    data_dict["b_shrunk_x"] = debugger_inst.get_vals_from_opti_debug("b_shrunk_x")
    data_dict["b_shrunk_u"] = debugger_inst.get_vals_from_opti_debug("b_shrunk_u")
    if verbose:
        display_controller_results(data_dict, ignore_covs, X_test, U_test)
    if return_data_dict:
        return debugger_inst, data_dict
    else:
        return debugger_inst


def display_controller_results(data_dict, ignore_covs, X_test, U_test):
    print("mu_x")
    print(data_dict["mu_x"])
    print("mu_u")
    print(data_dict["mu_u"])
    print("mu_d")
    print(data_dict["mu_d"])
    if not ignore_covs:
        print("Sigma_x")
        print(data_dict["Sigma_x"])
    print("X constraint b vector")
    print(X_test.b)
    print("Shrunk X b vectors over O.L. horizon")
    print(data_dict["b_shrunk_x"])
    print("U constraint b vector")
    print(U_test.b)
    print("Shrunk U b vectors over O.L. horizons")
    print(data_dict["b_shrunk_u"])


def retrieve_controller_results_piecewise(controller_inst, X_test, U_test,
                                          ignore_lld=False, ignore_covs=False, return_data_dict=False, verbose=True):
    global_returns = retrieve_controller_results(controller_inst, X_test, U_test,
                                                 ignore_covs=ignore_covs, return_data_dict=return_data_dict, verbose=verbose)
    if return_data_dict:
        debugger_inst, data_dict = global_returns
    else:
        debugger_inst, data_dict = global_returns, {}
    if not ignore_lld:
        data_dict["lld"] = debugger_inst.get_vals_from_opti_debug("lld")
    try:
        data_dict["hld"] = debugger_inst.get_vals_from_opti_debug("hld")
    # Attribute Error results from data_dict being bool due to the global controller results yielded False
    except AttributeError:
        print("Errored out (most probably with MINLP Error)")
        return debugger_inst, data_dict
    if not ignore_covs:
        data_dict["Sigma_d"] = debugger_inst.get_vals_from_opti_debug("Sigma_d")
    if verbose:
        display_controller_results_piecewise(data_dict, ignore_lld, ignore_covs)
    if return_data_dict:
        return debugger_inst, data_dict
    else:
        return debugger_inst


def display_controller_results_piecewise(data_dict, ignore_lld, ignore_covs):
    if not ignore_lld:
        print("lld")
        print(data_dict["lld"])
    print("hld")
    print(data_dict["hld"])
    if not ignore_covs:
        print("Sigma d")
        print(data_dict["Sigma_d"])


def construct_config_opts(print_level, add_monitor, early_termination, hsl_solver,
                          test_no_lam_p, hessian_approximation=True, jac_approx=False):
    # Ref https://casadi.sourceforge.net/v1.9.0/api/html/dd/df1/classCasADi_1_1IpoptSolver.html
    opts = {"ipopt.print_level": print_level}
    if add_monitor:
        # opts["monitor"] = ["nlp_g", "nlp_jac_g", "nlp_f", "nlp_grad_f"]
        opts["monitor"] = ["nlp_g"]
    if test_no_lam_p:
        opts["calc_lam_p"] = False
    if hessian_approximation:
        opts["ipopt.hessian_approximation"] = "limited-memory"
    if jac_approx:
        opts["ipopt.jacobian_approximation"] = "finite-difference-values"
    if hsl_solver:
        opts["ipopt.linear_solver"] = "ma27"
    acceptable_dual_inf_tol = 1e4
    acceptable_compl_inf_tol = 1e-1
    acceptable_iter = 5
    acceptable_constr_viol_tol = 1e-1
    acceptable_tol = 1e5
    max_iter = 15

    if early_termination:
        additional_opts = {"ipopt.acceptable_tol": acceptable_tol, "ipopt.acceptable_constr_viol_tol": acceptable_constr_viol_tol,
                           "ipopt.acceptable_dual_inf_tol": acceptable_dual_inf_tol, "ipopt.acceptable_iter": acceptable_iter,
                           "ipopt.acceptable_compl_inf_tol": acceptable_compl_inf_tol, "ipopt.max_iter": max_iter}
        opts.update(additional_opts)
    return opts


def construct_config_opts_minlp(print_level, add_monitor=False, early_termination=True, hsl_solver=False,
                                test_no_lam_p=True, hessian_approximation=True):
    opts = {"bonmin.print_level": print_level, 'bonmin.file_solution': 'yes', 'bonmin.expect_infeasible_problem': 'no'}
    # opts.update({"bonmin.allowable_gap": -100, 'bonmin.allowable_fraction_gap': -0.1, 'bonmin.cutoff_decr': -10})
    opts.update({"bonmin.allowable_gap": 2})
    opts["bonmin.integer_tolerance"] = 1e-2

    if add_monitor:
        opts["monitor"] = ["nlp_g", "nlp_jac_g", "nlp_f", "nlp_grad_f"]
    if early_termination:
        opts["bonmin.solution_limit"] = 4
        opts.update({"bonmin.allowable_gap": 5})
    if hsl_solver:
        opts['bonmin.linear_solver'] = 'ma27'
    if test_no_lam_p:
        opts["calc_lam_p"] = False
    if hessian_approximation:
        opts["bonmin.hessian_approximation"] = "limited-memory"


def adjust_path_length(path_info_dict, N, simulation_length, closed_loop):
    desired_length = N+1
    if closed_loop:
        desired_length += simulation_length+N+1
    gend_path = path_info_dict["path2track_w_MPC"]
    final_x_desired = gend_path[:, [-1]]
    if gend_path.shape[-1] < desired_length:
        len_to_add = desired_length - gend_path.shape[-1]
        temp = np.zeros([gend_path.shape[0], desired_length])
        temp[:, :gend_path.shape[-1]] = gend_path
        temp[:, gend_path.shape[-1]:] = np.ones([gend_path.shape[0], len_to_add]) * final_x_desired
        path_info_dict["path2track_w_MPC"] = temp


def fwdsim_w_pw_res(true_func_obj: GP_DS, ct_dyn_nom, dt, Bd, gp_input_mask, delta_input_mask, x_0, u_0, ret_residual=True,
                    no_noise=False, clip_to_region_fn=None, clip_state=False,
                    ret_collision_bool=False, integration_method='euler', dt_dyn_nom=None, **kwargs):
    x_0, u_0 = np.array(x_0, ndmin=2).reshape(-1, 1), np.array(u_0, ndmin=2).reshape(-1, 1)
    # x_0_delta = clip_to_region_fn(x_0) if clip_to_region_fn is not None else x_0
    x_0_delta = clip_to_region_fn(x_0) if clip_to_region_fn is not None else x_0  # Clipping to account for slight optimization errors ex: even -2e-8 prevents region idx assignment
    gp_input_0 = torch.from_numpy((gp_input_mask @ np.vstack([x_0_delta, u_0])).astype(np.float32))
    delta_input_0 = torch.from_numpy((delta_input_mask @ np.vstack([x_0_delta, u_0])).astype(np.float32))
    # print(delta_input_0)
    # print(delta_input_0)
    # print([str(region) for region in true_func_obj.regions])
    delta_dict = true_func_obj._generate_regionspec_mask(input_arr=delta_input_0, delta_var_inp=True)
    sampled_residual = true_func_obj.generate_outputs(input_arr=gp_input_0, no_noise=no_noise, return_op=True, noise_verbose=False,
                                                      mask_dict_override=delta_dict, ret_noise_sample=False)
    # print([str(region) for region in true_func_obj.regions])
    # print(x_0, sampled_residual, delta_dict)
    # # print(ct_dyn_nom(x=x_0, u=u_0)['f'])
    # # print(dt*(ct_dyn_nom(x=x_0, u=u_0)['f'] + Bd @ sampled_residual.numpy()))
    # print(x_0 + dt*(ct_dyn_nom(x=x_0, u=u_0)['f'] + Bd @ sampled_residual.numpy()))


    # Sampled residual = region_res_mean + w_k where w_k is sampled for region stochasticity function (assumed here to be gaussian).
    if integration_method == 'euler':
        sampled_ns = x_0 + dt*(ct_dyn_nom(x=x_0, u=u_0)['f'] + Bd @ sampled_residual.numpy())
    else:
        sampled_ns = dt_dyn_nom(x=x_0, u=u_0)['xf'] + (dt * Bd @ sampled_residual.numpy())
    sampled_ns_clipped = clip_to_region_fn(sampled_ns) if clip_state is not None else sampled_ns
    if np.isclose(sampled_ns_clipped, sampled_ns, atol=1e-4).all():
        collision = False
    else:
        # print("Collision detected. %s, %s" % (sampled_ns_clipped, sampled_ns))
        collision = True
    # print(x_0)
    # print(dt*(ct_dyn_nom(x=x_0, u=u_0)['f']))
    # print(dt*Bd @ sampled_residual.numpy())
    # print(sampled_ns)
    # print("Sampled residual: %s, dt*sampled res: %s" % (sampled_residual.numpy(), (dt*Bd @ sampled_residual.numpy())))

    ret = [sampled_ns_clipped]
    if ret_residual:
        ret = ret + [sampled_residual]
    if ret_collision_bool:
        ret = ret + [collision]
    return ret
