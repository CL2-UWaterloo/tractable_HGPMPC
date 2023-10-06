import matplotlib.pyplot as plt
import time

from smpc.controller_utils import OptiDebugger, construct_config_opts, construct_config_opts_minlp
import smpc.controller_utils as controller_utils
from mapping.util import Traj_DS, Mapping_DS
from mapping.nn_map import SoftLabelNet
from common.plotting_utils import save_fig
from mapping import ws_mappers
from common.data_save_utils import save_data, read_data
from .controller_classes import *

from sys_dyn.problem_setups import quad_2d_sys_1d_inp_res, planar_lti_1d_inp_res


def sanity_check_controller(x_init, velocity_override=0.7, early_termination=True, minimal_print=True,
                            N=2, closed_loop=False, simulation_length=5,
                            no_lam_p=False, hessian_approximation=False, jac_approx=False,
                            show_plot=True, plot_ol_traj=False,
                            problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True,
                            ignore_init_constr_check=False, add_scaling=False,
                            ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3,
                            track_gen_fn=test_quad_2d_track, waypoint_arr=((7, 0), (7, 7), (0, 7), (0, 0)),
                            x_threshold=7, z_threshold=7, x0_delim=2, x1_delim=4, x_min_threshold=-7, simulation_length_override=None,
                            save_desired_to_file=None, read_desired_from_file=None, online_N=None,
                            integration_method='euler', Q_override=None, R_override=None, debug_fine_forward=False):

    if problem_setup_fn is quad_2d_sys_1d_inp_res:
        sys_config_dict, inst_override = problem_setup_fn(velocity_limit_override=velocity_override,
                                                          x_threshold=x_threshold, z_threshold=z_threshold, x_min_threshold=x_min_threshold,
                                                          sampling_time=sampling_time, x0_delim=x0_delim, x1_delim=x1_delim,
                                                          ret_inst=True)
        limiter_names = ["x_threshold", "z_threshold"]
        assert integration_method == 'euler', "Euler integration only available for this system."
    elif problem_setup_fn is planar_lti_1d_inp_res:
        problem_setup_fn: planar_lti_1d_inp_res
        sys_config_dict, inst_override = problem_setup_fn(u_max=velocity_override, sampling_time=sampling_time,
                                                          x1_threshold=x_threshold, x2_threshold=z_threshold, x1_min_threshold=x_min_threshold,
                                                          x0_delim=x0_delim, x1_delim=x1_delim, ret_inst=True)
        limiter_names = ["x1_threshold", "x2_threshold"]
    else:
        sys_config_dict, inst_override = problem_setup_fn(ret_inst=True)

    if Q_override is not None:
        sys_config_dict["Q"] = Q_override
    if R_override is not None:
        sys_config_dict["R"] = R_override

    # print(sys_config_dict)
    lti = inst_override.lti
    if lti and integration_method == 'exact':
        if not inst_override.symbolic.exact_flag:
            print("Exact linearization not available for this system. Using Euler integration instead.")
            integration_method = 'euler'
    fn_config_dict = {"horizon_length": N if online_N is None else online_N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time, "x_init": x_init,
                      "lti": lti, "integration_method": integration_method}
    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts(print_level, add_monitor=False, early_termination=early_termination, hsl_solver=False,
                                 test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation,
                                 jac_approx=jac_approx)
    fn_config_dict["addn_solver_opts"] = opts

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = BaseGPMPC_sanity(**configs)

    # Pull relevant info from config dict.
    n_u = sys_config_dict["n_u"]

    # Nominal MPC solution for warmstarting.
    if read_desired_from_file is not None:
        desired_traj_data = read_data(file_name=read_desired_from_file)
        x_desired = desired_traj_data["x_desired"]
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(x_desired[0, :], x_desired[2, :], color='cyan',
        #         label='Discretized trajectory to track', marker='o', linewidth=3, markersize=10)
        u_warmstart = desired_traj_data["u_desired"]
        assert desired_traj_data["N"] >= online_N, "Desired trajectory has horizon length less than online horizon length."
    else:
        x_desired, u_warmstart = track_gen_fn(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                              viz=False, verbose=verbose, N=N, velocity_override=velocity_override,
                                              ret_inputs=True, sampling_time=sampling_time, x_init=x_init,
                                              waypoints_arr=waypoint_arr, inst_override=inst_override, integration=integration_method,
                                              Q_override=sys_config_dict["Q"], R_override=sys_config_dict["R"])
        if save_desired_to_file is not None:
            desired_traj_dict = {"x_desired": x_desired, "u_desired": u_warmstart, "waypoint_arr": waypoint_arr,
                                 "sampling_time": sampling_time, "num_discrete": num_discrete, "N": N,
                                 "velocity_override": velocity_override, "x_init": x_init, "sim_steps": simulation_length,
                                 "x_threshold": x_threshold, "z_threshold": z_threshold, "x0_delim": x0_delim,
                                 "x1_delim": x1_delim}
            save_data(data=desired_traj_dict, file_name=save_desired_to_file)

    u_warmstart = np.array(u_warmstart, ndmin=2).reshape((n_u, -1))
    np.set_printoptions(threshold=np.inf)
    print(u_warmstart)
    print(x_desired)
    np.set_printoptions(threshold=1000)
    tracking_matrix = sys_config_dict["tracking_matrix"]
    waypoints_to_track = tracking_matrix[:, :controller_inst.n_x] @ x_desired
    initial_info_dict = {"x_init": x_init, "u_warmstart": u_warmstart}

    if debug_fine_forward:
        x_test, u_test = np.array([[0.5, 0.5]]).T, np.array([[0.5, 0.5004]]).T
        ns_coarse = inst_override.symbolic.fd_linear_func_exact(x_test, u_test)
        start_state = x_test
        for i in range(int(sampling_time//1e-3)):
            print(start_state)
            start_state = inst_override.symbolic.fd_linear_func_exact_1ms(start_state, u_test)
        print(ns_coarse)
        print(start_state)


    if not closed_loop:
        debugger_inst = OptiDebugger(controller_inst)
        sol = controller_inst.run_ol_opt(initial_info_dict, x_desired=x_desired)

        if show_plot:
            fig, ax = plt.subplots(1, 1)
            controller_utils.plot_OL_opt_soln(debugger_inst=debugger_inst,
                                              state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                              ax=ax, waypoints_to_track=waypoints_to_track,
                                              ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
    else:
        simulation_length = simulation_length_override if simulation_length_override is not None else simulation_length
        data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                opt_verbose=verbose, x_desired=x_desired)

        if show_plot:
            mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                        data_dict_cl=data_dict_cl,
                                                        ret_mu_x_cl=True,
                                                        state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                        plot_ol_traj=plot_ol_traj, axes=None,
                                                        ax_xlim=(-0.05, sys_config_dict[limiter_names[0]]+0.1),
                                                        ax_ylim=(-0.05, sys_config_dict[limiter_names[1]]+0.1),
                                                        legend_loc='upper center', ignore_legend=True)


def gpmpc_BnC_controller(gp_fns, x_init, velocity_override=0.7, early_termination=True, minimal_print=True,
                         N=2, closed_loop=False, simulation_length=5,
                         no_lam_p=False, hessian_approximation=False, jac_approx=False,
                         show_plot=True, plot_ol_traj=False,
                         problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True, track_gen_fn=test_quad_2d_track,
                         ignore_init_constr_check=False, add_scaling=False,
                         ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3, add_delta_constraints=True,
                         fwd_sim="nom_dyn", true_ds_inst=None, include_res_in_ctrl=False,
                         infeas_debug_callback=False, collision_check=False, waypoint_arr=((7, 0), (7, 7), (0, 7), (0, 0)),
                         x_threshold=7, z_threshold=7, x0_delim=2, x1_delim=4, x_min_threshold=-7, simulation_length_override=None,
                         read_desired_from_file=None, online_N=None, integration_method='euler', use_prev_if_infeas=False,
                         Q_override=None, R_override=None, Bd_override=None, gp_input_mask_override=None):
    if problem_setup_fn is quad_2d_sys_1d_inp_res:
        sys_config_dict, inst_override = problem_setup_fn(velocity_limit_override=velocity_override,
                                                          x_threshold=x_threshold, z_threshold=z_threshold, x_min_threshold=x_min_threshold,
                                                          sampling_time=sampling_time, x0_delim=x0_delim, x1_delim=x1_delim,
                                                          ret_inst=True)
        limiter_names = ["x_threshold", "z_threshold"]
        assert integration_method == 'euler', "Euler integration only available for this system."
    elif problem_setup_fn is planar_lti_1d_inp_res:
        problem_setup_fn: planar_lti_1d_inp_res
        sys_config_dict, inst_override = problem_setup_fn(u_max=velocity_override, sampling_time=sampling_time,
                                                          x1_threshold=x_threshold, x2_threshold=z_threshold, x1_min_threshold=x_min_threshold,
                                                          x0_delim=x0_delim, x1_delim=x1_delim, ret_inst=True)
        limiter_names = ["x1_threshold", "x2_threshold"]
    else:
        sys_config_dict, inst_override = problem_setup_fn(ret_inst=True)

    true_ds_inst.regions = sys_config_dict["regions"]

    if Q_override is not None:
        sys_config_dict["Q"] = Q_override
    if R_override is not None:
        sys_config_dict["R"] = R_override
    if Bd_override is not None:
        sys_config_dict["Bd"] = Bd_override
    if gp_input_mask_override is not None:
        sys_config_dict["gp_input_mask"] = gp_input_mask_override
    # print(sys_config_dict)
    lti = inst_override.lti
    if lti and integration_method == 'exact':
        if not inst_override.symbolic.exact_flag:
            print("Exact linearization not available for this system. Using Euler integration instead.")
            integration_method = 'euler'

    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N if online_N is None else online_N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": problem_setup_fn()["Bd"],
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "collision_check": collision_check, "lti": lti, "integration_method": integration_method,
                      "use_prev_if_infeas": use_prev_if_infeas}

    # Neglects residual mean dynamics in the controller. Variance info already excluded from the formulation for these increments.
    if not include_res_in_ctrl:
        sys_config_dict.update({"Bd": np.zeros((sys_config_dict["n_x"], 1))})
    # sys_config_dict.update({"Bd": np.array([[0, 1, 0, 0, 0, 0]]).T})

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)
    print(configs["x0_delim"], configs["x1_delim"])

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts(print_level, add_monitor=False, early_termination=early_termination, hsl_solver=False,
                                 test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation,
                                 jac_approx=jac_approx)
    fn_config_dict["addn_solver_opts"] = opts

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = GPMPC_BnC(**configs)

    # Pull relevant info from config dict.
    n_u = sys_config_dict["n_u"]

    if read_desired_from_file is not None:
        desired_traj_data = read_data(file_name=read_desired_from_file)
        x_desired = desired_traj_data["x_desired"]
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(x_desired[0, :], x_desired[2, :], color='cyan',
        #         label='Discretized trajectory to track', marker='o', linewidth=3, markersize=10)
        u_warmstart = desired_traj_data["u_desired"]
        assert desired_traj_data["N"] >= online_N, "Desired trajectory has horizon length less than online horizon length."
    else:
        # Nominal MPC solution for warmstarting.
        x_desired, u_warmstart = track_gen_fn(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                              viz=False, verbose=verbose, N=N, velocity_override=velocity_override,
                                              ret_inputs=True, sampling_time=sampling_time, waypoints_arr=waypoint_arr,
                                              inst_override=inst_override, integration=integration_method)

    u_warmstart = np.array(u_warmstart, ndmin=2).reshape((n_u, -1))
    tracking_matrix = sys_config_dict["tracking_matrix"]
    waypoints_to_track = tracking_matrix[:, :controller_inst.n_x] @ x_desired
    initial_info_dict = {"x_init": x_init, "u_warmstart": u_warmstart}

    # print(controller_inst.opti.)

    if not closed_loop:
        debugger_inst = OptiDebugger(controller_inst)
        sol = controller_inst.run_ol_opt(initial_info_dict, x_desired=x_desired)

        if show_plot:
            fig, ax = plt.subplots(1, 1)
            controller_utils.plot_OL_opt_soln(debugger_inst=debugger_inst,
                                              state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                              ax=ax, waypoints_to_track=waypoints_to_track,
                                              ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
    else:
        simulation_length = simulation_length_override if simulation_length_override is not None else simulation_length
        data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                opt_verbose=verbose, x_desired=x_desired, u_desired=u_warmstart,
                                                                infeas_debug_callback=infeas_debug_callback)

        total_run_time = 0
        for stat_dict in sol_stats_cl:
            run_time = stat_dict['t_wall_total']
            total_run_time += run_time
        average_run_time = total_run_time / len(sol_stats_cl)
        print("Average run time: %s" % average_run_time)

        mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
        x_cl_for_cost = np.hstack([data_dict_ol['mu_x'][:, [0]] for data_dict_ol in data_dict_cl])
        x_dev = x_cl_for_cost - x_desired[:, :x_cl_for_cost.shape[1]]
        cl_cost = 0
        for i in range(x_dev.shape[1]):
            cl_cost += x_dev[:, [i]].T @ sys_config_dict["Q"] @ x_dev[:, [i]] + mu_u_cl[:, [i]].T @ sys_config_dict["R"] @ mu_u_cl[:, [i]]
        print("Closed loop cost: %s" % cl_cost)

        mu_x_cl = [data_dict_ol['mu_x'][:, 0] for data_dict_ol in data_dict_cl]
        # print([x_desired[:, i] for i in range(30, 50)])
        # print(mu_x_cl[30:50])

        if show_plot:
            mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                        data_dict_cl=data_dict_cl,
                                                        ret_mu_x_cl=True,
                                                        state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                        plot_ol_traj=plot_ol_traj, axes=None, regions=controller_inst.regions,
                                                        ax_xlim=(-0.05, sys_config_dict[limiter_names[0]]+0.1),
                                                        ax_ylim=(-0.05, sys_config_dict[limiter_names[1]]+0.1),
                                                        legend_loc='upper center')
            if collision_check:
                print('NUMBER OF COLLISIONS: %s' % controller_inst.collision_counter)


def gpmpc_D_controller(gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7, early_termination=True,
                       minimal_print=True, plot_shrunk_sets=False, skip_shrinking=False,
                       N=2, closed_loop=False, simulation_length=5,
                       no_lam_p=False, hessian_approximation=True, jac_approx=False,
                       show_plot=True, plot_ol_traj=False,
                       problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True, track_gen_fn=test_quad_2d_track,
                       ignore_init_constr_check=False, add_scaling=False,
                       ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3,
                       x_threshold=7, z_threshold=7, x0_delim=2, x1_delim=4, x_min_threshold=-7,
                       add_delta_constraints=False, fwd_sim="w_pw_res", true_ds_inst=None, include_res_in_ctrl=False,
                       infeas_debug_callback=False, plot_shrunk_sep=False, waypoint_arr=((7, 0), (7, 7), (0, 7), (0, 0)), collision_check=False,
                       simulation_length_override=None, boole_deno_override=False, integration_method="euler",
                       online_N=None, read_desired_from_file=None, use_prev_if_infeas=False,
                       Q_override=None, R_override=None, Bd_override=None, ret_config_dict=False):
    if problem_setup_fn is quad_2d_sys_1d_inp_res:
        sys_config_dict, inst_override = problem_setup_fn(velocity_limit_override=velocity_override,
                                                          x_threshold=x_threshold, z_threshold=z_threshold, x_min_threshold=x_min_threshold,
                                                          sampling_time=sampling_time, x0_delim=x0_delim, x1_delim=x1_delim,
                                                          ret_inst=True)
        limiter_names = ["x_threshold", "z_threshold"]
        assert integration_method == 'euler', "Euler integration only available for this system."
    elif problem_setup_fn is planar_lti_1d_inp_res:
        problem_setup_fn: planar_lti_1d_inp_res
        sys_config_dict, inst_override = problem_setup_fn(u_max=velocity_override, sampling_time=sampling_time,
                                                          x1_threshold=x_threshold, x2_threshold=z_threshold,
                                                          x0_delim=x0_delim, x1_delim=x1_delim, ret_inst=True)
        limiter_names = ["x1_threshold", "x2_threshold"]
    else:
        sys_config_dict, inst_override = problem_setup_fn(ret_inst=True)

    true_ds_inst.regions = sys_config_dict["regions"]

    if Q_override is not None:
        sys_config_dict["Q"] = Q_override
    if R_override is not None:
        sys_config_dict["R"] = R_override
    if Bd_override is not None:
        sys_config_dict["Bd"] = Bd_override
    # print(sys_config_dict)
    lti = inst_override.lti
    if lti and integration_method == 'exact':
        if not inst_override.symbolic.exact_flag:
            print("Exact linearization not available for this system. Using Euler integration instead.")
            integration_method = 'euler'
    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N if online_N is None else online_N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": sys_config_dict["Bd"],
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "satisfaction_prob": satisfaction_prob, "skip_shrinking": skip_shrinking,
                      "collision_check": collision_check, "boole_deno_override": boole_deno_override,
                      "lti": lti, "integration_method": integration_method, "use_prev_if_infeas": use_prev_if_infeas}
    np.random.seed(np.random.randint(0, 1000000))

    # Neglects residual mean dynamics in the controller. Variance info already excluded from the formulation for these increments.
    if not include_res_in_ctrl:
        sys_config_dict.update({"Bd": np.zeros((6, 1))})
    # sys_config_dict.update({"Bd": np.array([[0, 1, 0, 0, 0, 0]]).T})

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    if ret_config_dict:
        return configs

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts(print_level, add_monitor=False, early_termination=early_termination, hsl_solver=False,
                                 test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation,
                                 jac_approx=jac_approx)
    fn_config_dict["addn_solver_opts"] = opts

    # Pull relevant info from config dict.
    n_u = sys_config_dict["n_u"]

    # Nominal MPC solution for warmstarting.
    if read_desired_from_file is not None:
        desired_traj_data = read_data(file_name=read_desired_from_file)
        x_desired = desired_traj_data["x_desired"]
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(x_desired[0, :], x_desired[2, :], color='cyan',
        #         label='Discretized trajectory to track', marker='o', linewidth=3, markersize=10)
        u_desired = desired_traj_data["u_desired"]
        assert desired_traj_data["N"] >= online_N, "Desired trajectory has horizon length less than online horizon length."
    else:
        # Nominal MPC solution for warmstarting.
        x_desired, u_desired = track_gen_fn(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                            viz=False, verbose=verbose, N=N, velocity_override=velocity_override,
                                            ret_inputs=True, sampling_time=sampling_time, waypoints_arr=waypoint_arr,
                                            inst_override=inst_override, integration=integration_method, x_init=x_init)

    u_desired = np.array(u_desired, ndmin=2).reshape((n_u, -1))
    tracking_matrix = sys_config_dict["tracking_matrix"]
    waypoints_to_track = tracking_matrix[:, :sys_config_dict["n_x"]] @ x_desired
    initial_info_dict = {"x_init": x_init, "u_warmstart": u_desired}

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = GPMPC_D(**configs)

    if not closed_loop:
        debugger_inst = OptiDebugger(controller_inst)
        sol = controller_inst.run_ol_opt(initial_info_dict, x_desired=x_desired, u_desired=u_desired,
                                         plot_shrunk_sets=plot_shrunk_sets, plot_shrunk_sep=plot_shrunk_sep)
        print("PRINTING SOL STATS")
        print(sol.stats())
        print("Total time for iteration %s" % sol.stats()['t_wall_total'])

        if show_plot:
            fig, ax = plt.subplots(1, 1)
            controller_utils.plot_OL_opt_soln(debugger_inst=debugger_inst,
                                              state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                              ax=ax, waypoints_to_track=waypoints_to_track,
                                              ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
    else:
        simulation_length = simulation_length_override if simulation_length_override is not None else simulation_length
        data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                opt_verbose=verbose, x_desired=x_desired, u_desired=u_desired,
                                                                infeas_debug_callback=infeas_debug_callback)

        total_run_time = 0
        for stat_dict in sol_stats_cl:
            run_time = stat_dict['t_wall_total']
            total_run_time += run_time
        average_run_time = total_run_time / len(sol_stats_cl)
        print("Average run time: %s" % average_run_time)

        mu_x_cl = [data_dict_ol['mu_x'] for data_dict_ol in data_dict_cl]
        # mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
        # print(mu_u_cl)
        # theta_cl = [data_dict_ol['mu_x'][:, 0][4] for data_dict_ol in data_dict_cl]
        # print(theta_cl)
        mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
        x_cl_for_cost = np.hstack([data_dict_ol['mu_x'][:, [0]] for data_dict_ol in data_dict_cl])
        x_dev = x_cl_for_cost - x_desired[:, :x_cl_for_cost.shape[1]]
        cl_cost = 0
        for i in range(x_dev.shape[1]):
            cl_cost += x_dev[:, [i]].T @ sys_config_dict["Q"] @ x_dev[:, [i]] + mu_u_cl[:, [i]].T @ sys_config_dict["R"] @ mu_u_cl[:, [i]]
        print("Closed loop cost: %s" % cl_cost)

        data_to_store = {"average_run_time": average_run_time, "collision_count": controller_inst.collision_counter,
                         "collision_idxs": controller_inst.collision_idxs, "cl_cost": cl_cost,
                         "data_dict_cl": data_dict_cl, "sol_stats_cl": sol_stats_cl}
        print('PRINTING RUN STATISTICS')
        print(data_to_store["average_run_time"])
        print(data_to_store["collision_count"])
        print(data_to_store["collision_idxs"])
        print(data_to_store["cl_cost"])

        save_data(data_to_store, file_name="horizon_qualit_N24", update_data=True)

        if show_plot:

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                        data_dict_cl=data_dict_cl,
                                                        ret_mu_x_cl=True,
                                                        state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                        plot_ol_traj=plot_ol_traj, axes=[ax],
                                                        ax_xlim=(-0.05, sys_config_dict[limiter_names[0]]+0.1),
                                                        ax_ylim=(-0.05, sys_config_dict[limiter_names[1]]+0.1),
                                                        # ax_xlim=(-0.05, sys_config_dict["x_threshold"]), ax_ylim=(-0.05, sys_config_dict["z_threshold"]),
                                                        legend_loc='upper center',
                                                        regions=controller_inst.regions, ignore_legend=True)
            # save_fig(axes=[ax], fig_name='boundary_tracking_nlp', tick_sizes=14, tick_skip=1, k_range=None)

            if collision_check:
                print('NUMBER OF COLLISIONS: %s' % controller_inst.collision_counter)

            # fig, axes = plt.subplots(2, 1)
            # k_range = [i for i in range(mu_u_cl.shape[1])]
            # axes[0].plot(k_range, mu_u_cl[0, :], label="u_0")
            # axes[1].plot(k_range, mu_u_cl[1, :]-mu_u_cl[0, :], label="u_0")
            # for i in range(2):
            #     axes[i].tick_params(axis='both', labelsize=14)
            #     axes[i].set_xticks(k_range[::2])


def gpmpc_QP_controller(gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7, early_termination=True,
                        minimal_print=True, plot_shrunk_sets=False, skip_shrinking=False,
                        N=2, closed_loop=False, simulation_length=5,
                        no_lam_p=False, hessian_approximation=True, jac_approx=False,
                        show_plot=True, plot_ol_traj=False,
                        problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True, track_gen_fn=test_quad_2d_track,
                        ignore_init_constr_check=False, add_scaling=False,
                        ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3,
                        x_threshold=7, z_threshold=7, x0_delim=2, x1_delim=4, x_min_threshold=-7, add_delta_constraints=False,
                        fwd_sim="nom_dyn", true_ds_inst=None, include_res_in_ctrl=False,
                        infeas_debug_callback=False, plot_shrunk_sep=False, waypoint_arr=((7, 0), (7, 7), (0, 7), (0, 0)), collision_check=False,
                        boole_deno_override=False, integration_method='euler', simulation_length_override=None, debug_ltv=False,
                        online_N=None, read_desired_from_file=None, use_prev_if_infeas=False,
                        Q_override=None, R_override=None, Bd_override=None):
    if problem_setup_fn is quad_2d_sys_1d_inp_res:
        sys_config_dict, inst_override = problem_setup_fn(velocity_limit_override=velocity_override,
                                                          x_threshold=x_threshold, z_threshold=z_threshold, x_min_threshold=x_min_threshold,
                                                          sampling_time=sampling_time, x0_delim=x0_delim, x1_delim=x1_delim,
                                                          ret_inst=True)
        limiter_names = ["x_threshold", "z_threshold"]
        assert integration_method == 'euler', "Euler integration only available for this system."
        # print(limiter_names)
    elif problem_setup_fn is planar_lti_1d_inp_res:
        problem_setup_fn: planar_lti_1d_inp_res
        sys_config_dict, inst_override = problem_setup_fn(u_max=velocity_override, sampling_time=sampling_time,
                                                          x1_threshold=x_threshold, x2_threshold=z_threshold, x1_min_threshold=x_min_threshold,
                                                          x0_delim=x0_delim, x1_delim=x1_delim, ret_inst=True)
        limiter_names = ["x1_threshold", "x2_threshold"]
        # print(limiter_names)
    else:
        sys_config_dict, inst_override = problem_setup_fn(ret_inst=True)

    true_ds_inst.regions = sys_config_dict["regions"]

    lti = inst_override.lti
    if lti and integration_method == 'exact':
        if not inst_override.symbolic.exact_flag:
            print("Exact linearization not available for this system. Using Euler integration instead.")
            integration_method = 'euler'

    if Q_override is not None:
        sys_config_dict["Q"] = Q_override
    if R_override is not None:
        sys_config_dict["R"] = R_override
    if Bd_override is not None:
        sys_config_dict["Bd"] = Bd_override
    # print(sys_config_dict)
    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N if online_N is None else online_N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": sys_config_dict["Bd"],
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "satisfaction_prob": satisfaction_prob, "skip_shrinking": skip_shrinking,
                      "collision_check": collision_check, "boole_deno_override": boole_deno_override,
                      "integration_method": integration_method, "lti": lti, "use_prev_if_infeas": use_prev_if_infeas}
    np.random.seed(np.random.randint(0, 1000000))

    # Neglects residual mean dynamics in the controller. Variance info already excluded from the formulation for these increments.
    if not include_res_in_ctrl:
        sys_config_dict.update({"Bd": np.zeros((6, 1))})
    # sys_config_dict.update({"Bd": np.array([[0, 1, 0, 0, 0, 0]]).T})

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts(print_level, add_monitor=False, early_termination=early_termination, hsl_solver=False,
                                 test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation,
                                 jac_approx=jac_approx)
    fn_config_dict["addn_solver_opts"] = opts

    # Pull relevant info from config dict.
    n_u = sys_config_dict["n_u"]

    # Nominal MPC solution for warmstarting.
    if read_desired_from_file is not None:
        desired_traj_data = read_data(file_name=read_desired_from_file)
        x_desired = desired_traj_data["x_desired"]
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(x_desired[0, :], x_desired[2, :], color='cyan',
        #         label='Discretized trajectory to track', marker='o', linewidth=3, markersize=10)
        u_desired = desired_traj_data["u_desired"]
        assert desired_traj_data["N"] >= online_N, "Desired trajectory has horizon length less than online horizon length."
    else:
        # Nominal MPC solution for warmstarting.
        x_desired, u_desired = track_gen_fn(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                            viz=False, verbose=verbose, N=N, velocity_override=velocity_override,
                                            ret_inputs=True, sampling_time=sampling_time, waypoints_arr=waypoint_arr,
                                            inst_override=inst_override, integration=integration_method, x_init=x_init)

    u_desired = np.array(u_desired, ndmin=2).reshape((n_u, -1))
    tracking_matrix = sys_config_dict["tracking_matrix"]
    waypoints_to_track = tracking_matrix[:, :sys_config_dict["n_x"]] @ x_desired
    initial_info_dict = {"x_init": x_init, "u_warmstart": u_desired}

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = GPMPC_QP(**configs)

    if debug_ltv:
        single_idx = False
        if single_idx:
            controller_inst.set_traj_to_track(x_desired, u_desired)
            controller_inst.generate_hld_for_warmstart()
            controller_inst.setup_resmean_warmstart_vec()
            controller_inst.shrink_by_desired(simulation_length)
            dt = controller_inst.dt
            idx = 24
            u = u_desired[:, idx]
            # u = np.zeros((2, 1)).squeeze()
            print(x_desired[:, idx], u)
            xplus_euler = x_desired[:, idx] + dt * (controller_inst.ct_dyn_nom(x=x_desired[:, idx], u=u)['f'])
            # xplus_euler = x_desired[:, idx] + dt * (controller_inst.ct_dyn_nom(x=x_desired[:, idx], u=u)['f'] + controller_inst.Bd @ controller_inst.resmean_warmstart[:, idx])
            xplus_rk4 = controller_inst.sys_dyn.rk4(x=x_desired[:, idx], u=u)['f']
            # xplus_rk4 = controller_inst.sys_dyn.rk4(x=x_desired[:, idx], u=u)['f'] + (dt * controller_inst.Bd @ controller_inst.resmean_warmstart[:, idx])
            A_k, B_k = controller_inst.linearize_and_discretize(x_desired[:, idx], u, cs_equivalent=True, method='zoh')
            xplus_zoh = A_k @ x_desired[:, idx] + B_k @ u
            # xplus_zoh = A_k @ x_desired[:, idx] + B_k @ u + (dt * controller_inst.Bd @ controller_inst.resmean_warmstart[:, idx])
            # partial_der_calc = controller_inst.sys_dyn.df_func(x_desired[:, idx], u)
            # print(partial_der_calc)
            # A_c, B_c = np.array(partial_der_calc[0]), np.array(partial_der_calc[1])
            # A_k_ex = scipy.linalg.expm(A_c * dt)
            # B_k_ex = np.linalg.inv(A_c) @ (A_k_ex - np.eye(A_c.shape[0])) @ B_c
            # xplus_exact = A_k_ex @ x_desired[:, 10] + B_k_ex @ u_desired[:, 10] + (dt * controller_inst.Bd @ controller_inst.resmean_warmstart[:, 10])
            # print(xplus_euler, xplus_zoh, xplus_exact)
            print("rk4, euler, zoh", xplus_rk4, xplus_euler, xplus_zoh)
            # partial_der_calc = controller_inst.sys_dyn.df_func(x_desired[:, idx], u)
            # print(partial_der_calc)
            # # print(quad_2D_dyn().symbolic.fd_linear_func(x_desired[:, idx], u))
            # # F = quad_2D_dyn().symbolic.fd_linear_func
            # dfdx = partial_der_calc['dfdx'].toarray()
            # dfdu = partial_der_calc['dfdu'].toarray()
            # next_state = quad_2D_dyn(velocity_limit_override=5).symbolic.linear_dynamics_func(x0=x_desired[:, idx], p=u)['xf']
            # print(next_state)
            partial_der_calc = controller_inst.sys_dyn.df_func(x_desired[:, idx], u)
            A_c, B_c = np.array(partial_der_calc[0]), np.array(partial_der_calc[1])
            print("ct_mats", A_c, B_c)
            theta_curr = x_desired[4, idx]
            u1_curr = u[0]
            u2_curr = u[1]
            m, g, l = 0.027, 9.8, 0.0397
            Iyy = 1.4e-5
            A_c_manual = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, cs.cos(theta_curr) * (u1_curr + u2_curr) / m, 0],
                                   [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, -cs.sin(theta_curr) * (u1_curr + u2_curr) / m, 0],
                                   [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]])
            B_c_manual = np.array([[0, 0], [cs.sin(theta_curr)/m, cs.sin(theta_curr)/m],
                                   [0, 0], [cs.cos(theta_curr)/m, cs.cos(theta_curr)/m],
                                   [0, 0], [-l/(1.414*Iyy), l/(1.414*Iyy)]])
            print("manual_cts", A_c_manual, B_c_manual)
            state_dim, input_dim, A, B = controller_inst.n_x, n_u, A_c, B_c
            M = np.zeros((state_dim + input_dim, state_dim + input_dim))
            M[:state_dim, :state_dim] = A
            M[:state_dim, state_dim:] = B

            Md = scipy.linalg.expm(M * dt)
            Ad = Md[:state_dim, :state_dim]
            # print(Ad, A_k)
            Bd = Md[:state_dim, state_dim:]
            # print(Bd, B_k)
            # print(Ad.shape, Bd.shape, A_k.shape, B_k.shape)
            # print((Ad @ x_desired[:, idx]).shape)
            # print((Bd @ u).shape)
            xplus_exact = Ad @ x_desired[:, idx] + Bd @ u
            # xplus_exact = Ad @ x_desired[:, idx] + Bd @ u + (dt * controller_inst.Bd @ controller_inst.resmean_warmstart[:, idx])
            print("exact", xplus_exact)
            print("cvodes_ct_sym", controller_inst.sys_dyn.fd_func)
            print("cvodes_ct_evald", controller_inst.sys_dyn.fd_func(x_desired[:, idx], u, 0, 0, 0, 0))
            # print(controller_inst.sys_dyn.f_disc_linear_func(x_desired[:, idx], u))
            disc_lind = controller_inst.sys_dyn.f_disc_linear_func(x_desired[:, idx], u)
            Ad, Bd = disc_lind[0], disc_lind[1]
            print("dt_mats_cvodes", Ad, Bd)
            print("dt_mats_manual", np.eye(controller_inst.n_x) + dt * A_c_manual, dt * B_c_manual)
            xplus_cvodes_lind = Ad @ x_desired[:, idx] + Bd @ u
            print("cvodes_linearized_evald", xplus_cvodes_lind)
        else:
            dt = controller_inst.dt
            cvodes_lind_errors = []
            euler_errors = []
            zoh_errors = []
            rk4_errors = []
            for idx in range(simulation_length):
                u = u_desired[:, idx]
                # u = np.zeros((2, 1)).squeeze()

                # ct integrated using casadi built-in
                cvodes_ct_evald = controller_inst.sys_dyn.fd_func(x_desired[:, idx], u, 0, 0, 0, 0)[0]

                # discretized version of ct integrated method. Computes jacobian of the integration function that has already incorporated sampling time.
                disc_lind = controller_inst.sys_dyn.f_disc_linear_func(x_desired[:, idx], u)
                Ad, Bd = disc_lind[0], disc_lind[1]
                xplus_cvodes_lind = Ad @ x_desired[:, idx] + Bd @ u

                # Euler integration non-linear
                xplus_euler = x_desired[:, idx] + dt * (controller_inst.ct_dyn_nom(x=x_desired[:, idx], u=u)['f'])

                # RK4 integration non-linear
                xplus_rk4 = controller_inst.sys_dyn.rk4(x=x_desired[:, idx], u=u)['f']

                # Zoh system without cvodes
                A_dk, B_dk = controller_inst.linearize_and_discretize(x_desired[:, idx], u, cs_equivalent=True, method='zoh')
                xplus_zoh = A_dk @ x_desired[:, idx] + B_dk @ u

                cvodes_lind_errors.append(np.array((cvodes_ct_evald - xplus_cvodes_lind).T))
                euler_errors.append(np.array((cvodes_ct_evald - xplus_euler).T))
                zoh_errors.append(np.array((cvodes_ct_evald - xplus_zoh).T))
                rk4_errors.append(np.array((cvodes_ct_evald - xplus_rk4).T))
            cvodes_lind_errors = np.fabs(np.vstack(cvodes_lind_errors))
            euler_errors = np.fabs(np.vstack(euler_errors))
            zoh_errors = np.fabs(np.vstack(zoh_errors))
            rk4_errors = np.fabs(np.vstack(rk4_errors))
            print("CVODES AVG LIND ERRORS BY COLUMN")
            print(np.mean(cvodes_lind_errors, axis=0))
            print("CVODES MAX LIND ERRORS BY COLUMN")
            print(np.max(cvodes_lind_errors, axis=0))
            print("EULER AVG NONLIN ERRORS BY COLUMN")
            print(np.mean(euler_errors, axis=0))
            print("EULER MAX NONLIN ERRORS BY COLUMN")
            print(np.max(euler_errors, axis=0))
            print("ZOH AVG LIND ERRORS BY COLUMN")
            print(np.mean(zoh_errors, axis=0))
            print("ZOH MAX LIND ERRORS BY COLUMN")
            print(np.max(zoh_errors, axis=0))
            print("RK4 AVG NONLIN ERRORS BY COLUMN")
            print(np.mean(rk4_errors, axis=0))
            print("RK4 MAX NONLIN ERRORS BY COLUMN")
            print(np.max(rk4_errors, axis=0))
            #
            # print(cvodes_lind_errors)
            # print(euler_errors)
            # print(zoh_errors)


    else:
        if not closed_loop:
            debugger_inst = OptiDebugger(controller_inst)
            sol = controller_inst.run_ol_opt(initial_info_dict, x_desired=x_desired, u_desired=u_desired,
                                             plot_shrunk_sets=plot_shrunk_sets, plot_shrunk_sep=plot_shrunk_sep)
            print("PRINTING SOL STATS")
            print(sol.stats())
            print("Total time for iteration %s" % sol.stats()['t_wall_total'])

            if show_plot:
                fig, ax = plt.subplots(1, 1)
                controller_utils.plot_OL_opt_soln(debugger_inst=debugger_inst,
                                                  state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                  ax=ax, waypoints_to_track=waypoints_to_track,
                                                  ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
        else:
            simulation_length = simulation_length_override if simulation_length_override is not None else simulation_length
            data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                    opt_verbose=verbose, x_desired=x_desired, u_desired=u_desired,
                                                                    infeas_debug_callback=infeas_debug_callback)

            total_run_time = 0
            for stat_dict in sol_stats_cl:
                run_time = stat_dict['t_wall_total']
                total_run_time += run_time
            average_run_time = total_run_time / len(sol_stats_cl)
            print("Average run time: %s" % average_run_time)

            mu_x_cl = [data_dict_ol['mu_x'] for data_dict_ol in data_dict_cl]
            # mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
            # print(mu_u_cl)
            # theta_cl = [data_dict_ol['mu_x'][:, 0][4] for data_dict_ol in data_dict_cl]
            # print(theta_cl)
            mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
            x_cl_for_cost = np.hstack([data_dict_ol['mu_x'][:, [0]] for data_dict_ol in data_dict_cl])
            x_dev = x_cl_for_cost - x_desired[:, :x_cl_for_cost.shape[1]]
            print(np.mean(x_dev, axis=1))
            cl_cost = 0
            for i in range(x_dev.shape[1]):
                cl_cost += x_dev[:, [i]].T @ sys_config_dict["Q"] @ x_dev[:, [i]] + mu_u_cl[:, [i]].T @ sys_config_dict["R"] @ mu_u_cl[:, [i]]
            print("Closed loop cost: %s" % cl_cost)

            if show_plot:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                            data_dict_cl=data_dict_cl,
                                                            ret_mu_x_cl=True,
                                                            state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                            plot_ol_traj=plot_ol_traj, axes=[ax],
                                                            ax_xlim=(-0.05, sys_config_dict[limiter_names[0]]+0.1),
                                                            ax_ylim=(-0.05, sys_config_dict[limiter_names[1]]+0.1),
                                                            legend_loc='upper center',
                                                            regions=controller_inst.regions, ignore_legend=True)
                # save_fig(axes=[ax], fig_name='boundary_tracking_nlp', tick_sizes=14, tick_skip=1, k_range=None)

                if collision_check:
                    print('NUMBER OF COLLISIONS: %s' % controller_inst.collision_counter)

                # fig, axes = plt.subplots(2, 1)
                # k_range = [i for i in range(mu_u_cl.shape[1])]
                # axes[0].plot(k_range, mu_u_cl[0, :], label="u_0")
                # axes[1].plot(k_range, mu_u_cl[1, :]-mu_u_cl[0, :], label="u_0")
                # for i in range(2):
                #     axes[i].tick_params(axis='both', labelsize=14)
                #     axes[i].set_xticks(k_range[::2])


def gpmpc_BnC_controller_multiple(gp_fns, x_init, velocity_override=0.7, early_termination=True, minimal_print=True,
                                  N=2, closed_loop=False, simulation_length=5,
                                  no_lam_p=False, hessian_approximation=False, jac_approx=False,
                                  show_plot=True, plot_ol_traj=False,
                                  problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True, track_gen_fn=test_quad_2d_track,
                                  ignore_init_constr_check=False, add_scaling=False,
                                  ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3, add_delta_constraints=True,
                                  fwd_sim="nom_dyn", true_ds_inst=None, include_res_in_ctrl=False,
                                  infeas_debug_callback=False, collision_check=False, waypoint_arr=((7, 0), (7, 7), (0, 7), (0, 0)),
                                  x_threshold=7, z_threshold=7, x0_delim=2, x1_delim=4, x_min_threshold=-7, simulation_length_override=None,
                                  read_desired_from_file=None, online_N=None, integration_method='euler', use_prev_if_infeas=False,
                                  Q_override=None, R_override=None,
                                  num_multiple=5, data_save_file='gpmpc_d_runs'):
    if problem_setup_fn is quad_2d_sys_1d_inp_res:
        sys_config_dict, inst_override = problem_setup_fn(velocity_limit_override=velocity_override,
                                                          x_threshold=x_threshold, z_threshold=z_threshold, x_min_threshold=x_min_threshold,
                                                          sampling_time=sampling_time, x0_delim=x0_delim, x1_delim=x1_delim,
                                                          ret_inst=True)
        limiter_names = ["x_threshold", "z_threshold"]
        assert integration_method == 'euler', "Euler integration only available for this system."
    elif problem_setup_fn is planar_lti_1d_inp_res:
        problem_setup_fn: planar_lti_1d_inp_res
        sys_config_dict, inst_override = problem_setup_fn(u_max=velocity_override, sampling_time=sampling_time,
                                                          x1_threshold=x_threshold, x2_threshold=z_threshold,
                                                          x0_delim=x0_delim, x1_delim=x1_delim, ret_inst=True)
        limiter_names = ["x1_threshold", "x2_threshold"]
    else:
        sys_config_dict, inst_override = problem_setup_fn(ret_inst=True)

    true_ds_inst.regions = sys_config_dict["regions"]

    if Q_override is not None:
        sys_config_dict["Q"] = Q_override
    if R_override is not None:
        sys_config_dict["R"] = R_override
    # print(sys_config_dict)
    lti = inst_override.lti
    if lti and integration_method == 'exact':
        if not inst_override.symbolic.exact_flag:
            print("Exact linearization not available for this system. Using Euler integration instead.")
            integration_method = 'euler'

    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N if online_N is None else online_N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": problem_setup_fn()["Bd"],
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "collision_check": collision_check, "lti": lti, "integration_method": integration_method,
                      "use_prev_if_infeas": use_prev_if_infeas}
    # Neglects residual mean dynamics in the controller. Variance info already excluded from the formulation for these increments.
    if not include_res_in_ctrl:
        sys_config_dict.update({"Bd": np.zeros((sys_config_dict["n_x"], 1))})
    # sys_config_dict.update({"Bd": np.array([[0, 1, 0, 0, 0, 0]]).T})

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts(print_level, add_monitor=False, early_termination=early_termination, hsl_solver=False,
                                 test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation,
                                 jac_approx=jac_approx)
    fn_config_dict["addn_solver_opts"] = opts

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = GPMPC_BnC(**configs)

    # Pull relevant info from config dict.
    n_u = sys_config_dict["n_u"]

    if read_desired_from_file is not None:
        desired_traj_data = read_data(file_name=read_desired_from_file)
        x_desired = desired_traj_data["x_desired"]
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(x_desired[0, :], x_desired[2, :], color='cyan',
        #         label='Discretized trajectory to track', marker='o', linewidth=3, markersize=10)
        u_warmstart = desired_traj_data["u_desired"]
        assert desired_traj_data["N"] >= online_N, "Desired trajectory has horizon length less than online horizon length."
    else:
        # Nominal MPC solution for warmstarting.
        x_desired, u_warmstart = track_gen_fn(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                              viz=False, verbose=verbose, N=N, velocity_override=velocity_override,
                                              ret_inputs=True, sampling_time=sampling_time, waypoints_arr=waypoint_arr,
                                              inst_override=inst_override, integration=integration_method)

    u_warmstart = np.array(u_warmstart, ndmin=2).reshape((n_u, -1))
    tracking_matrix = sys_config_dict["tracking_matrix"]
    waypoints_to_track = tracking_matrix[:, :controller_inst.n_x] @ x_desired
    initial_info_dict = {"x_init": x_init, "u_warmstart": u_warmstart}

    # print(controller_inst.opti.)

    data_store_dicts = []
    for run_idx in range(num_multiple):
        if not closed_loop:
            debugger_inst = OptiDebugger(controller_inst)
            sol = controller_inst.run_ol_opt(initial_info_dict, x_desired=x_desired)

            if show_plot:
                fig, ax = plt.subplots(1, 1)
                controller_utils.plot_OL_opt_soln(debugger_inst=debugger_inst,
                                                  state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                  ax=ax, waypoints_to_track=waypoints_to_track,
                                                  ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
        else:
            simulation_length = simulation_length_override if simulation_length_override is not None else simulation_length
            run_start_time = time.time()
            data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                    opt_verbose=verbose, x_desired=x_desired, u_desired=u_warmstart,
                                                                    infeas_debug_callback=infeas_debug_callback)

            mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
            x_cl_for_cost = np.hstack([data_dict_ol['mu_x'][:, [0]] for data_dict_ol in data_dict_cl])
            x_dev = x_cl_for_cost - x_desired[:, :x_cl_for_cost.shape[1]]
            cl_cost = 0
            for i in range(x_dev.shape[1]):
                cl_cost += x_dev[:, [i]].T @ sys_config_dict["Q"] @ x_dev[:, [i]] + mu_u_cl[:, [i]].T @ sys_config_dict["R"] @ mu_u_cl[:, [i]]
            # print("Closed loop cost: %s" % cl_cost)

            total_run_time = 0
            run_times = []
            for stat_dict in sol_stats_cl:
                run_time = stat_dict['t_wall_total']
                run_times.append(run_time)
                total_run_time += run_time
            average_run_time = total_run_time / len(sol_stats_cl)
            # print("Average run time: %s" % average_run_time)

            data_to_store = {"average_run_time": average_run_time, "run_times": run_times, "collision_count": controller_inst.collision_counter,
                             "collision_idxs": controller_inst.collision_idxs, "cl_cost": cl_cost,
                             "data_dict_cl": data_dict_cl, "sol_stats_cl": sol_stats_cl,
                             "run_start_time": run_start_time}
            print('PRINTING RUN STATISTICS')
            print(data_to_store["average_run_time"])
            print(data_to_store["collision_count"])
            print(data_to_store["collision_idxs"])
            print(data_to_store["cl_cost"])
            data_store_dicts.append(data_to_store)

            controller_inst.collision_counter = 0
            controller_inst.collision_idxs = []

            save_data(data_store_dicts, file_name=data_save_file, update_data=True)
            if show_plot:
                mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                            data_dict_cl=data_dict_cl,
                                                            ret_mu_x_cl=True,
                                                            state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                            plot_ol_traj=plot_ol_traj, axes=None,
                                                            ax_xlim=(-0.05, sys_config_dict[limiter_names[0]]+0.1),
                                                            ax_ylim=(-0.05, sys_config_dict[limiter_names[1]]+0.1),
                                                            legend_loc='upper center')
                if collision_check:
                    print('NUMBER OF COLLISIONS: %s' % controller_inst.collision_counter)


def gpmpc_QP_controller_multiple(gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7, early_termination=True,
                        minimal_print=True, plot_shrunk_sets=False, skip_shrinking=False,
                        N=2, closed_loop=False, simulation_length=5,
                        no_lam_p=False, hessian_approximation=True, jac_approx=False,
                        show_plot=True, plot_ol_traj=False,
                        problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True, track_gen_fn=test_quad_2d_track,
                        ignore_init_constr_check=False, add_scaling=False,
                        ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3,
                        x_threshold=7, z_threshold=7, x0_delim=2, x1_delim=4, x_min_threshold=-7, add_delta_constraints=False,
                        fwd_sim="nom_dyn", true_ds_inst=None, include_res_in_ctrl=False,
                        infeas_debug_callback=False, plot_shrunk_sep=False, waypoint_arr=((7, 0), (7, 7), (0, 7), (0, 0)), collision_check=False,
                        boole_deno_override=False, integration_method='euler', simulation_length_override=None, debug_ltv=False,
                        online_N=None, read_desired_from_file=None, use_prev_if_infeas=False, Q_override=None, R_override=None, Bd_override=None,
                                 num_multiple=5, data_save_file='gpmpc_d_runs'):
    if problem_setup_fn is quad_2d_sys_1d_inp_res:
        sys_config_dict, inst_override = problem_setup_fn(velocity_limit_override=velocity_override,
                                                          x_threshold=x_threshold, z_threshold=z_threshold, x_min_threshold=x_min_threshold,
                                                          sampling_time=sampling_time, x0_delim=x0_delim, x1_delim=x1_delim,
                                                          ret_inst=True)
        limiter_names = ["x_threshold", "z_threshold"]
        assert integration_method == 'euler', "Euler integration only available for this system."
        # print(limiter_names)
    elif problem_setup_fn is planar_lti_1d_inp_res:
        problem_setup_fn: planar_lti_1d_inp_res
        sys_config_dict, inst_override = problem_setup_fn(u_max=velocity_override, sampling_time=sampling_time,
                                                          x1_threshold=x_threshold, x2_threshold=z_threshold, x1_min_threshold=x_min_threshold,
                                                          x0_delim=x0_delim, x1_delim=x1_delim, ret_inst=True)
        limiter_names = ["x1_threshold", "x2_threshold"]
        # print(limiter_names)
    else:
        sys_config_dict, inst_override = problem_setup_fn(ret_inst=True)

    true_ds_inst.regions = sys_config_dict["regions"]

    lti = inst_override.lti
    if lti and integration_method == 'exact':
        if not inst_override.symbolic.exact_flag:
            print("Exact linearization not available for this system. Using Euler integration instead.")
            integration_method = 'euler'

    if Q_override is not None:
        sys_config_dict["Q"] = Q_override
    if R_override is not None:
        sys_config_dict["R"] = R_override
    if Bd_override is not None:
        sys_config_dict["Bd"] = Bd_override
    # print(sys_config_dict)
    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N if online_N is None else online_N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": sys_config_dict["Bd"],
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "satisfaction_prob": satisfaction_prob, "skip_shrinking": skip_shrinking,
                      "collision_check": collision_check, "boole_deno_override": boole_deno_override,
                      "integration_method": integration_method, "lti": lti, "use_prev_if_infeas": use_prev_if_infeas}
    np.random.seed(np.random.randint(0, 1000000))

    # Neglects residual mean dynamics in the controller. Variance info already excluded from the formulation for these increments.
    if not include_res_in_ctrl:
        sys_config_dict.update({"Bd": np.zeros((6, 1))})
    # sys_config_dict.update({"Bd": np.array([[0, 1, 0, 0, 0, 0]]).T})

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts(print_level, add_monitor=False, early_termination=early_termination, hsl_solver=False,
                                 test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation,
                                 jac_approx=jac_approx)
    fn_config_dict["addn_solver_opts"] = opts

    # Pull relevant info from config dict.
    n_u = sys_config_dict["n_u"]

    # Nominal MPC solution for warmstarting.
    if read_desired_from_file is not None:
        desired_traj_data = read_data(file_name=read_desired_from_file)
        x_desired = desired_traj_data["x_desired"]
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(x_desired[0, :], x_desired[2, :], color='cyan',
        #         label='Discretized trajectory to track', marker='o', linewidth=3, markersize=10)
        u_desired = desired_traj_data["u_desired"]
        assert desired_traj_data["N"] >= online_N, "Desired trajectory has horizon length less than online horizon length."
    else:
        # Nominal MPC solution for warmstarting.
        x_desired, u_desired = track_gen_fn(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                            viz=False, verbose=verbose, N=N, velocity_override=velocity_override,
                                            ret_inputs=True, sampling_time=sampling_time, waypoints_arr=waypoint_arr,
                                            inst_override=inst_override, integration=integration_method, x_init=x_init)

    u_desired = np.array(u_desired, ndmin=2).reshape((n_u, -1))
    tracking_matrix = sys_config_dict["tracking_matrix"]
    waypoints_to_track = tracking_matrix[:, :sys_config_dict["n_x"]] @ x_desired
    initial_info_dict = {"x_init": x_init, "u_warmstart": u_desired}

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = GPMPC_QP(**configs)

    data_store_dicts = []
    for run_idx in range(num_multiple):
        if debug_ltv:
            single_idx = False
            if single_idx:
                controller_inst.set_traj_to_track(x_desired, u_desired)
                controller_inst.generate_hld_for_warmstart()
                controller_inst.setup_resmean_warmstart_vec()
                controller_inst.shrink_by_desired(simulation_length)
                dt = controller_inst.dt
                idx = 24
                u = u_desired[:, idx]
                # u = np.zeros((2, 1)).squeeze()
                print(x_desired[:, idx], u)
                xplus_euler = x_desired[:, idx] + dt * (controller_inst.ct_dyn_nom(x=x_desired[:, idx], u=u)['f'])
                # xplus_euler = x_desired[:, idx] + dt * (controller_inst.ct_dyn_nom(x=x_desired[:, idx], u=u)['f'] + controller_inst.Bd @ controller_inst.resmean_warmstart[:, idx])
                xplus_rk4 = controller_inst.sys_dyn.rk4(x=x_desired[:, idx], u=u)['f']
                # xplus_rk4 = controller_inst.sys_dyn.rk4(x=x_desired[:, idx], u=u)['f'] + (dt * controller_inst.Bd @ controller_inst.resmean_warmstart[:, idx])
                A_k, B_k = controller_inst.linearize_and_discretize(x_desired[:, idx], u, cs_equivalent=True, method='zoh')
                xplus_zoh = A_k @ x_desired[:, idx] + B_k @ u
                # xplus_zoh = A_k @ x_desired[:, idx] + B_k @ u + (dt * controller_inst.Bd @ controller_inst.resmean_warmstart[:, idx])
                # partial_der_calc = controller_inst.sys_dyn.df_func(x_desired[:, idx], u)
                # print(partial_der_calc)
                # A_c, B_c = np.array(partial_der_calc[0]), np.array(partial_der_calc[1])
                # A_k_ex = scipy.linalg.expm(A_c * dt)
                # B_k_ex = np.linalg.inv(A_c) @ (A_k_ex - np.eye(A_c.shape[0])) @ B_c
                # xplus_exact = A_k_ex @ x_desired[:, 10] + B_k_ex @ u_desired[:, 10] + (dt * controller_inst.Bd @ controller_inst.resmean_warmstart[:, 10])
                # print(xplus_euler, xplus_zoh, xplus_exact)
                print("rk4, euler, zoh", xplus_rk4, xplus_euler, xplus_zoh)
                # partial_der_calc = controller_inst.sys_dyn.df_func(x_desired[:, idx], u)
                # print(partial_der_calc)
                # # print(quad_2D_dyn().symbolic.fd_linear_func(x_desired[:, idx], u))
                # # F = quad_2D_dyn().symbolic.fd_linear_func
                # dfdx = partial_der_calc['dfdx'].toarray()
                # dfdu = partial_der_calc['dfdu'].toarray()
                # next_state = quad_2D_dyn(velocity_limit_override=5).symbolic.linear_dynamics_func(x0=x_desired[:, idx], p=u)['xf']
                # print(next_state)
                partial_der_calc = controller_inst.sys_dyn.df_func(x_desired[:, idx], u)
                A_c, B_c = np.array(partial_der_calc[0]), np.array(partial_der_calc[1])
                print("ct_mats", A_c, B_c)
                theta_curr = x_desired[4, idx]
                u1_curr = u[0]
                u2_curr = u[1]
                m, g, l = 0.027, 9.8, 0.0397
                Iyy = 1.4e-5
                A_c_manual = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, cs.cos(theta_curr) * (u1_curr + u2_curr) / m, 0],
                                       [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, -cs.sin(theta_curr) * (u1_curr + u2_curr) / m, 0],
                                       [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]])
                B_c_manual = np.array([[0, 0], [cs.sin(theta_curr)/m, cs.sin(theta_curr)/m],
                                       [0, 0], [cs.cos(theta_curr)/m, cs.cos(theta_curr)/m],
                                       [0, 0], [-l/(1.414*Iyy), l/(1.414*Iyy)]])
                print("manual_cts", A_c_manual, B_c_manual)
                state_dim, input_dim, A, B = controller_inst.n_x, n_u, A_c, B_c
                M = np.zeros((state_dim + input_dim, state_dim + input_dim))
                M[:state_dim, :state_dim] = A
                M[:state_dim, state_dim:] = B

                Md = scipy.linalg.expm(M * dt)
                Ad = Md[:state_dim, :state_dim]
                # print(Ad, A_k)
                Bd = Md[:state_dim, state_dim:]
                # print(Bd, B_k)
                # print(Ad.shape, Bd.shape, A_k.shape, B_k.shape)
                # print((Ad @ x_desired[:, idx]).shape)
                # print((Bd @ u).shape)
                xplus_exact = Ad @ x_desired[:, idx] + Bd @ u
                # xplus_exact = Ad @ x_desired[:, idx] + Bd @ u + (dt * controller_inst.Bd @ controller_inst.resmean_warmstart[:, idx])
                print("exact", xplus_exact)
                print("cvodes_ct_sym", controller_inst.sys_dyn.fd_func)
                print("cvodes_ct_evald", controller_inst.sys_dyn.fd_func(x_desired[:, idx], u, 0, 0, 0, 0))
                # print(controller_inst.sys_dyn.f_disc_linear_func(x_desired[:, idx], u))
                disc_lind = controller_inst.sys_dyn.f_disc_linear_func(x_desired[:, idx], u)
                Ad, Bd = disc_lind[0], disc_lind[1]
                print("dt_mats_cvodes", Ad, Bd)
                print("dt_mats_manual", np.eye(controller_inst.n_x) + dt * A_c_manual, dt * B_c_manual)
                xplus_cvodes_lind = Ad @ x_desired[:, idx] + Bd @ u
                print("cvodes_linearized_evald", xplus_cvodes_lind)
            else:
                dt = controller_inst.dt
                cvodes_lind_errors = []
                euler_errors = []
                zoh_errors = []
                rk4_errors = []
                for idx in range(simulation_length):
                    u = u_desired[:, idx]
                    # u = np.zeros((2, 1)).squeeze()

                    # ct integrated using casadi built-in
                    cvodes_ct_evald = controller_inst.sys_dyn.fd_func(x_desired[:, idx], u, 0, 0, 0, 0)[0]

                    # discretized version of ct integrated method. Computes jacobian of the integration function that has already incorporated sampling time.
                    disc_lind = controller_inst.sys_dyn.f_disc_linear_func(x_desired[:, idx], u)
                    Ad, Bd = disc_lind[0], disc_lind[1]
                    xplus_cvodes_lind = Ad @ x_desired[:, idx] + Bd @ u

                    # Euler integration non-linear
                    xplus_euler = x_desired[:, idx] + dt * (controller_inst.ct_dyn_nom(x=x_desired[:, idx], u=u)['f'])

                    # RK4 integration non-linear
                    xplus_rk4 = controller_inst.sys_dyn.rk4(x=x_desired[:, idx], u=u)['f']

                    # Zoh system without cvodes
                    A_dk, B_dk = controller_inst.linearize_and_discretize(x_desired[:, idx], u, cs_equivalent=True, method='zoh')
                    xplus_zoh = A_dk @ x_desired[:, idx] + B_dk @ u

                    cvodes_lind_errors.append(np.array((cvodes_ct_evald - xplus_cvodes_lind).T))
                    euler_errors.append(np.array((cvodes_ct_evald - xplus_euler).T))
                    zoh_errors.append(np.array((cvodes_ct_evald - xplus_zoh).T))
                    rk4_errors.append(np.array((cvodes_ct_evald - xplus_rk4).T))
                cvodes_lind_errors = np.fabs(np.vstack(cvodes_lind_errors))
                euler_errors = np.fabs(np.vstack(euler_errors))
                zoh_errors = np.fabs(np.vstack(zoh_errors))
                rk4_errors = np.fabs(np.vstack(rk4_errors))
                print("CVODES AVG LIND ERRORS BY COLUMN")
                print(np.mean(cvodes_lind_errors, axis=0))
                print("CVODES MAX LIND ERRORS BY COLUMN")
                print(np.max(cvodes_lind_errors, axis=0))
                print("EULER AVG NONLIN ERRORS BY COLUMN")
                print(np.mean(euler_errors, axis=0))
                print("EULER MAX NONLIN ERRORS BY COLUMN")
                print(np.max(euler_errors, axis=0))
                print("ZOH AVG LIND ERRORS BY COLUMN")
                print(np.mean(zoh_errors, axis=0))
                print("ZOH MAX LIND ERRORS BY COLUMN")
                print(np.max(zoh_errors, axis=0))
                print("RK4 AVG NONLIN ERRORS BY COLUMN")
                print(np.mean(rk4_errors, axis=0))
                print("RK4 MAX NONLIN ERRORS BY COLUMN")
                print(np.max(rk4_errors, axis=0))
                #
                # print(cvodes_lind_errors)
                # print(euler_errors)
                # print(zoh_errors)


        else:
            if not closed_loop:
                debugger_inst = OptiDebugger(controller_inst)
                sol = controller_inst.run_ol_opt(initial_info_dict, x_desired=x_desired, u_desired=u_desired,
                                                 plot_shrunk_sets=plot_shrunk_sets, plot_shrunk_sep=plot_shrunk_sep)
                print("PRINTING SOL STATS")
                print(sol.stats())
                print("Total time for iteration %s" % sol.stats()['t_wall_total'])

                if show_plot:
                    fig, ax = plt.subplots(1, 1)
                    controller_utils.plot_OL_opt_soln(debugger_inst=debugger_inst,
                                                      state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                      ax=ax, waypoints_to_track=waypoints_to_track,
                                                      ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
            else:
                simulation_length = simulation_length_override if simulation_length_override is not None else simulation_length
                run_start_time = time.time()
                data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                        opt_verbose=verbose, x_desired=x_desired, u_desired=u_desired,
                                                                        infeas_debug_callback=infeas_debug_callback)

                total_run_time = 0
                for stat_dict in sol_stats_cl:
                    run_time = stat_dict['t_wall_total']
                    total_run_time += run_time
                average_run_time = total_run_time / len(sol_stats_cl)
                print("Average run time: %s" % average_run_time)

                mu_x_cl = [data_dict_ol['mu_x'] for data_dict_ol in data_dict_cl]
                # mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
                # print(mu_u_cl)
                # theta_cl = [data_dict_ol['mu_x'][:, 0][4] for data_dict_ol in data_dict_cl]
                # print(theta_cl)
                mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
                x_cl_for_cost = np.hstack([data_dict_ol['mu_x'][:, [0]] for data_dict_ol in data_dict_cl])
                x_dev = x_cl_for_cost - x_desired[:, :x_cl_for_cost.shape[1]]
                print(np.mean(x_dev, axis=1))
                cl_cost = 0
                for i in range(x_dev.shape[1]):
                    cl_cost += x_dev[:, [i]].T @ sys_config_dict["Q"] @ x_dev[:, [i]] + mu_u_cl[:, [i]].T @ sys_config_dict["R"] @ mu_u_cl[:, [i]]
                # print("Closed loop cost: %s" % cl_cost)
                total_run_time = 0
                run_times = []
                for stat_dict in sol_stats_cl:
                    run_time = stat_dict['t_wall_total']
                    run_times.append(run_time)
                    total_run_time += run_time
                average_run_time = total_run_time / len(sol_stats_cl)
                # print("Average run time: %s" % average_run_time)

                data_to_store = {"average_run_time": average_run_time, "run_times": run_times, "collision_count": controller_inst.collision_counter,
                                 "collision_idxs": controller_inst.collision_idxs, "cl_cost": cl_cost,
                                 "data_dict_cl": data_dict_cl, "sol_stats_cl": sol_stats_cl,
                                 "run_start_time": run_start_time}
                print('PRINTING RUN STATISTICS')
                print(data_to_store["average_run_time"])
                print(data_to_store["collision_count"])
                print(data_to_store["collision_idxs"])
                print(data_to_store["cl_cost"])
                data_store_dicts.append(data_to_store)

                save_data(data_store_dicts, file_name=data_save_file, update_data=True)
                if show_plot:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                    mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                                data_dict_cl=data_dict_cl,
                                                                ret_mu_x_cl=True,
                                                                state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                                plot_ol_traj=plot_ol_traj, axes=[ax],
                                                                ax_xlim=(-0.05, sys_config_dict[limiter_names[0]]+0.1),
                                                                ax_ylim=(-0.05, sys_config_dict[limiter_names[1]]+0.1),
                                                                legend_loc='upper center',
                                                                regions=controller_inst.regions, ignore_legend=True)
                    # save_fig(axes=[ax], fig_name='boundary_tracking_nlp', tick_sizes=14, tick_skip=1, k_range=None)

                    if collision_check:
                        print('NUMBER OF COLLISIONS: %s' % controller_inst.collision_counter)

                    # fig, axes = plt.subplots(2, 1)
                    # k_range = [i for i in range(mu_u_cl.shape[1])]
                    # axes[0].plot(k_range, mu_u_cl[0, :], label="u_0")
                    # axes[1].plot(k_range, mu_u_cl[1, :]-mu_u_cl[0, :], label="u_0")
                    # for i in range(2):
                    #     axes[i].tick_params(axis='both', labelsize=14)
                    #     axes[i].set_xticks(k_range[::2])


# def gpmpc_BnC_controller_multiple(gp_fns, x_init, velocity_override=0.7, early_termination=True, minimal_print=True,
#                                   N=2, closed_loop=False, simulation_length=5,
#                                   no_lam_p=False, hessian_approximation=False, jac_approx=False,
#                                   show_plot=True, plot_ol_traj=False,
#                                   problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True,
#                                   ignore_init_constr_check=False, add_scaling=False,
#                                   ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3, add_delta_constraints=True,
#                                   fwd_sim="nom_dyn", true_ds_inst=None, include_res_in_ctrl=False,
#                                   infeas_debug_callback=False, collision_check=False,
#                                   num_multiple=5, data_save_file='gpmpc_d_runs'):
#     sys_config_dict = problem_setup_fn(velocity_limit_override=velocity_override)
#     fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N, 'piecewise': True,
#                       'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
#                       "ignore_callback": ignore_callback, "sampling_time": sampling_time,
#                       "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
#                       "true_ds_inst": true_ds_inst, "Bd_fwd_sim": problem_setup_fn()["Bd"],
#                       "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
#                       "collision_check": collision_check}
#     # Neglects residual mean dynamics in the controller. Variance info already excluded from the formulation for these increments.
#     if not include_res_in_ctrl:
#         sys_config_dict.update({"Bd": np.zeros((6, 1))})
#     # sys_config_dict.update({"Bd": np.array([[0, 1, 0, 0, 0, 0]]).T})
#
#     configs = {}
#     configs.update(sys_config_dict)
#     configs.update(fn_config_dict)
#
#     run_start_time = time.time()
#
#     print_level = 0 if minimal_print else 5
#     opts = construct_config_opts(print_level, add_monitor=False, early_termination=early_termination, hsl_solver=False,
#                                  test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation,
#                                  jac_approx=jac_approx)
#     fn_config_dict["addn_solver_opts"] = opts
#
#     # Pull relevant info from config dict.
#     n_u = sys_config_dict["n_u"]
#
#     # Nominal MPC solution for warmstarting.
#     x_desired, u_warmstart = test_quad_2d_track(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
#                                                 viz=False, verbose=verbose, N=N, velocity_override=velocity_override,
#                                                 ret_inputs=True, sampling_time=sampling_time)
#
#     u_warmstart = np.array(u_warmstart, ndmin=2).reshape((n_u, -1))
#     tracking_matrix = sys_config_dict["tracking_matrix"]
#     waypoints_to_track = tracking_matrix[:, :sys_config_dict["n_x"]] @ x_desired
#     initial_info_dict = {"x_init": x_init, "u_warmstart": u_warmstart}
#
#     data_store_dicts = []
#     for run_idx in range(num_multiple):
#         # Instantiate controller and setup optimization problem to be solved.
#         controller_inst = GPMPC_BnC(**configs)
#
#         if not closed_loop:
#             debugger_inst = OptiDebugger(controller_inst)
#             sol = controller_inst.run_ol_opt(initial_info_dict, x_desired=x_desired)
#
#             if show_plot:
#                 fig, ax = plt.subplots(1, 1)
#                 controller_utils.plot_OL_opt_soln(debugger_inst=debugger_inst,
#                                                   state_plot_idxs=sys_config_dict["state_plot_idxs"],
#                                                   ax=ax, waypoints_to_track=waypoints_to_track,
#                                                   ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
#         else:
#             data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
#                                                                     opt_verbose=verbose, x_desired=x_desired,
#                                                                     infeas_debug_callback=infeas_debug_callback)
#
#             total_run_time = 0
#             run_times = []
#             for stat_dict in sol_stats_cl:
#                 run_time = stat_dict['t_wall_total']
#                 run_times.append(run_time)
#                 total_run_time += run_time
#             average_run_time = total_run_time / len(sol_stats_cl)
#             print("Average run time: %s" % average_run_time)
#
#             mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
#             x_cl_for_cost = np.hstack([data_dict_ol['mu_x'][:, [0]] for data_dict_ol in data_dict_cl])
#             x_dev = x_cl_for_cost - x_desired[:, :x_cl_for_cost.shape[1]]
#             cl_cost = 0
#             for i in range(x_dev.shape[1]):
#                 cl_cost += x_dev[:, [i]].T @ sys_config_dict["Q"] @ x_dev[:, [i]] + mu_u_cl[:, [i]].T @ sys_config_dict["R"] @ mu_u_cl[:, [i]]
#             data_to_store = {"average_run_time": average_run_time, "run_times": run_times, "collision_count": controller_inst.collision_counter,
#                              "collision_idxs": controller_inst.collision_idxs, "cl_cost": cl_cost,
#                              "data_dict_cl": data_dict_cl, "sol_stats_cl": sol_stats_cl,
#                              "run_start_time": run_start_time}
#             print('PRINTING RUN STATISTICS')
#             print(data_to_store["average_run_time"])
#             print(data_to_store["collision_count"])
#             print(data_to_store["collision_idxs"])
#             print(data_to_store["cl_cost"])
#             data_store_dicts.append(data_to_store)
#
#             if show_plot:
#                 mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
#                                                             data_dict_cl=data_dict_cl,
#                                                             ret_mu_x_cl=True,
#                                                             state_plot_idxs=sys_config_dict["state_plot_idxs"],
#                                                             plot_ol_traj=plot_ol_traj, axes=None,
#                                                             ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
#                 if collision_check:
#                     print('NUMBER OF COLLISIONS: %s' % controller_inst.collision_counter)
#
#     save_data(data_store_dicts, file_name=data_save_file)


def gpmpc_D_controller_multiple(gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7, early_termination=True,
                       minimal_print=True, plot_shrunk_sets=False, skip_shrinking=False,
                       N=2, closed_loop=False, simulation_length=5,
                       no_lam_p=False, hessian_approximation=True, jac_approx=False,
                       show_plot=True, plot_ol_traj=False,
                       problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True, track_gen_fn=test_quad_2d_track,
                       ignore_init_constr_check=False, add_scaling=False,
                       ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3,
                       x_threshold=7, z_threshold=7, x0_delim=2, x1_delim=4, x_min_threshold=-7,
                       add_delta_constraints=False, fwd_sim="w_pw_res", true_ds_inst=None, include_res_in_ctrl=False,
                       infeas_debug_callback=False, plot_shrunk_sep=False, waypoint_arr=((7, 0), (7, 7), (0, 7), (0, 0)), collision_check=False,
                       simulation_length_override=None, boole_deno_override=False, integration_method="euler",
                       online_N=None, read_desired_from_file=None, use_prev_if_infeas=False,
                       Q_override=None, R_override=None, Bd_override=None, num_multiple=5, data_save_file='gpmpc_d_runs'):
    if problem_setup_fn is quad_2d_sys_1d_inp_res:
        sys_config_dict, inst_override = problem_setup_fn(velocity_limit_override=velocity_override,
                                                          x_threshold=x_threshold, z_threshold=z_threshold, x_min_threshold=x_min_threshold,
                                                          sampling_time=sampling_time, x0_delim=x0_delim, x1_delim=x1_delim,
                                                          ret_inst=True)
        limiter_names = ["x_threshold", "z_threshold"]
        assert integration_method == 'euler', "Euler integration only available for this system."
    elif problem_setup_fn is planar_lti_1d_inp_res:
        problem_setup_fn: planar_lti_1d_inp_res
        sys_config_dict, inst_override = problem_setup_fn(u_max=velocity_override, sampling_time=sampling_time,
                                                          x1_threshold=x_threshold, x2_threshold=z_threshold,
                                                          x0_delim=x0_delim, x1_delim=x1_delim, ret_inst=True)
        limiter_names = ["x1_threshold", "x2_threshold"]
    else:
        sys_config_dict, inst_override = problem_setup_fn(ret_inst=True)

    true_ds_inst.regions = sys_config_dict["regions"]

    if Q_override is not None:
        sys_config_dict["Q"] = Q_override
    if R_override is not None:
        sys_config_dict["R"] = R_override
    if Bd_override is not None:
        sys_config_dict["Bd"] = Bd_override
    # print(sys_config_dict)
    lti = inst_override.lti
    if lti and integration_method == 'exact':
        if not inst_override.symbolic.exact_flag:
            print("Exact linearization not available for this system. Using Euler integration instead.")
            integration_method = 'euler'
    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N if online_N is None else online_N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": sys_config_dict["Bd"],
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "satisfaction_prob": satisfaction_prob, "skip_shrinking": skip_shrinking,
                      "collision_check": collision_check, "boole_deno_override": boole_deno_override,
                      "lti": lti, "integration_method": integration_method, "use_prev_if_infeas": use_prev_if_infeas}
    np.random.seed(np.random.randint(0, 1000000))

    # Neglects residual mean dynamics in the controller. Variance info already excluded from the formulation for these increments.
    if not include_res_in_ctrl:
        sys_config_dict.update({"Bd": np.zeros((6, 1))})
    # sys_config_dict.update({"Bd": np.array([[0, 1, 0, 0, 0, 0]]).T})

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts(print_level, add_monitor=False, early_termination=early_termination, hsl_solver=False,
                                 test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation,
                                 jac_approx=jac_approx)
    fn_config_dict["addn_solver_opts"] = opts

    # Pull relevant info from config dict.
    n_u = sys_config_dict["n_u"]

    # Nominal MPC solution for warmstarting.
    if read_desired_from_file is not None:
        desired_traj_data = read_data(file_name=read_desired_from_file)
        x_desired = desired_traj_data["x_desired"]
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(x_desired[0, :], x_desired[2, :], color='cyan',
        #         label='Discretized trajectory to track', marker='o', linewidth=3, markersize=10)
        u_desired = desired_traj_data["u_desired"]
        assert desired_traj_data["N"] >= online_N, "Desired trajectory has horizon length less than online horizon length."
    else:
        # Nominal MPC solution for warmstarting.
        x_desired, u_desired = track_gen_fn(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                            viz=False, verbose=verbose, N=N, velocity_override=velocity_override,
                                            ret_inputs=True, sampling_time=sampling_time, waypoints_arr=waypoint_arr,
                                            inst_override=inst_override, integration=integration_method, x_init=x_init)

    u_desired = np.array(u_desired, ndmin=2).reshape((n_u, -1))
    tracking_matrix = sys_config_dict["tracking_matrix"]
    waypoints_to_track = tracking_matrix[:, :sys_config_dict["n_x"]] @ x_desired
    initial_info_dict = {"x_init": x_init, "u_warmstart": u_desired}

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = GPMPC_D(**configs)

    data_store_dicts = []
    for run_idx in range(num_multiple):
        if not closed_loop:
            debugger_inst = OptiDebugger(controller_inst)
            sol = controller_inst.run_ol_opt(initial_info_dict, x_desired=x_desired, u_desired=u_desired,
                                             plot_shrunk_sets=plot_shrunk_sets, plot_shrunk_sep=plot_shrunk_sep)
            print("PRINTING SOL STATS")
            print(sol.stats())
            print("Total time for iteration %s" % sol.stats()['t_wall_total'])

            if show_plot:
                fig, ax = plt.subplots(1, 1)
                controller_utils.plot_OL_opt_soln(debugger_inst=debugger_inst,
                                                  state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                  ax=ax, waypoints_to_track=waypoints_to_track,
                                                  ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
        else:
            simulation_length = simulation_length_override if simulation_length_override is not None else simulation_length
            run_start_time = time.time()
            data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                    opt_verbose=verbose, x_desired=x_desired, u_desired=u_desired,
                                                                    infeas_debug_callback=infeas_debug_callback)

            total_run_time = 0
            for stat_dict in sol_stats_cl:
                run_time = stat_dict['t_wall_total']
                total_run_time += run_time
            average_run_time = total_run_time / len(sol_stats_cl)
            print("Average run time: %s" % average_run_time)

            mu_x_cl = [data_dict_ol['mu_x'] for data_dict_ol in data_dict_cl]
            # mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
            # print(mu_u_cl)
            # theta_cl = [data_dict_ol['mu_x'][:, 0][4] for data_dict_ol in data_dict_cl]
            # print(theta_cl)
            mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
            x_cl_for_cost = np.hstack([data_dict_ol['mu_x'][:, [0]] for data_dict_ol in data_dict_cl])
            x_dev = x_cl_for_cost - x_desired[:, :x_cl_for_cost.shape[1]]
            print(np.mean(x_dev, axis=1))
            cl_cost = 0
            for i in range(x_dev.shape[1]):
                cl_cost += x_dev[:, [i]].T @ sys_config_dict["Q"] @ x_dev[:, [i]] + mu_u_cl[:, [i]].T @ sys_config_dict["R"] @ mu_u_cl[:, [i]]
            # print("Closed loop cost: %s" % cl_cost)
            total_run_time = 0
            run_times = []
            for stat_dict in sol_stats_cl:
                run_time = stat_dict['t_wall_total']
                run_times.append(run_time)
                total_run_time += run_time
            average_run_time = total_run_time / len(sol_stats_cl)
            # print("Average run time: %s" % average_run_time)

            data_to_store = {"average_run_time": average_run_time, "run_times": run_times, "collision_count": controller_inst.collision_counter,
                             "collision_idxs": controller_inst.collision_idxs, "cl_cost": cl_cost,
                             "data_dict_cl": data_dict_cl, "sol_stats_cl": sol_stats_cl,
                             "run_start_time": run_start_time}
            print('PRINTING RUN STATISTICS')
            print(data_to_store["average_run_time"])
            print(data_to_store["collision_count"])
            print(data_to_store["collision_idxs"])
            print(data_to_store["cl_cost"])
            data_store_dicts.append(data_to_store)

            save_data(data_store_dicts, file_name=data_save_file, update_data=True)

            if show_plot:

                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                            data_dict_cl=data_dict_cl,
                                                            ret_mu_x_cl=True,
                                                            state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                            plot_ol_traj=plot_ol_traj, axes=[ax],
                                                            ax_xlim=(-0.05, sys_config_dict[limiter_names[0]]+0.1),
                                                            ax_ylim=(-0.05, sys_config_dict[limiter_names[1]]+0.1),
                                                            # ax_xlim=(-0.05, sys_config_dict["x_threshold"]), ax_ylim=(-0.05, sys_config_dict["z_threshold"]),
                                                            legend_loc='upper center',
                                                            regions=controller_inst.regions, ignore_legend=True)
                # save_fig(axes=[ax], fig_name='boundary_tracking_nlp', tick_sizes=14, tick_skip=1, k_range=None)

                if collision_check:
                    print('NUMBER OF COLLISIONS: %s' % controller_inst.collision_counter)

                # fig, axes = plt.subplots(2, 1)
                # k_range = [i for i in range(mu_u_cl.shape[1])]
                # axes[0].plot(k_range, mu_u_cl[0, :], label="u_0")
                # axes[1].plot(k_range, mu_u_cl[1, :]-mu_u_cl[0, :], label="u_0")
                # for i in range(2):
                #     axes[i].tick_params(axis='both', labelsize=14)
                #     axes[i].set_xticks(k_range[::2])


# def gpmpc_D_controller_multiple(gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7, early_termination=True,
#                                 minimal_print=True, plot_shrunk_sets=False, skip_shrinking=False,
#                                 N=2, closed_loop=False, simulation_length=5,
#                                 no_lam_p=False, hessian_approximation=True, jac_approx=False,
#                                 show_plot=True, plot_ol_traj=False,
#                                 problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True,
#                                 ignore_init_constr_check=False, add_scaling=False,
#                                 ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3, add_delta_constraints=True,
#                                 fwd_sim="nom_dyn", true_ds_inst=None, include_res_in_ctrl=True,
#                                 infeas_debug_callback=False, plot_shrunk_sep=False, waypoint_arr=None, collision_check=False,
#                                 boole_deno_override=False, num_multiple=5, data_save_file='gpmpc_d_runs'):
#     sys_config_dict = problem_setup_fn(velocity_limit_override=velocity_override)
#     fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N, 'piecewise': True,
#                       'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
#                       "ignore_callback": ignore_callback, "sampling_time": sampling_time,
#                       "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
#                       "true_ds_inst": true_ds_inst, "Bd_fwd_sim": problem_setup_fn()["Bd"],
#                       "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
#                       "satisfaction_prob": satisfaction_prob, "skip_shrinking": skip_shrinking,
#                       "collision_check": collision_check, "boole_deno_override": boole_deno_override}
#     np.random.seed(np.random.randint(0, 1000000))
#
#     run_start_time = time.time()
#
#     # Neglects residual mean dynamics in the controller. Variance info already excluded from the formulation for these increments.
#     if not include_res_in_ctrl:
#         sys_config_dict.update({"Bd": np.zeros((6, 1))})
#     # sys_config_dict.update({"Bd": np.array([[0, 1, 0, 0, 0, 0]]).T})
#
#     configs = {}
#     configs.update(sys_config_dict)
#     configs.update(fn_config_dict)
#
#     print_level = 0 if minimal_print else 5
#     opts = construct_config_opts(print_level, add_monitor=False, early_termination=early_termination, hsl_solver=False,
#                                  test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation,
#                                  jac_approx=jac_approx)
#     fn_config_dict["addn_solver_opts"] = opts
#
#     # Pull relevant info from config dict.
#     n_u = sys_config_dict["n_u"]
#
#     # Nominal MPC solution for warmstarting.
#     x_desired, u_desired = test_quad_2d_track(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
#                                               viz=False, verbose=verbose, N=N, velocity_override=velocity_override,
#                                               ret_inputs=True, sampling_time=sampling_time, waypoints_arr=waypoint_arr,
#                                               x_init=x_init)
#
#     u_desired = np.array(u_desired, ndmin=2).reshape((n_u, -1))
#     tracking_matrix = sys_config_dict["tracking_matrix"]
#     waypoints_to_track = tracking_matrix[:, :sys_config_dict["n_x"]] @ x_desired
#     initial_info_dict = {"x_init": x_init, "u_warmstart": u_desired}
#
#     # Instantiate controller and setup optimization problem to be solved.
#     data_store_dicts = []
#     for run_idx in range(num_multiple):
#         controller_inst = GPMPC_D(**configs)
#
#         if not closed_loop:
#             debugger_inst = OptiDebugger(controller_inst)
#             sol = controller_inst.run_ol_opt(initial_info_dict, x_desired=x_desired, u_desired=u_desired,
#                                              plot_shrunk_sets=plot_shrunk_sets, plot_shrunk_sep=plot_shrunk_sep)
#             print("PRINTING SOL STATS")
#             print(sol.stats())
#             print("Total time for iteration %s" % sol.stats()['t_wall_total'])
#
#             if show_plot:
#                 fig, ax = plt.subplots(1, 1)
#                 controller_utils.plot_OL_opt_soln(debugger_inst=debugger_inst,
#                                                   state_plot_idxs=sys_config_dict["state_plot_idxs"],
#                                                   ax=ax, waypoints_to_track=waypoints_to_track,
#                                                   ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
#         else:
#             data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
#                                                                     opt_verbose=verbose, x_desired=x_desired, u_desired=u_desired,
#                                                                     infeas_debug_callback=infeas_debug_callback)
#
#             total_run_time = 0
#             run_times = []
#             for stat_dict in sol_stats_cl:
#                 run_time = stat_dict['t_wall_total']
#                 run_times.append(run_time)
#                 total_run_time += run_time
#             average_run_time = total_run_time / len(sol_stats_cl)
#             print("Average run time: %s" % average_run_time)
#
#             mu_u_cl = np.hstack([data_dict_ol['mu_u'][:, [0]] for data_dict_ol in data_dict_cl])
#             x_cl_for_cost = np.hstack([data_dict_ol['mu_x'][:, [0]] for data_dict_ol in data_dict_cl])
#             x_dev = x_cl_for_cost - x_desired[:, :x_cl_for_cost.shape[1]]
#             cl_cost = 0
#             for i in range(x_dev.shape[1]):
#                 cl_cost += x_dev[:, [i]].T @ sys_config_dict["Q"] @ x_dev[:, [i]] + mu_u_cl[:, [i]].T @ sys_config_dict["R"] @ mu_u_cl[:, [i]]
#             data_to_store = {"average_run_time": average_run_time, "run_times": run_times, "collision_count": controller_inst.collision_counter,
#                              "collision_idxs": controller_inst.collision_idxs, "cl_cost": cl_cost,
#                              "data_dict_cl": data_dict_cl, "sol_stats_cl": sol_stats_cl,
#                              "run_start_time": run_start_time}
#             print('PRINTING RUN STATISTICS')
#             print(data_to_store["average_run_time"])
#             print(data_to_store["collision_count"])
#             print(data_to_store["collision_idxs"])
#             print(data_to_store["cl_cost"])
#             data_store_dicts.append(data_to_store)
#
#             # print(mu_u_cl)
#             # theta_cl = [data_dict_ol['mu_x'][:, 0][4] for data_dict_ol in data_dict_cl]
#             # print(theta_cl)
#
#             if show_plot:
#                 fig, ax = plt.subplots(1, 1, figsize=(12, 8))
#                 mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
#                                                             data_dict_cl=data_dict_cl,
#                                                             ret_mu_x_cl=True,
#                                                             state_plot_idxs=sys_config_dict["state_plot_idxs"],
#                                                             plot_ol_traj=plot_ol_traj, axes=[ax],
#                                                             ax_xlim=(-0.05, 7), ax_ylim=(-0.05, 7), legend_loc='upper center',
#                                                             regions=controller_inst.regions, ignore_legend=True)
#                 save_fig(axes=[ax], fig_name='boundary_tracking_nlp', tick_sizes=14, tick_skip=2, k_range=list(range(7)))
#
#                 if collision_check:
#                     print('NUMBER OF COLLISIONS: %s' % controller_inst.collision_counter)
#
#                 # fig, axes = plt.subplots(2, 1)
#                 # k_range = [i for i in range(mu_u_cl.shape[1])]
#                 # axes[0].plot(k_range, mu_u_cl[0, :], label="u_0")
#                 # axes[1].plot(k_range, mu_u_cl[1, :]-mu_u_cl[0, :], label="u_0")
#                 # for i in range(2):
#                 #     axes[i].tick_params(axis='both', labelsize=14)
#                 #     axes[i].set_xticks(k_range[::2])
#
#     save_data(data_store_dicts, file_name=data_save_file)



def gpmpc_That_controller(initial_waypoints, gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7, early_termination=True,
                          minimal_print=True, plot_shrunk_sets=False, skip_shrinking=False,
                          N=2, closed_loop=False, simulation_length=5, num_mapping_updates=2,
                          no_lam_p=False, hessian_approximation=False, jac_approx=False,
                          show_plot=True, plot_ol_traj=False, ws_mapping_fineness_param=(75,),
                          problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True,
                          ignore_init_constr_check=False, add_scaling=False,
                          ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3, add_delta_constraints=True,
                          fwd_sim="nom_dyn", true_ds_inst=None, include_res_in_ctrl=False,
                          infeas_debug_callback=False, plot_shrunk_sep=False, waypoint_arr=None,
                          num_runs_per_train=2, collision_check=False):
    sys_config_dict = problem_setup_fn(velocity_limit_override=velocity_override)
    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": problem_setup_fn()["Bd"],
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "satisfaction_prob": satisfaction_prob, "skip_shrinking": skip_shrinking,
                      "collision_check": collision_check}
    np.random.seed(np.random.randint(0, 1000000))

    # Neglects residual mean dynamics in the controller. Variance info already excluded from the formulation for these increments.
    if not include_res_in_ctrl:
        sys_config_dict.update({"Bd": np.zeros((6, 1))})
    # sys_config_dict.update({"Bd": np.array([[0, 1, 0, 0, 0, 0]]).T})

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts(print_level, add_monitor=False, early_termination=early_termination, hsl_solver=False,
                                 test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation,
                                 jac_approx=jac_approx)
    fn_config_dict["addn_solver_opts"] = opts

    # Pull relevant info from config dict.
    n_u = sys_config_dict["n_u"]

    # Nominal MPC solution for warmstarting.
    x_desired, u_desired = test_quad_2d_track(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                              viz=False, verbose=verbose, N=N, velocity_override=velocity_override,
                                              ret_inputs=True, sampling_time=sampling_time, waypoints_arr=waypoint_arr,
                                              x_init=x_init)

    u_desired = np.array(u_desired, ndmin=2).reshape((n_u, -1))
    tracking_matrix = sys_config_dict["tracking_matrix"]
    waypoints_to_track = tracking_matrix[:, :sys_config_dict["n_x"]] @ x_desired
    initial_info_dict = {"x_init": x_init, "u_warmstart": u_desired}

    mapping_model: SoftLabelNet
    mapping_ds_inst: Mapping_DS
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    mapping_model, mapping_ds_inst = ws_mappers.trajds_setup_and_train(ds_inst_in=true_ds_inst, pw_gp_wrapped=gp_fns, fineness_param=ws_mapping_fineness_param,
                                                                       waypoints_arr=initial_waypoints, no_train=False, num_discrete=num_discrete, simulation_length=simulation_length,
                                                                       batch_size=simulation_length, regions=sys_config_dict['regions'],
                                                                       ax=ax, itn_num=0)


    fn_config_dict["That_predictor"] = mapping_model
    fn_config_dict["initial_mapping_ds"] = mapping_ds_inst

    # Instantiate controller and setup optimization problem to be solved.
    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)
    controller_inst = GPMPC_That(**configs)

    for k in range(num_mapping_updates):
        data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                opt_verbose=verbose, x_desired=x_desired, u_desired=u_desired,
                                                                infeas_debug_callback=infeas_debug_callback)


        print("CLOSED LOOP COST FOR ITERATION {}:".format(k))
        mu_x_cl = [data_dict_ol['mu_x'][:, 0] for data_dict_ol in data_dict_cl]
        mu_u_cl = [data_dict_ol['mu_u'][:, 0] for data_dict_ol in data_dict_cl]
        # print("mu_u_cl length")
        # print(len(mu_u_cl))
        print(controller_utils.calc_cl_cost(mu_x_cl, mu_u_cl, x_desired, np.zeros((n_u, simulation_length)),
                                            controller_inst.Q, controller_inst.R))
        if show_plot:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                        data_dict_cl=data_dict_cl,
                                                        ret_mu_x_cl=True,
                                                        state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                        plot_ol_traj=plot_ol_traj, axes=[ax],
                                                        ax_xlim=(-1, 7.1), ax_ylim=(-0.5, 7.1), legend_loc='upper center',
                                                        regions=sys_config_dict['regions'], itn_num=k+1)


        mapping_model = ws_mappers.update_mapping_ds_and_train(mapping_ds_inst, ws_mapping_fineness_param,
                                                               no_train=False, test_incr_only=False, num_runs=num_runs_per_train,
                                                               itn_num=k+1)

        controller_inst.That_predictor = mapping_model


def gpmpc_MINLP_controller(gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7, early_termination=True,
                           minimal_print=True, plot_shrunk_sets=False, skip_shrinking=False,
                           N=2, closed_loop=False, simulation_length=5,
                           no_lam_p=False, hessian_approximation=False,
                           show_plot=True, plot_ol_traj=False,
                           problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True,
                           ignore_init_constr_check=False, add_scaling=False,
                           ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3, add_delta_constraints=True,
                           fwd_sim="nom_dyn", true_ds_inst=None, include_res_in_ctrl=False,
                           infeas_debug_callback=False, plot_shrunk_sep=False, waypoint_arr=None, add_monitor=False,
                           enable_softplus=True, shrink_by_param=True):
    sys_config_dict = problem_setup_fn(velocity_limit_override=velocity_override)
    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": problem_setup_fn()["Bd"],
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "satisfaction_prob": satisfaction_prob, "skip_shrinking": skip_shrinking,
                      "solver": "bonmin", "enable_softplus": enable_softplus, "shrink_by_param": shrink_by_param}
    np.random.seed(np.random.randint(0, 1000000))

    # Neglects residual mean dynamics in the controller. Variance info already excluded from the formulation for these increments.
    if not include_res_in_ctrl:
        sys_config_dict.update({"Bd": np.zeros((6, 1))})
    # sys_config_dict.update({"Bd": np.array([[0, 1, 0, 0, 0, 0]]).T})

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts_minlp(print_level=print_level, add_monitor=add_monitor, early_termination=early_termination, hsl_solver=False,
                                       test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation)
    fn_config_dict["addn_solver_opts"] = opts

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = GPMPC_MINLP(**configs)

    # Pull relevant info from config dict.
    n_u = sys_config_dict["n_u"]

    # Nominal MPC solution for warmstarting.
    x_desired, u_desired = test_quad_2d_track(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                              viz=False, verbose=False, N=N, velocity_override=velocity_override,
                                              ret_inputs=True, sampling_time=sampling_time, waypoints_arr=waypoint_arr,
                                              x_init=x_init)

    u_desired = np.array(u_desired, ndmin=2).reshape((n_u, -1))
    tracking_matrix = sys_config_dict["tracking_matrix"]
    waypoints_to_track = tracking_matrix[:, :controller_inst.n_x] @ x_desired
    initial_info_dict = {"x_init": x_init, "u_warmstart": u_desired}

    # print(controller_inst.opti.)

    if not closed_loop:
        debugger_inst = OptiDebugger(controller_inst)
        sol = controller_inst.run_ol_opt(initial_info_dict, x_desired=x_desired, u_desired=u_desired,
                                         plot_shrunk_sets=plot_shrunk_sets, plot_shrunk_sep=plot_shrunk_sep)

        if show_plot:
            fig, ax = plt.subplots(1, 1)
            controller_utils.plot_OL_opt_soln(debugger_inst=debugger_inst,
                                              state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                              ax=ax, waypoints_to_track=waypoints_to_track,
                                              ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
    else:
        data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                opt_verbose=verbose, x_desired=x_desired, u_desired=u_desired,
                                                                infeas_debug_callback=infeas_debug_callback)

        mu_x_cl = [data_dict_ol['mu_x'] for data_dict_ol in data_dict_cl]
        if show_plot:
            mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                        data_dict_cl=data_dict_cl,
                                                        ret_mu_x_cl=True,
                                                        state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                        plot_ol_traj=plot_ol_traj, axes=None,
                                                        ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')
