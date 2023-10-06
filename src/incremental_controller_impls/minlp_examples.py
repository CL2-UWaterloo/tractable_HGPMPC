import matplotlib.pyplot as plt
import time

from common.data_save_utils import save_data, read_data
from smpc.controller_utils import OptiDebugger, construct_config_opts, construct_config_opts_minlp
import smpc.controller_utils as controller_utils
from .controller_classes import *


def MINLP_order_check(sys_config_dict, opti_dict, opti_inst, controller_inst, N, shrunk=False):
    n_x, n_u, n_d = sys_config_dict["n_x"], sys_config_dict["n_u"], sys_config_dict["n_d"]
    mu_x_size = n_x * (N + 1)
    mu_u_size = n_u * N
    mu_d_size = n_d * N
    b_shrunk_x_size = (2*n_x) * N if shrunk else 0  # 2*n_x because box constraint and N instead of N+1 since 1st state
                                                    # known with certainty and hence original vector not opt shrunk vec
    hld_size = (N+1) * controller_inst.num_regions
    print(opti_dict["mu_x"])
    print(opti_inst.x[0])
    print(opti_inst.x[mu_x_size - 1])
    print(opti_dict["mu_u"])
    print(opti_inst.x[mu_x_size])
    print(opti_inst.x[mu_x_size + mu_u_size - 1])
    if shrunk:
        print(opti_dict["b_shrunk_x"])
        print(opti_inst.x[mu_x_size + mu_u_size])
        print(opti_inst.x[mu_x_size + mu_u_size + b_shrunk_x_size - 1])
    print(opti_dict["mu_d"])
    print(opti_inst.x[mu_x_size + mu_u_size + b_shrunk_x_size])
    print(opti_inst.x[mu_x_size + mu_d_size + b_shrunk_x_size + mu_u_size - 1])
    print(opti_dict["hld"])
    print(opti_inst.x[mu_x_size + mu_u_size + b_shrunk_x_size + mu_d_size])
    print(opti_inst.x[mu_x_size + mu_d_size + b_shrunk_x_size + mu_u_size + hld_size - 1])
    try:
        print(opti_inst.x[mu_x_size + mu_d_size + b_shrunk_x_size + mu_u_size + hld_size])
    except RuntimeError:
        print("All opt vars finished correctly.")

    if False in controller_inst.discrete_bool_vec[-1:-hld_size]:
        print(controller_inst.discrete_bool_vec)
        print("Discrete variables not all True")


def MINLP_var_check(opti_inst, opti_dict):
    for var_idx in range(opti_inst.nx):
        print(opti_inst.debug.x_describe(var_idx))
    opti_param_names = ["x_init", "x_desired", "u_desired", "P"]
    opti_params_retrieved = [opti_dict[opti_param_names[param_idx]] for param_idx in range(len(opti_param_names))]
    print(opti_params_retrieved)
    for param_idx in range(opti_inst.np):
        print(opti_inst.debug.p[param_idx])


def MINLP_constr_check(opti_inst, n_x, n_u, n_d, N, num_regions, controller_inst, ignore_init_constr_check, shrunk=False):
    # for constr_idx in range(opti_inst.ng):
    #     print(opti_inst.debug.g_describe(constr_idx))
    init_constr = n_x
    nom_dyn_constr = n_x * (N+1 - 1)
    res_mean_constr = n_d * N
    # nominal dynamics + residual mean dynamics are interspersed in the constraint list.
    net_dynamics_constr = nom_dyn_constr + res_mean_constr
    delta_domain_constr = (2*num_regions) * N  # Last timestep has no delta. 2x since 1 constraint for upper bound and 1 for lower bound.
    # 2*delta input size since box constraint
    region_interior_constr = (num_regions * 2*np.linalg.matrix_rank(controller_inst.delta_input_mask)) * N
    delta_timestep_constr = N
    # region_interior_constr and delta_timestep_constr are interspersed in the constraint list.
    net_region_constr = region_interior_constr + delta_timestep_constr
    redundant_init_constr = n_x
    state_constr = (2*n_x) * (N+1 - 1*(1 if ignore_init_constr_check else 0))  # 2*n_x since box constraint
    shrunk_comp_constr = (2*n_x) * N if shrunk else 0
    input_constr = (2*n_u) * N
    # state_constr and input_constr (and shrunk vec computation if applicable) are interspersed in the constraint list.
    net_state_input_constr = state_constr + input_constr + shrunk_comp_constr
    total_constr = init_constr + net_dynamics_constr + delta_domain_constr +\
                   net_region_constr + redundant_init_constr + net_state_input_constr
    assert total_constr == opti_inst.ng, "Total number of constraints is not as expected. Expected: {}, Actual: {}".format(total_constr, opti_inst.ng)
    incr_constr_array = np.hstack([0, np.cumsum([init_constr, net_dynamics_constr, delta_domain_constr, net_region_constr, redundant_init_constr, net_state_input_constr])])
    print(incr_constr_array)
    inequality_groups = [2, 3, 5, 6]  # Note: Group 3 i.e. net region constr has both inequalities (region_interior_constr) and equalities (delta_timestep_constr) interspersed
    equality_groups = [0, 1, 3, 4]
    if shrunk:
        equality_groups.append(6)  # If shrunk is True, then the shrunk computation constraint is interspersed in state+input constraint set leading to it being included in the equality list.
    # for constr_set in range(len(incr_constr_array)-1):
    #     for constr_idx in range(incr_constr_array[constr_set], incr_constr_array[constr_set+1]):
    #         print(opti_inst.debug.g_describe(constr_idx))


def setup_minlp_sanity_test(N, sampling_time, opti_inst, controller_inst, n_u, n_x, n_d):
    x_init = np.array([3, 0, 3.5, 0, 0, 0])
    x_array = np.zeros((6, N + 1))
    x_array[:, 0] = x_init
    u_array = np.ones((2, N)) * np.array([[0.15, 0.15]]).T
    for i in range(N):
        x_array[:, [i+1]] = x_array[:, [i]] + sampling_time * controller_inst.ct_dyn_nom(x_array[:, [i]], u_array[:, [i]])
    # print(x_array)
    num_regions = controller_inst.num_regions
    hld_mat = np.zeros((num_regions, (N + 1)))
    # Region delta assignment step
    for k in range(N+1):
        # 0 pad the inputs since we have already checked the deltas don't depend on them using the assertion above.
        joint_vec = np.vstack([x_array[:, [k]], np.zeros([n_u, 1])]).squeeze()
        for region_idx, region in enumerate(controller_inst.regions):
            hld_mat[region_idx, k] = 1 if region.check_satisfaction(
                (controller_inst.delta_input_mask @ joint_vec).T).item() is True else 0
            try:
                assert hld_mat[region_idx, k] == (1 if (
                            controller_inst.region_Hs[region_idx] @ np.array(controller_inst.delta_input_mask @ joint_vec,
                                                                  ndmin=2).T <= controller_inst.region_bs[
                                region_idx]).all() else 0)
            except AssertionError:
                print("Region {} not satisfied by delta input mask.".format(region_idx))
    # print(hld_mat)

    # Note: All arrays need to be flattened in a certain order for them to match the order in which they have been added to the opti inst.
    # mu_d constraints are going to be neglected for this part so I'll just set them to zero.
    x = cs.vertcat(cs.vec(x_array), cs.vec(u_array), cs.vec(np.zeros((n_d, N))), cs.vec(hld_mat))
    # print(x)

    # Param vec creation. Placeholders for the most part except for x_init.
    mu_x_size, mu_u_size, mu_d_size = n_x * (N + 1), n_u * N, n_d * (N + 1)
    x_des_shape, u_des_shape = np.zeros(mu_x_size), np.zeros(mu_u_size)
    P_mat_shape = np.zeros((n_x * n_x))
    p = np.concatenate((x_init.flatten(order='F'), x_des_shape, u_des_shape, P_mat_shape))

    assert x.shape[0] == opti_inst.nx,\
        "x vector for debugging is not the correct size. x.shape[0]: %s, opti_inst.nx: %s" % (x.shape[0], opti_inst.nx)
    assert p.shape[0] == opti_inst.np,\
        "p vector for debugging is not the correct size. p.shape[0]: %s, opti_inst.np: %s" % (p.shape[0], opti_inst.np)

    return x, p


def setup_minlp_res_test(N, sampling_time, opti_inst, controller_inst: GPMPC_MINLP, n_u, n_x, n_d, shrunk=False,
                         x_init_override=None):
    if x_init_override is None:
        x_init = np.array([3, 0, 3.5, 0, 0, 0])
    else:
        x_init = x_init_override
    x_array = np.zeros((6, N + 1))
    x_array[:, 0] = x_init
    u_array = np.ones((2, N)) * np.array([[0.15, 0.15]]).T
    # print(x_array)
    if shrunk:
        """
        NOTE: Here the shrunk vectors are defined for N+1 timesteps for ease of understanding. However, as we well know,
        we are making the assumption that the initial state is known with certainty and hence there is no shrinking there.
        Hence when adding b_shrunk_x_array to the 'x' vector, we only add cs.vec(b_shrunk_x_array[:, 1:]) since the first vector
        is not a variable but a hard-coded vector = controller_inst.X.b_np 
        """
        Sigma_d_array = np.zeros((n_d, N))
        Sigma_size = n_x+n_u+n_d
        # Sigma_array = [np.zeros((Sigma_size, Sigma_size)) for k in range(N)]
        Sigma_x_array = [np.zeros((n_x, n_x)) for k in range(N+1)]
        # Sigma_x_array[0] += np.eye(n_x) * 1e-4
        assert n_d == 1, "This test only works for one dimensional disturbances currently."
        b_shrunk_x_array = np.zeros((2*n_x, N + 1))
    mu_d_array = np.zeros((n_d, N))
    num_regions = controller_inst.num_regions
    hld_mat = np.zeros((num_regions, (N + 1)))
    Bd = controller_inst.Bd
    assert n_d == 1, "This test only works for one dimensional disturbances currently."
    # Region delta assignment step
    for k in range(N):
        # 0 pad the inputs since we have already checked the deltas don't depend on them using the assertion above.
        joint_vec = np.vstack([x_array[:, [k]], np.zeros([n_u, 1])]).squeeze()
        mu_x, mu_u = controller_inst.opti_dict["mu_x"], controller_inst.opti_dict["mu_u"]
        _, gp_inp_vec_size, gp_input = controller_inst.create_gp_inp_sym(mu_x, mu_u)
        mean_res_compute_fns, _ = controller_inst.setup_hybrid_means_manual_preds(gp_inp_vec_size, gp_input)
        if shrunk:
            sigma_res_compute_fns, _ = controller_inst.setup_hybrid_covs_manual_preds(gp_inp_vec_size, gp_input)
            assert n_d == 1, "This test only works for one dimensional disturbances currently."
        gp_inp_vec = np.array(controller_inst.gp_input_mask @ joint_vec, ndmin=2).reshape((-1, 1))
        for region_idx, region in enumerate(controller_inst.regions):
            hld_mat[region_idx, k] = 1 if region.check_satisfaction(
                (controller_inst.delta_input_mask @ joint_vec).T).item() is True else 0
            """
            Note the 0 index here is required since the res compute function expects an input with N columns (since that's what the opt var will be).
            However here, it's just for a single timestep's vector and gets broadcasted automatically by casadi to a matrix with N columns thus 
            yielding an output with N columns having *identical* entries due to the broadcasting. As a result, the [0] index is just to pull out one of
            those identical values.
            """
            mu_d_r_k = mean_res_compute_fns[region_idx](gp_inp_vec)[0]
            # print(mu_d_r_k)
            mu_d_array[:, [k]] = mu_d_array[:, [k]] + hld_mat[region_idx, k] * mu_d_r_k
            if shrunk:
                sigma_d_r_k = sigma_res_compute_fns[region_idx](gp_inp_vec)[0]
                delta_elem = hld_mat[region_idx, k]
                test_softplus = True
                if test_softplus:
                    delta_elem = np.log(1+np.exp(75*delta_elem))/75
                Sigma_d_array[:, [k]] = Sigma_d_array[:, [k]] + delta_elem * sigma_d_r_k
            try:
                assert hld_mat[region_idx, k] == (1 if (
                            controller_inst.region_Hs[region_idx] @ np.array(controller_inst.delta_input_mask @ joint_vec,
                                                                             ndmin=2).T <= controller_inst.region_bs[
                                region_idx]).all() else 0)
            except AssertionError:
                print("Region {} not satisfied by delta input mask.".format(region_idx))
        x_array[:, [k+1]] = x_array[:, [k]] + sampling_time * (controller_inst.ct_dyn_nom(x_array[:, [k]], u_array[:, [k]]) +
                                                               Bd @ mu_d_array[:, [k]])
        if shrunk:
            test_Sigma_d_calc = False
            test_Sigma_x_calc = False
            if test_Sigma_d_calc:
                print(Sigma_d_array)
            affine_transform = controller_inst.create_affine_transform(x_lin=x_array[:, k], u_lin=u_array[:, k], cs_equivalent=False)
            Sigma_i = controller_inst.computesigma_wrapped(Sigma_x_array[k], Sigma_d_array[:, [k]])
            Sigma_x_array[k+1] = controller_inst.dt * (affine_transform @ Sigma_i @ affine_transform.T)
            if test_Sigma_x_calc:
                print(Sigma_x_array)
            """
            1e-4 is a delta tol that is also added in the controller class to prevent numerical issues. Note this is not incorporated into Sigma_x computation
            of the controller class since it would compound over the horizon. Rather, it is included here to prevent numerical issues in the computation of
            the shrunk b array.
            """
            delta_tol = 1e-4
            if delta_tol is None:
                sqrt_arr = np.array(np.sqrt(np.diag(Sigma_x_array[k+1])), ndmin=2).reshape((-1, 1))
            else:
                sqrt_arr = np.array(np.sqrt(np.diag(Sigma_x_array[k+1]) + delta_tol), ndmin=2).reshape((-1, 1))
            # sqrt_arr = cs.sqrt(cs.diag(Sigma_x_array[k+1]))
            # sqrt_arr = cs.mmax([sqrt_arr, cs.DM.ones(2*n_x, 1)*1e-4])
            b_shrunk_x_array[:, [k+1]] = controller_inst.X.b_np - (np.fabs(controller_inst.X.H_np) @ (sqrt_arr * controller_inst.inverse_cdf_x))
    # print(hld_mat)

    # Note: All arrays need to be flattened in a certain order for them to match the order in which they have been added to the opti inst.
    # mu_d constraints are going to be neglected for this part so I'll just set them to zero.
    if not shrunk:
        x = cs.vertcat(cs.vec(x_array), cs.vec(u_array), cs.vec(mu_d_array), cs.vec(hld_mat))
    else:
        x = cs.vertcat(cs.vec(x_array), cs.vec(u_array), cs.vec(b_shrunk_x_array[:, 1:]), cs.vec(mu_d_array), cs.vec(hld_mat))
    # print(x)

    # Param vec creation. Placeholders for the most part except for x_init.
    mu_x_size, mu_u_size, mu_d_size = n_x * (N + 1), n_u * N, n_d * (N + 1)
    x_des_shape, u_des_shape = np.zeros(mu_x_size), np.zeros(mu_u_size)
    P_mat_shape = np.zeros((n_x * n_x))
    p = np.concatenate((x_init.flatten(order='F'), x_des_shape, u_des_shape, P_mat_shape))

    assert x.shape[0] == opti_inst.nx,\
        "x vector for debugging is not the correct size. x.shape[0]: %s, opti_inst.nx: %s" % (x.shape[0], opti_inst.nx)
    assert p.shape[0] == opti_inst.np,\
        "p vector for debugging is not the correct size. p.shape[0]: %s, opti_inst.np: %s" % (p.shape[0], opti_inst.np)

    return x, p


def gpmpc_MINLP_controller_A(gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7, early_termination=True,
                             minimal_print=True, plot_shrunk_sets=False,
                             N=3, closed_loop=False, simulation_length=5,
                             no_lam_p=False, hessian_approximation=False,
                             show_plot=True, plot_ol_traj=False,
                             problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True,
                             ignore_init_constr_check=False, add_scaling=False,
                             ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3,
                             add_delta_constraints=True, true_ds_inst=None,
                             infeas_debug_callback=False, plot_shrunk_sep=False, waypoint_arr=None, add_monitor=False,
                             enable_softplus=True, shrink_by_param=True, debug_only=False):

    """
    include_res_in_ctrl removed since this test uses only nominal dynamics to test correct setting of the delta variables.
    fwd_sim also removed for the same reason since we're using nom_dyn for this test.
    Bd_fwd_sim and Bd explicitly set to zero vectors for the same reason.
    skip_shrinking removed and set to True below since no residual covariance dynamics.
    ignore_variance_cost also removed since no residual covariance dynamics.
    """

    fwd_sim = "nom_dyn"
    skip_shrinking = True
    ignore_variance_costs = True

    sys_config_dict = problem_setup_fn(velocity_limit_override=velocity_override)
    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": np.zeros((sys_config_dict["n_x"], 1)),
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "satisfaction_prob": satisfaction_prob, "skip_shrinking": skip_shrinking,
                      "solver": "bonmin", "enable_softplus": enable_softplus, "shrink_by_param": shrink_by_param,
                      "ignore_variance_costs": ignore_variance_costs}
    np.random.seed(np.random.randint(0, 1000000))

    # Neglects residual mean dynamics in the controller. Variance info already excluded from the formulation for these increments.
    sys_config_dict.update({"Bd": np.zeros((6, 1))})

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts_minlp(print_level=print_level, add_monitor=add_monitor,
                                       early_termination=early_termination, hsl_solver=False,
                                       test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation)
    fn_config_dict["addn_solver_opts"] = opts

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = GPMPC_MINLP(**configs)

    if debug_only:
        opti_inst = controller_inst.opti
        opti_dict = controller_inst.opti_dict
        order_check = False
        var_check = False
        constr_check = True
        test_case_gen = True
        n_x, n_u, n_d = sys_config_dict["n_x"], sys_config_dict["n_u"], sys_config_dict["n_d"]
        num_regions = controller_inst.num_regions
        if order_check:
            MINLP_order_check(sys_config_dict, opti_dict, opti_inst, controller_inst, N)
        if var_check:
            MINLP_var_check(opti_inst, opti_dict)
        if constr_check:
            MINLP_constr_check(opti_inst, n_x, n_u, n_d, N, num_regions, controller_inst, ignore_init_constr_check, shrunk=False)
        if test_case_gen:
            x, p = setup_minlp_sanity_test(N, sampling_time, opti_inst, controller_inst, n_u, n_x, n_d)

            assert_violation = True
            f = cs.Function('f', [opti_inst.x, opti_inst.p], [opti_inst.g])
            constr_viol_split = cs.vertsplit(f(x, p), 1)
            print(constr_viol_split)
            eq_constraint_idxs = []
            ineq_constraint_idxs = []
            constr_viol = 0
            num_viols = 0
            for i in range(opti_inst.ng):
                if '==' in opti_inst.debug.g_describe(i):
                    # For this step, can ignore any constraint violations for the mu_d[:, k] = mean_vec constraint.
                    # We'll fix this in B). mu_d will still be calc'd here but Bd will zero out its influence.
                    if "mean_vec - mu_d_arr" not in opti_inst.debug.g_describe(i):
                        if np.abs(constr_viol_split[i]) >= 1e-5:
                            print("Error")
                            print(i, opti_inst.debug.g_describe(i))
                            print(constr_viol_split[i])
                            num_viols += 1
                        if assert_violation:
                            assert np.abs(constr_viol_split[i]) <= 1e-5, "%s %s %s" % (
                            i, opti_inst.debug.g_describe(i), constr_viol_split[i])
                        eq_constraint_idxs.append(i)
                        constr_viol += np.abs(constr_viol_split[i])
                else:
                    ineq_constraint_idxs.append(i)
                    if assert_violation:
                        try:
                            assert constr_viol_split[i] <= 0
                        except AssertionError:
                            print("Error")
                            print("%s %s %s" % (i, opti_inst.debug.g_describe(i), constr_viol_split[i]))
                            if "big_M" in opti_inst.debug.g_describe(i):
                                for region in controller_inst.regions:
                                    region_b = region.b_np
                                    big_M_vec = controller_inst.big_M
                                    adjusted_vec = region_b + big_M_vec
                                    print(region_b)
                                    print(big_M_vec)
                                    print(adjusted_vec)
                                    print()

            if constr_viol <= 1e-4:
                print("All constraints passed with total violation less than 1e-4")
            print("Total constraint violation: %s" % constr_viol)
            print("Number of violations: %s" % num_viols)

    else:
        # Pull relevant info from config dict.
        n_u = sys_config_dict["n_u"]

        # Nominal MPC solution for warmstarting.
        x_desired, u_desired = test_quad_2d_track(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                                  viz=False, verbose=False, N=N, velocity_override=velocity_override,
                                                  ret_inputs=True, sampling_time=sampling_time, x_init=x_init)

        print(x_desired)
        print(u_desired)

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
                                                                    opt_verbose=verbose, x_desired=x_desired,
                                                                    u_desired=u_desired,
                                                                    infeas_debug_callback=infeas_debug_callback)

            mu_x_cl = [data_dict_ol['mu_x'] for data_dict_ol in data_dict_cl]
            if show_plot:
                mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                            data_dict_cl=data_dict_cl,
                                                            ret_mu_x_cl=True,
                                                            state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                            plot_ol_traj=plot_ol_traj, axes=None,
                                                            ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')


def gpmpc_MINLP_controller_B(gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7, early_termination=True,
                             minimal_print=True, plot_shrunk_sets=False,
                             N=3, closed_loop=False, simulation_length=5,
                             no_lam_p=False, hessian_approximation=False,
                             show_plot=True, plot_ol_traj=False,
                             problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True,
                             ignore_init_constr_check=False, add_scaling=False,
                             ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3,
                             add_delta_constraints=True, true_ds_inst=None,
                             infeas_debug_callback=False, plot_shrunk_sep=False, waypoint_arr=None, add_monitor=False,
                             enable_softplus=True, shrink_by_param=True, debug_only=False):

    """
    include_res_in_ctrl removed since this test uses only nominal dynamics to test correct setting of the delta variables.
    fwd_sim also removed for the same reason since we're using nom_dyn for this test.
    Bd_fwd_sim and Bd explicitly set to zero vectors for the same reason.
    skip_shrinking removed and set to True below since no residual covariance dynamics.
    ignore_variance_cost also removed since no residual covariance dynamics.
    """

    fwd_sim = "w_pw_res"
    skip_shrinking = True
    ignore_variance_costs = True

    sys_config_dict = problem_setup_fn(velocity_limit_override=velocity_override)
    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": problem_setup_fn()["Bd"],
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "satisfaction_prob": satisfaction_prob, "skip_shrinking": skip_shrinking,
                      "solver": "bonmin", "enable_softplus": enable_softplus, "shrink_by_param": shrink_by_param,
                      "ignore_variance_costs": ignore_variance_costs}
    np.random.seed(np.random.randint(0, 1000000))

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts_minlp(print_level=print_level, add_monitor=add_monitor,
                                       early_termination=early_termination, hsl_solver=False,
                                       test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation)
    fn_config_dict["addn_solver_opts"] = opts

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = GPMPC_MINLP(**configs)

    if debug_only:
        opti_inst = controller_inst.opti
        opti_dict = controller_inst.opti_dict
        order_check = False
        var_check = False
        constr_check = True
        test_case_gen = True
        n_x, n_u, n_d = sys_config_dict["n_x"], sys_config_dict["n_u"], sys_config_dict["n_d"]
        num_regions = controller_inst.num_regions
        if order_check:
            MINLP_order_check(sys_config_dict, opti_dict, opti_inst, controller_inst, N)
        if var_check:
            MINLP_var_check(opti_inst, opti_dict)
        if constr_check:
            MINLP_constr_check(opti_inst, n_x, n_u, n_d, N, num_regions, controller_inst, ignore_init_constr_check, shrunk=False)
        if test_case_gen:
            x, p = setup_minlp_res_test(N, sampling_time, opti_inst, controller_inst, n_u, n_x, n_d)

            assert_violation = True
            f = cs.Function('f', [opti_inst.x, opti_inst.p], [opti_inst.g])
            constr_viol_split = cs.vertsplit(f(x, p), 1)
            print(constr_viol_split)
            eq_constraint_idxs = []
            ineq_constraint_idxs = []
            constr_viol = 0
            num_viols = 0
            for i in range(opti_inst.ng):
                if '==' in opti_inst.debug.g_describe(i):
                    if np.abs(constr_viol_split[i]) >= 1e-5:
                        print("Error")
                        print(i, opti_inst.debug.g_describe(i))
                        print(constr_viol_split[i])
                        num_viols += 1
                    if assert_violation:
                        assert np.abs(constr_viol_split[i]) <= 1e-5, "%s %s %s" % (
                        i, opti_inst.debug.g_describe(i), constr_viol_split[i])
                    eq_constraint_idxs.append(i)
                    constr_viol += np.abs(constr_viol_split[i])
                else:
                    ineq_constraint_idxs.append(i)
                    if assert_violation:
                        try:
                            assert constr_viol_split[i] <= 0
                        except AssertionError:
                            print("Error")
                            print("%s %s %s" % (i, opti_inst.debug.g_describe(i), constr_viol_split[i]))
                            if "big_M" in opti_inst.debug.g_describe(i):
                                for region in controller_inst.regions:
                                    region_b = region.b_np
                                    big_M_vec = controller_inst.big_M
                                    adjusted_vec = region_b + big_M_vec
                                    print(region_b)
                                    print(big_M_vec)
                                    print(adjusted_vec)
                                    print()

            if constr_viol <= 1e-4:
                print("All constraints passed with total violation less than 1e-4")
            print("Total constraint violation: %s" % constr_viol)
            print("Number of violations: %s" % num_viols)

    else:
        # Pull relevant info from config dict.
        n_u = sys_config_dict["n_u"]

        # Nominal MPC solution for warmstarting.
        x_desired, u_desired = test_quad_2d_track(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                                  viz=False, verbose=False, N=N, velocity_override=velocity_override,
                                                  ret_inputs=True, sampling_time=sampling_time, x_init=x_init)

        print(x_desired)
        print(u_desired)

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
                                                                    opt_verbose=verbose, x_desired=x_desired,
                                                                    u_desired=u_desired,
                                                                    infeas_debug_callback=infeas_debug_callback)

            mu_x_cl = [data_dict_ol['mu_x'] for data_dict_ol in data_dict_cl]
            if show_plot:
                mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                            data_dict_cl=data_dict_cl,
                                                            ret_mu_x_cl=True,
                                                            state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                            plot_ol_traj=plot_ol_traj, axes=None,
                                                            ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center')


def gpmpc_MINLP_controller_C(gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7, early_termination=True,
                             minimal_print=True, plot_shrunk_sets=False,
                             N=3, closed_loop=False, simulation_length=5,
                             no_lam_p=False, hessian_approximation=False,
                             show_plot=True, plot_ol_traj=False,
                             problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True, track_gen_fn=test_quad_2d_track,
                             ignore_init_constr_check=False, add_scaling=False,
                             ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3,
                             x_threshold=7, z_threshold=7, x0_delim=2, x1_delim=4, x_min_threshold=-7,
                             add_delta_constraints=True, true_ds_inst=None, add_monitor=False,
                             infeas_debug_callback=False, plot_shrunk_sep=False, waypoint_arr=((7, 0), (7, 7), (0, 7), (0, 0)),
                             enable_softplus=True, debug_only=False, ignore_variance_costs=True,
                             override_debug_vector=False, ignore_cost=False, periodic_callback_freq=None,
                             shrink_by_param=True, collision_check=False, integration_method='euler', simulation_length_override=None,
                             online_N=None, read_desired_from_file=None, use_prev_if_infeas=False,
                             Q_override=None, R_override=None, Bd_override=None):

    """
    include_res_in_ctrl removed since this test uses only nominal dynamics to test correct setting of the delta variables.
    fwd_sim also removed for the same reason since we're using nom_dyn for this test.
    Bd_fwd_sim and Bd explicitly set to zero vectors for the same reason.
    skip_shrinking removed and set to True below since no residual covariance dynamics.
    ignore_variance_cost also removed since no residual covariance dynamics.
    """

    fwd_sim = "w_pw_res"
    skip_shrinking = False

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
                                                          x1_threshold=x_threshold, x2_threshold=z_threshold,
                                                          x0_delim=x0_delim, x1_delim=x1_delim, ret_inst=True)
        limiter_names = ["x1_threshold", "x2_threshold"]
        # print(limiter_names)
    else:
        sys_config_dict, inst_override = problem_setup_fn(ret_inst=True)
    lti = inst_override.lti
    if lti and integration_method == 'exact':
        if not inst_override.symbolic.exact_flag:
            print("Exact linearization not available for this system. Using Euler integration instead.")
            integration_method = 'euler'

    true_ds_inst.regions = sys_config_dict["regions"]

    if Q_override is not None:
        sys_config_dict["Q"] = Q_override
    if R_override is not None:
        sys_config_dict["R"] = R_override
    if Bd_override is not None:
        sys_config_dict["Bd"] = Bd_override

    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N if online_N is None else online_N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": problem_setup_fn()["Bd"],
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "satisfaction_prob": satisfaction_prob, "skip_shrinking": skip_shrinking,
                      "solver": "bonmin", "enable_softplus": enable_softplus, "shrink_by_param": shrink_by_param,
                      "ignore_variance_costs": ignore_variance_costs, "ignore_cost": ignore_cost,
                      "periodic_callback_freq": periodic_callback_freq, "collision_check": collision_check,
                      "integration_method": integration_method, "lti": lti, "use_prev_if_infeas": use_prev_if_infeas}
    np.random.seed(np.random.randint(0, 1000000))

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts_minlp(print_level=print_level, add_monitor=add_monitor,
                                       early_termination=early_termination, hsl_solver=False,
                                       test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation)
    fn_config_dict["addn_solver_opts"] = opts

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = GPMPC_MINLP(**configs)

    # print(controller_inst.opti.debug.g_describe(220))

    if debug_only:
        opti_inst = controller_inst.opti
        opti_dict = controller_inst.opti_dict
        order_check = False
        var_check = False
        constr_check = False
        test_case_gen = True
        gradient_check = True
        n_x, n_u, n_d = sys_config_dict["n_x"], sys_config_dict["n_u"], sys_config_dict["n_d"]
        num_regions = controller_inst.num_regions
        if order_check:
            MINLP_order_check(sys_config_dict, opti_dict, opti_inst, controller_inst, N, shrunk=True)
        if var_check:
            MINLP_var_check(opti_inst, opti_dict)
        if constr_check:
            MINLP_constr_check(opti_inst, n_x, n_u, n_d, N, num_regions, controller_inst, ignore_init_constr_check, shrunk=True)
        if test_case_gen:
            if override_debug_vector:
                x_init_override = x_init.squeeze()
            else:
                x_init_override = None
            x, p = setup_minlp_res_test(N, sampling_time, opti_inst, controller_inst, n_u, n_x, n_d, shrunk=True,
                                        x_init_override=x_init_override)
            test_Sigma_d_calc = False
            test_Sigma_x_calc = False
            if test_Sigma_d_calc:
                # Note to compare against Sigma_d computed in setup_minlp_res_test, print the Sigma_d's in the function and
                # then compare against values printed here.
                for i in range(N):
                    Sigma_d_k = cs.Function('f', [opti_inst.x, opti_inst.p], [opti_dict["Sigma_d"][i]])
                    print(Sigma_d_k(x, p))
            if test_Sigma_x_calc:
                # Note to compare against Sigma_d computed in setup_minlp_res_test, print the Sigma_d's in the function and
                # then compare against values printed here.
                for i in range(1, N+1):
                    Sigma_x_k = cs.Function('f', [opti_inst.x, opti_inst.p], [opti_dict["Sigma_x"][i]])
                    print(Sigma_x_k(x, p))

            assert_violation = False
            f = cs.Function('f', [opti_inst.x, opti_inst.p], [opti_inst.g])
            if override_debug_vector:
                x = [-3.11213e-012, -3.71074e-012, -2.5384e-010, -3.63648e-011, -9.18095e-012, -2.56273e-012, 0.0242498, 1.03058e-009, -6.39651e-010, -0.148177, -3.529e-011, 1.78735, 0.0343162, -1.09773e-009, -0.0296355, 0.527793, 0.35747, -1.59217, 0.10288, 0.299861, 0.0759232, -0.629419, 0.0390351, -0.371791, 0.12635, 0.400985, -0.0499607, -8.58017e-005, -0.0353231, 0.304823, 0.230788, 0.331688, -0.0499779, 0.00145946, 0.0256414, 0.554748, 0.321522, 0.381916, -0.049686, -0.00141534, 0.136591, 0.52024, 0.422023, 0.65154, -0.0499691, 0.000194645, 0.240639, 0.239054, 0.576602, 1.13238, -0.0499301, -0.000250473, 0.28845, -0.344714, 0.827304, 1.80347, -0.0499802, 0.30144, 0.219507, -1.21605, 1.23917, 2.38696, 0.0103078, 0.956529, -0.0237031, -1.31428, 0.12007, 0.124526, 0.182142, 0.173714, 0.0563234, 0.0593665, 0.17407, 0.175757, 0.132175, 0.132798, 0.132192, 0.132106, 0.134004, 0.133303, 0.136921, 0.135466, 0.160329, 0.158156, 0.18098, 0.180735, 6.93025, 5, 0.05, 5, 1.48353, 100, 6.93025, 5, 7, 5, 1.48353, 100, 6.92306, 5, 0.05, 5, 1.48353, 100, 6.92306, 5, 7, 5, 1.48353, 100, 6.92273, 5, 0.05, 5, 1.48353, 100, 6.92273, 5, 7, 5, 1.48353, 100, 6.92144, 5, 0.05, 5, 1.48353, 100, 6.92144, 5, 7, 5, 1.48353, 100, 6.92119, 5, 0.05, 5, 1.48353, 100, 6.92119, 5, 7, 5, 1.48353, 100, 6.92113, 5, 0.05, 5, 1.48353, 100, 6.92113, 5, 7, 5, 1.48353, 100, 6.92112, 5, 0.05, 5, 1.48353, 100, 6.92112, 5, 7, 5, 1.48353, 100, 6.92112, 5, 0.05, 5, 1.48353, 100, 6.92112, 5, 7, 5, 1.48353, 100, 6.92112, 5, 0.05, 5, 1.48353, 100, 6.92112, 5, 7, 5, 1.48353, 100, 6.92112, 5, 0.05, 5, 1.48353, 100, 6.92112, 5, 7, 5, 1.48353, 100, 0.121249, 0.0503322, 0.342821, -0.182514, 0.121209, 0.121982, 0.120586, 0.121354, 0.121132, 0.255853, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
            constr_viol_split = cs.vertsplit(f(x, p), 1)
            print(constr_viol_split)
            eq_constraint_idxs = []
            ineq_constraint_idxs = []
            constr_viol = 0
            num_viols = 0
            for i in range(opti_inst.ng):
                if '==' in opti_inst.debug.g_describe(i):
                    if np.abs(constr_viol_split[i]) >= 1e-4:
                        print("Error")
                        print(i, opti_inst.debug.g_describe(i))
                        print(constr_viol_split[i])
                        num_viols += 1
                    if assert_violation:
                        assert np.abs(constr_viol_split[i]) <= 1e-4, "%s %s %s" % (
                        i, opti_inst.debug.g_describe(i), constr_viol_split[i])
                    eq_constraint_idxs.append(i)
                    constr_viol += np.abs(constr_viol_split[i])
                else:
                    ineq_constraint_idxs.append(i)
                    if assert_violation:
                        try:
                            assert constr_viol_split[i] <= 0
                        except AssertionError:
                            print("Error")
                            print("%s %s %s" % (i, opti_inst.debug.g_describe(i), constr_viol_split[i]))
                            if "big_M" in opti_inst.debug.g_describe(i):
                                for region in controller_inst.regions:
                                    region_b = region.b_np
                                    big_M_vec = controller_inst.big_M
                                    adjusted_vec = region_b + big_M_vec
                                    print(region_b)
                                    print(big_M_vec)
                                    print(adjusted_vec)
                                    print()

            if constr_viol <= 1e-4:
                print("All constraints passed with total violation less than 1e-4")
            print("Total constraint violation: %s" % constr_viol)
            print("Number of violations: %s" % num_viols)
        if gradient_check:
            # print(opti_inst.g[90])
            mu_x_sym = cs.MX.sym('mu_x_sym', controller_inst.n_x, 1)
            for i in range(1, N+1):
                Sigma_x_k = cs.Function('f', [opti_inst.x, opti_inst.p], [opti_dict["Sigma_x"][i]])
                print(Sigma_x_k(x, p))
            test_gradient = cs.Function('grad_comp', [opti_inst.x, opti_inst.p],
                                        [cs.jacobian(opti_inst.g, opti_inst.x)])
            print(test_gradient(x, p)[120, :])


    else:
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

        print(x_desired)
        print(u_desired)

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
            simulation_length = simulation_length_override if simulation_length_override is not None else simulation_length
            data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                    opt_verbose=verbose, x_desired=x_desired,
                                                                    u_desired=u_desired,
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


def gpmpc_MINLP_controller_C_multiple(gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7, early_termination=True,
                             minimal_print=True, plot_shrunk_sets=False,
                             N=3, closed_loop=False, simulation_length=5,
                             no_lam_p=False, hessian_approximation=False,
                             show_plot=True, plot_ol_traj=False,
                             problem_setup_fn=quad_2d_sys_1d_inp_res, verbose=True, track_gen_fn=test_quad_2d_track,
                             ignore_init_constr_check=False, add_scaling=False,
                             ignore_callback=False, num_discrete=50, sampling_time=20 * 10e-3,
                             x_threshold=7, z_threshold=7, x0_delim=2, x1_delim=4, x_min_threshold=-7,
                             add_delta_constraints=True, true_ds_inst=None, add_monitor=False,
                             infeas_debug_callback=False, plot_shrunk_sep=False, waypoint_arr=((7, 0), (7, 7), (0, 7), (0, 0)),
                             enable_softplus=True, debug_only=False, ignore_variance_costs=True,
                             override_debug_vector=False, ignore_cost=False, periodic_callback_freq=None,
                             shrink_by_param=True, collision_check=False, integration_method='euler', simulation_length_override=None,
                             online_N=None, read_desired_from_file=None, use_prev_if_infeas=False, Q_override=None, R_override=None, Bd_override=None,
                                      num_multiple=5, data_save_file='gpmpc_d_runs'):

    """
    include_res_in_ctrl removed since this test uses only nominal dynamics to test correct setting of the delta variables.
    fwd_sim also removed for the same reason since we're using nom_dyn for this test.
    Bd_fwd_sim and Bd explicitly set to zero vectors for the same reason.
    skip_shrinking removed and set to True below since no residual covariance dynamics.
    ignore_variance_cost also removed since no residual covariance dynamics.
    """

    fwd_sim = "w_pw_res"
    skip_shrinking = False

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
                                                          x1_threshold=x_threshold, x2_threshold=z_threshold,
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

    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N if online_N is None else online_N, 'piecewise': True,
                      'ignore_init_constr_check': ignore_init_constr_check, "add_scaling": add_scaling,
                      "ignore_callback": ignore_callback, "sampling_time": sampling_time,
                      "add_delta_constraints": add_delta_constraints, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": problem_setup_fn()["Bd"],
                      "infeas_debug_callback": (infeas_debug_callback and not closed_loop),
                      "satisfaction_prob": satisfaction_prob, "skip_shrinking": skip_shrinking,
                      "solver": "bonmin", "enable_softplus": enable_softplus, "shrink_by_param": shrink_by_param,
                      "ignore_variance_costs": ignore_variance_costs, "ignore_cost": ignore_cost,
                      "periodic_callback_freq": periodic_callback_freq, "collision_check": collision_check,
                      "integration_method": integration_method, "lti": lti, "use_prev_if_infeas": use_prev_if_infeas}
    np.random.seed(np.random.randint(0, 1000000))

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts_minlp(print_level=print_level, add_monitor=add_monitor,
                                       early_termination=early_termination, hsl_solver=False,
                                       test_no_lam_p=no_lam_p, hessian_approximation=hessian_approximation)
    fn_config_dict["addn_solver_opts"] = opts

    # Instantiate controller and setup optimization problem to be solved.
    controller_inst = GPMPC_MINLP(**configs)

    # print(controller_inst.opti.debug.g_describe(220))

    if debug_only:
        opti_inst = controller_inst.opti
        opti_dict = controller_inst.opti_dict
        order_check = False
        var_check = False
        constr_check = False
        test_case_gen = True
        gradient_check = True
        n_x, n_u, n_d = sys_config_dict["n_x"], sys_config_dict["n_u"], sys_config_dict["n_d"]
        num_regions = controller_inst.num_regions
        if order_check:
            MINLP_order_check(sys_config_dict, opti_dict, opti_inst, controller_inst, N, shrunk=True)
        if var_check:
            MINLP_var_check(opti_inst, opti_dict)
        if constr_check:
            MINLP_constr_check(opti_inst, n_x, n_u, n_d, N, num_regions, controller_inst, ignore_init_constr_check, shrunk=True)
        if test_case_gen:
            if override_debug_vector:
                x_init_override = x_init.squeeze()
            else:
                x_init_override = None
            x, p = setup_minlp_res_test(N, sampling_time, opti_inst, controller_inst, n_u, n_x, n_d, shrunk=True,
                                        x_init_override=x_init_override)
            test_Sigma_d_calc = False
            test_Sigma_x_calc = False
            if test_Sigma_d_calc:
                # Note to compare against Sigma_d computed in setup_minlp_res_test, print the Sigma_d's in the function and
                # then compare against values printed here.
                for i in range(N):
                    Sigma_d_k = cs.Function('f', [opti_inst.x, opti_inst.p], [opti_dict["Sigma_d"][i]])
                    print(Sigma_d_k(x, p))
            if test_Sigma_x_calc:
                # Note to compare against Sigma_d computed in setup_minlp_res_test, print the Sigma_d's in the function and
                # then compare against values printed here.
                for i in range(1, N+1):
                    Sigma_x_k = cs.Function('f', [opti_inst.x, opti_inst.p], [opti_dict["Sigma_x"][i]])
                    print(Sigma_x_k(x, p))

            assert_violation = False
            f = cs.Function('f', [opti_inst.x, opti_inst.p], [opti_inst.g])
            if override_debug_vector:
                x = [-3.11213e-012, -3.71074e-012, -2.5384e-010, -3.63648e-011, -9.18095e-012, -2.56273e-012, 0.0242498, 1.03058e-009, -6.39651e-010, -0.148177, -3.529e-011, 1.78735, 0.0343162, -1.09773e-009, -0.0296355, 0.527793, 0.35747, -1.59217, 0.10288, 0.299861, 0.0759232, -0.629419, 0.0390351, -0.371791, 0.12635, 0.400985, -0.0499607, -8.58017e-005, -0.0353231, 0.304823, 0.230788, 0.331688, -0.0499779, 0.00145946, 0.0256414, 0.554748, 0.321522, 0.381916, -0.049686, -0.00141534, 0.136591, 0.52024, 0.422023, 0.65154, -0.0499691, 0.000194645, 0.240639, 0.239054, 0.576602, 1.13238, -0.0499301, -0.000250473, 0.28845, -0.344714, 0.827304, 1.80347, -0.0499802, 0.30144, 0.219507, -1.21605, 1.23917, 2.38696, 0.0103078, 0.956529, -0.0237031, -1.31428, 0.12007, 0.124526, 0.182142, 0.173714, 0.0563234, 0.0593665, 0.17407, 0.175757, 0.132175, 0.132798, 0.132192, 0.132106, 0.134004, 0.133303, 0.136921, 0.135466, 0.160329, 0.158156, 0.18098, 0.180735, 6.93025, 5, 0.05, 5, 1.48353, 100, 6.93025, 5, 7, 5, 1.48353, 100, 6.92306, 5, 0.05, 5, 1.48353, 100, 6.92306, 5, 7, 5, 1.48353, 100, 6.92273, 5, 0.05, 5, 1.48353, 100, 6.92273, 5, 7, 5, 1.48353, 100, 6.92144, 5, 0.05, 5, 1.48353, 100, 6.92144, 5, 7, 5, 1.48353, 100, 6.92119, 5, 0.05, 5, 1.48353, 100, 6.92119, 5, 7, 5, 1.48353, 100, 6.92113, 5, 0.05, 5, 1.48353, 100, 6.92113, 5, 7, 5, 1.48353, 100, 6.92112, 5, 0.05, 5, 1.48353, 100, 6.92112, 5, 7, 5, 1.48353, 100, 6.92112, 5, 0.05, 5, 1.48353, 100, 6.92112, 5, 7, 5, 1.48353, 100, 6.92112, 5, 0.05, 5, 1.48353, 100, 6.92112, 5, 7, 5, 1.48353, 100, 6.92112, 5, 0.05, 5, 1.48353, 100, 6.92112, 5, 7, 5, 1.48353, 100, 0.121249, 0.0503322, 0.342821, -0.182514, 0.121209, 0.121982, 0.120586, 0.121354, 0.121132, 0.255853, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
            constr_viol_split = cs.vertsplit(f(x, p), 1)
            print(constr_viol_split)
            eq_constraint_idxs = []
            ineq_constraint_idxs = []
            constr_viol = 0
            num_viols = 0
            for i in range(opti_inst.ng):
                if '==' in opti_inst.debug.g_describe(i):
                    if np.abs(constr_viol_split[i]) >= 1e-4:
                        print("Error")
                        print(i, opti_inst.debug.g_describe(i))
                        print(constr_viol_split[i])
                        num_viols += 1
                    if assert_violation:
                        assert np.abs(constr_viol_split[i]) <= 1e-4, "%s %s %s" % (
                        i, opti_inst.debug.g_describe(i), constr_viol_split[i])
                    eq_constraint_idxs.append(i)
                    constr_viol += np.abs(constr_viol_split[i])
                else:
                    ineq_constraint_idxs.append(i)
                    if assert_violation:
                        try:
                            assert constr_viol_split[i] <= 0
                        except AssertionError:
                            print("Error")
                            print("%s %s %s" % (i, opti_inst.debug.g_describe(i), constr_viol_split[i]))
                            if "big_M" in opti_inst.debug.g_describe(i):
                                for region in controller_inst.regions:
                                    region_b = region.b_np
                                    big_M_vec = controller_inst.big_M
                                    adjusted_vec = region_b + big_M_vec
                                    print(region_b)
                                    print(big_M_vec)
                                    print(adjusted_vec)
                                    print()

            if constr_viol <= 1e-4:
                print("All constraints passed with total violation less than 1e-4")
            print("Total constraint violation: %s" % constr_viol)
            print("Number of violations: %s" % num_viols)
        if gradient_check:
            # print(opti_inst.g[90])
            mu_x_sym = cs.MX.sym('mu_x_sym', controller_inst.n_x, 1)
            for i in range(1, N+1):
                Sigma_x_k = cs.Function('f', [opti_inst.x, opti_inst.p], [opti_dict["Sigma_x"][i]])
                print(Sigma_x_k(x, p))
            test_gradient = cs.Function('grad_comp', [opti_inst.x, opti_inst.p],
                                        [cs.jacobian(opti_inst.g, opti_inst.x)])
            print(test_gradient(x, p)[120, :])
    else:
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

        print(x_desired)
        print(u_desired)

        u_desired = np.array(u_desired, ndmin=2).reshape((n_u, -1))
        tracking_matrix = sys_config_dict["tracking_matrix"]
        waypoints_to_track = tracking_matrix[:, :controller_inst.n_x] @ x_desired
        initial_info_dict = {"x_init": x_init, "u_warmstart": u_desired}

        # print(controller_inst.opti.)

        data_store_dicts = []
        for run_idx in range(num_multiple):
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
                simulation_length = simulation_length_override if simulation_length_override is not None else simulation_length
                run_start_time = time.time()
                data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                        opt_verbose=verbose, x_desired=x_desired,
                                                                        u_desired=u_desired,
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
