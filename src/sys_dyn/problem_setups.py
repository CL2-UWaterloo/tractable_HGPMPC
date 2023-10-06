import numpy as np

from ds_utils import box_constraint, test_1d_op_1d_inp_poly
from .nlsys_utils import quad_2D_dyn, planar_nl_sys, planar_lti_sys
from smpc.utils import planar_region_gen_and_viz


class ProblemSetup:
    def __init__(self, X, U, n_u, n_d, Bd, Q, R, satisfaction_prob, regions,
                 gp_inputs, gp_input_mask, delta_control_variables, delta_input_mask, tracking_matrix,
                 x_start_limit=None, x_end_limit=None, u_start_limit=None, u_end_limit=None):
        self.n_d = n_d
        self.Bd = Bd
        self.Q = Q
        self.R = R
        self.satisfaction_prob = satisfaction_prob

        self.state_space_poly = X
        self.input_space_poly = U
        self.x_start_limit = self.state_space_poly.lb
        self.x_end_limit = self.state_space_poly.ub
        self.u_start_limit = self.input_space_poly.lb
        self.u_end_limit = self.input_space_poly.ub

        # Overrides: If desired to provide overconservative state and input constraints for the purpose of generating solutions for warmstarting.
        # Only needed if warmstarting naively instead of using something like RRT.
        if x_start_limit is not None:
            self.x_start_limit = x_start_limit
            self.x_end_limit = x_end_limit
            self.u_start_limit = u_start_limit
            self.u_end_limit = u_end_limit

        self.gp_inputs = gp_inputs
        self.gp_input_mask = gp_input_mask
        self.gp_input_subset_lim_lb, self.gp_input_subset_lim_ub = gp_input_mask @ np.vstack([self.x_start_limit, np.zeros((n_u, 1))]),\
                                                                   gp_input_mask @ np.vstack([self.x_end_limit, np.zeros((n_u, 1))])

        self.regions = regions
        self.delta_control_variables = delta_control_variables
        self.delta_input_mask = delta_input_mask
        self.region_subset_lim_lb, self.region_subset_lim_ub = delta_input_mask @ np.vstack([self.x_start_limit, np.zeros((n_u, 1))]),\
                                                               delta_input_mask @ np.vstack([self.x_end_limit, np.zeros((n_u, 1))])

        self.tracking_matrix = tracking_matrix
        self.tracking_subset_lim_lb, self.tracking_subset_lim_ub = tracking_matrix @ np.vstack([self.x_start_limit, np.zeros((n_u, 1))]),\
                                                                   tracking_matrix @ np.vstack([self.x_end_limit, np.zeros((n_u, 1))])


    def get_problem_setup_dict(self):
        problem_setup_dict = {}
        problem_setup_dict.update({"n_d": self.n_d, "Bd": self.Bd, "Q": self.Q, "R": self.R})
        problem_setup_dict.update({"satisfaction_prob": self.satisfaction_prob, "regions": self.regions})
        problem_setup_dict.update({"gp_inputs": self.gp_inputs, "gp_input_mask": self.gp_input_mask,
                                   "delta_control_variables": self.delta_control_variables, "delta_input_mask": self.delta_input_mask,
                                   "tracking_matrix": self.tracking_matrix})
        problem_setup_dict.update({"gpinp_subset_lim_lb": self.gp_input_subset_lim_lb, "gpinp_subset_lim_ub": self.gp_input_subset_lim_ub,
                                   "region_subset_lim_lb": self.region_subset_lim_lb, "region_subset_lim_ub": self.region_subset_lim_ub,
                                   "tracking_subset_lim_lb": self.tracking_subset_lim_lb, "tracking_subset_lim_ub": self.tracking_subset_lim_ub})
        problem_setup_dict.update({"x_start_limit": self.x_start_limit, "x_end_limit": self.x_end_limit,
                                   "u_start_limit": self.u_start_limit, "u_end_limit": self.u_end_limit})

        return problem_setup_dict


def quad_2d_sys_1d_inp_res(velocity_limit_override=0.3, region_viz=False,
                           sampling_time=20 * 10e-3, x_threshold=4, z_threshold=4, x_min_threshold=0,
                           x0_delim=1, x1_delim=2, ret_inst=False):
    # State, input, residual dimensions. LQR Cost matrices, state and input constraint sets

    quad_2d_inst = quad_2D_dyn(velocity_limit_override=velocity_limit_override,
                               dt=sampling_time, x_threshold=x_threshold, z_threshold=z_threshold, x_min_threshold=x_min_threshold)
    dyn_info = quad_2d_inst.get_dyn_info_dict()
    quad_2d_symbolic = quad_2d_inst.dynamics

    Q_test = np.diag(np.ones(quad_2d_inst.nx)*1)
    # Tracking in x and z assigned high priority
    Q_test[0, 0] = 5
    Q_test[2, 2] = 5
    R_test = np.diag(np.ones(quad_2d_inst.nu)*0.1)

    # u_start_limit, u_end_limit = U_test.lb, U_test.ub
    # s_start_limit, s_end_limit = X_test.lb, X_test.ub
    # x_init = np.array([[0, 0, 0, 0, 0, 0]]).T

    # GP input (Bd) matrix. Delta segregation (B_\delta) matrix. Tracking matrix to limit tracking cost to subset of state vars.
    gp_inputs = 'state_input'
    # 2-D state 1-D input. Gp input is z_dot (aim is to see performance issues with baseline on vertical stretch)
    # gp_input_mask = np.array([[0, 0, 0, 1, 0, 0, 0, 0]])
    gp_input_mask = np.array([[0, 0, 0, 0, 0, 1, 0, 0]])

    delta_control_variables = 'state_input'
    # x and z control the regions
    delta_input_mask = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]])
    # x and z are to be tracked only. Other states default to 0 track.
    tracking_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]])
    # Residual learns error in x dynamics
    Bd_test = np.array([[1, 0, 0, 0, 0, 0]]).T
    # Bd_test = np.array([[0, 0, 1, 0, 0, 0]]).T

    n_d = 1
    # During DS creation we need to be able to extract the ranges/limits for only relevant variables not necessarily entire space/input constraints.
    # Limiting delta consideration only to those variables activated by the delta_input_mask instead of the entire joint state-input vector
    region_subset_lim_lb, region_subset_lim_ub = delta_input_mask @ np.vstack([dyn_info["X"].lb, np.zeros((dyn_info["n_u"], 1))]),\
                                                 delta_input_mask @ np.vstack([dyn_info["X"].ub, np.zeros((dyn_info["n_u"], 1))])
    # x0_delim, x1_delim = 0.071425, 0.14285
    # print(x0_delim, x1_delim)
    # print(region_subset_lim_lb, region_subset_lim_ub)
    regions = planar_region_gen_and_viz(viz=region_viz, s_start_limit=region_subset_lim_lb, s_end_limit=region_subset_lim_ub,
                                        x0_delim=x0_delim, x1_delim=x1_delim)
    # Input sample generation in GP_DS is limited only to ranges of relevant variables.
    # gpinp_subset_lb, gpinp_subset_ub = gp_input_mask @ np.vstack([dyn_info["X_test"].lb, np.zeros((dyn_info["n_u"], 1))]),\
    #                                    gp_input_mask @ np.vstack([dyn_info["X_test"].ub, np.zeros((dyn_info["n_u"], 1))])

    # Probability with which state and input constraints are to be satisfied.
    satisfaction_prob = 0.85  # Note this will get an override by the fn_config_dict in the examples.

    problem_setup_inst = ProblemSetup(X=dyn_info["X"], U=dyn_info["U"], n_u=dyn_info["n_u"],
                                      n_d=n_d, Bd=Bd_test, Q=Q_test, R=R_test, satisfaction_prob=satisfaction_prob,
                                      regions=regions, gp_inputs=gp_inputs, gp_input_mask=gp_input_mask,
                                      delta_control_variables=delta_control_variables, delta_input_mask=delta_input_mask,
                                      tracking_matrix=tracking_matrix
                                      )

    problem_setup_dict = {}
    problem_setup_dict.update(dyn_info)
    problem_setup_dict.update(problem_setup_inst.get_problem_setup_dict())
    problem_setup_dict.update({"x0_delim": x0_delim, "x1_delim": x1_delim})

    if ret_inst:
        return problem_setup_dict, quad_2d_inst
    else:
        return problem_setup_dict


def planar_lti_1d_inp_res(u_max=5, region_viz=False, sampling_time=20 * 10e-3, x1_threshold=4, x2_threshold=4, x1_min_threshold=0,
                          x0_delim=1, x1_delim=2, ret_inst=False):
    # State, input, residual dimensions. LQR Cost matrices, state and input constraint sets
    planar_lti_inst = planar_lti_sys(dt=sampling_time, x1_threshold=x1_threshold, x2_threshold=x2_threshold,
                                     x1_min_threshold=x1_min_threshold, u_max=u_max)
    dyn_info = planar_lti_inst.get_dyn_info_dict()
    planar_lti_symbolic = planar_lti_inst.dynamics

    Q_test = np.diag(np.ones(planar_lti_inst.nx)) * 5
    R_test = np.diag(np.ones(planar_lti_inst.nu) * 0.01)

    # GP input (Bd) matrix. Delta segregation (B_\delta) matrix. Tracking matrix to limit tracking cost to subset of state vars.
    gp_inputs = 'state_input'
    # 2-D state 1-D input. Gp input is u1 (Don't want to set gp input to be the same as variables controlling the deltas)
    gp_input_mask = np.array([[0, 1, 0, 0]])
    delta_control_variables = 'state_input'
    # x1 and x2 control the regions
    delta_input_mask = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    # x1 and x2 are to be tracked only. Other states default to 0 track.
    tracking_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    # Residual learns error in x2 dynamics
    Bd_test = np.array([[1, 0]]).T
    print(gp_input_mask)
    print(Bd_test)

    n_d = 1
    # During DS creation we need to be able to extract the ranges/limits for only relevant variables not necessarily entire space/input constraints.
    # Limiting delta consideration only to those variables activated by the delta_input_mask instead of the entire joint state-input vector
    region_subset_lim_lb, region_subset_lim_ub = delta_input_mask @ np.vstack([dyn_info["X"].lb, np.zeros((dyn_info["n_u"], 1))]),\
                                                 delta_input_mask @ np.vstack([dyn_info["X"].ub, np.zeros((dyn_info["n_u"], 1))])
    regions = planar_region_gen_and_viz(viz=region_viz, s_start_limit=region_subset_lim_lb, s_end_limit=region_subset_lim_ub,
                                        x0_delim=x0_delim, x1_delim=x1_delim)

    # Probability with which state and input constraints are to be satisfied.
    satisfaction_prob = 0.85  # Note this will get an override by the fn_config_dict in the examples.

    problem_setup_inst = ProblemSetup(X=dyn_info["X"], U=dyn_info["U"], n_u=dyn_info["n_u"],
                                      n_d=n_d, Bd=Bd_test, Q=Q_test, R=R_test, satisfaction_prob=satisfaction_prob,
                                      regions=regions, gp_inputs=gp_inputs, gp_input_mask=gp_input_mask,
                                      delta_control_variables=delta_control_variables, delta_input_mask=delta_input_mask,
                                      tracking_matrix=tracking_matrix
                                      )

    problem_setup_dict = {}
    problem_setup_dict.update(dyn_info)
    problem_setup_dict.update(problem_setup_inst.get_problem_setup_dict())
    problem_setup_dict.update({"x0_delim": x0_delim, "x1_delim": x1_delim})

    if ret_inst:
        return problem_setup_dict, planar_lti_inst
    else:
        return problem_setup_dict


def planar_sys_1d_inp_res(velocity_limit_override=0.3, region_viz=False):
    # State, input, residual dimensions. LQR Cost matrices, state and input constraint sets

    planar_sys_inst = planar_nl_sys(velocity_limit_override=velocity_limit_override)
    dyn_info = planar_sys_inst.get_dyn_info_dict()
    planar_symbolic = planar_sys_inst.dynamics

    Q_test = np.diag(np.ones(planar_sys_inst.nx)*5)
    R_test = np.diag(np.ones(planar_sys_inst.nu)*1)

    # u_start_limit, u_end_limit = U_test.lb, U_test.ub
    # s_start_limit, s_end_limit = X_test.lb, X_test.ub
    # x_init = np.array([[0, 0, 0, 0, 0, 0]]).T

    # GP input (Bd) matrix. Delta segregation (B_\delta) matrix. Tracking matrix to limit tracking cost to subset of state vars.
    gp_inputs = 'state_input'
    # 2-D state 1-D input. Gp input is x1
    gp_input_mask = np.array([[1, 0, 0, 0]])
    delta_control_variables = 'state_input'
    # x1, x2 control the regions
    delta_input_mask = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    # x1 and x2 are to be tracked.
    tracking_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    # Residual learns error in x2 dynamics
    Bd_test = np.array([[0, 1]]).T

    n_d = 1
    # During DS creation we need to be able to extract the ranges/limits for only relevant variables not necessarily entire space/input constraints.
    # Limiting delta consideration only to those variables activated by the delta_input_mask instead of the entire joint state-input vector
    region_subset_lim_lb, region_subset_lim_ub = delta_input_mask @ np.vstack([dyn_info["X"].lb, np.zeros((dyn_info["n_u"], 1))]),\
                                                 delta_input_mask @ np.vstack([dyn_info["X"].ub, np.zeros((dyn_info["n_u"], 1))])
    x0_delim, x1_delim = 2, 4
    # print(region_subset_lim_lb, region_subset_lim_ub)
    regions = planar_region_gen_and_viz(viz=region_viz, s_start_limit=region_subset_lim_lb, s_end_limit=region_subset_lim_ub,
                                        x0_delim=x0_delim, x1_delim=x1_delim)
    # Input sample generation in GP_DS is limited only to ranges of relevant variables.
    # gpinp_subset_lb, gpinp_subset_ub = gp_input_mask @ np.vstack([dyn_info["X_test"].lb, np.zeros((dyn_info["n_u"], 1))]),\
    #                                    gp_input_mask @ np.vstack([dyn_info["X_test"].ub, np.zeros((dyn_info["n_u"], 1))])

    # Probability with which state and input constraints are to be satisfied.
    satisfaction_prob = 0.85  # Note this will get an override by the fn_config_dict in the examples.

    problem_setup_inst = ProblemSetup(X=dyn_info["X"], U=dyn_info["U"], n_u=dyn_info["n_u"],
                                      n_d=n_d, Bd=Bd_test, Q=Q_test, R=R_test, satisfaction_prob=satisfaction_prob,
                                      regions=regions, gp_inputs=gp_inputs, gp_input_mask=gp_input_mask,
                                      delta_control_variables=delta_control_variables, delta_input_mask=delta_input_mask,
                                      tracking_matrix=tracking_matrix
                                      )

    problem_setup_dict = {}
    problem_setup_dict.update(dyn_info)
    problem_setup_dict.update(problem_setup_inst.get_problem_setup_dict())
    problem_setup_dict.update({"x0_delim": x0_delim, "x1_delim": x1_delim})

    return problem_setup_dict


def test_ds_quad_2d(poly_coeffs):
    regions = quad_2d_sys_1d_inp_res()["regions"]
    s_start_limit, s_end_limit = quad_2d_sys_1d_inp_res()["s_start_limit"], quad_2d_sys_1d_inp_res()["s_end_limit"]
    gp_input_mask = quad_2d_sys_1d_inp_res()["input_mask"]
    delta_input_mask = quad_2d_sys_1d_inp_res()["delta_input_mask"]
    n_u = quad_2d_sys_1d_inp_res()["n_u"]
    num_samples = 1000
    viz = True
    noise_std_devs = [0.05, 0.02, 0.03]
    test_1d_op_1d_inp_poly(regions=regions, poly_coeffs=poly_coeffs,
                           start_limit=gp_input_mask @ s_start_limit, end_limit=gp_input_mask @ s_end_limit,
                           gp_input_mask=gp_input_mask, delta_input_mask=delta_input_mask, n_u=n_u,
                           num_points=num_samples,
                           noise_vars=[noise_std_dev ** 2 for noise_std_dev in noise_std_devs],
                           no_viz=not viz, fineness_param=(10, 10))

# def quad_2d_sys_1d_inp_res(ds_construct=False, region_viz=False, velocity_limit_override=0.3):
#     # State, input, residual dimensions. LQR Cost matrices, state and input constraint sets
#     n_x, n_u, n_d = 6, 2, 1
#     quad_2d_inst = quad_2D_dyn(velocity_limit_override=velocity_limit_override)
#     quad_2d_symbolic, U_test, X_test = quad_2d_inst.symbolic, quad_2d_inst.action_space, quad_2d_inst.state_space
#     Q_test = np.diag(np.ones(quad_2d_symbolic.nx)*1)
#     # Tracking in x and z assigned high priority
#     Q_test[0, 0] = 50
#     Q_test[2, 2] = 50
#     R_test = np.diag(np.ones(quad_2d_symbolic.nu)*0.1)
#     u_start_limit, u_end_limit = U_test.lb, U_test.ub
#     s_start_limit, s_end_limit = X_test.lb, X_test.ub
#     x_init = np.array([[0, 0, 0, 0, 0, 0]]).T
#
#     # GP input (Bd) matrix. Delta segregation (B_\delta) matrix. Tracking matrix to limit tracking cost to subset of state vars.
#     gp_inputs = 'state_input'
#     # 2-D state 1-D input. Gp input is z_dot (aim is to see performance issues with baseline on vertical stretch)
#     gp_input_mask = np.array([[0, 0, 0, 1, 0, 0, 0, 0]])
#     delta_control_variables = 'state_input'
#     # x and z control the regions
#     delta_input_mask = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]])
#     # x and z are to be tracked only. Other states default to 0 track.
#     tracking_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]])
#     # Residual learns error in x dynamics
#     Bd_test = np.array([[1, 0, 0, 0, 0, 0]]).T
#
#     zdot_delim = 0
#     # Limiting delta consideration only to those variables activated by the delta_input_mask instead of the entire joint state-input vector
#     region_subset_lim_lb, region_subset_lim_ub = delta_input_mask @ np.vstack([s_start_limit, np.zeros((n_u, 1))]), delta_input_mask @ np.vstack([s_end_limit, np.zeros((n_u, 1))])
#     regions = [box_constraint(np.array([[-velocity_limit_override]]), np.array([[zdot_delim]])), box_constraint(np.array([[zdot_delim]]), np.array([[velocity_limit_override]]))]
#
#     # Probability with which state and input constraints are to be satisfied.
#     satisfaction_prob = 0.85
#
#     problem_setup_dict = {}
#     problem_setup_dict.update({"sys_dyn": quad_2d_symbolic, "Bd_test": Bd_test, "Q_test": Q_test, "R_test": R_test})
#     problem_setup_dict.update({"n_x": n_x, "n_u": n_u, "n_d": n_d})
#     problem_setup_dict.update({"s_start_limit": s_start_limit, "s_end_limit": s_end_limit,
#                                "u_start_limit": u_start_limit, "u_end_limit": u_end_limit, "x_init": x_init})
#     problem_setup_dict.update({"zdot_delim": zdot_delim, "regions": regions, "X_test": X_test, "U_test": U_test})
#     problem_setup_dict.update({"gp_inputs": gp_inputs, "gp_input_mask": gp_input_mask,
#                                "delta_control_variables": delta_control_variables, "delta_input_mask": delta_input_mask,
#                                "tracking_matrix": tracking_matrix})
#     problem_setup_dict.update({"satisfaction_prob": satisfaction_prob})
#     if ds_construct:
#         problem_setup_dict.update({"delta_subset": True,
#                                    "region_subset_lim_lb": region_subset_lim_lb, "region_subset_lim_ub": region_subset_lim_ub})
#     return problem_setup_dict
