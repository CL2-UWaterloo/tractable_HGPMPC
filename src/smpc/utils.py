import time

import torch
import numpy as np
import casadi as cs
import sys
from io import StringIO
import scipy
import math
from common.box_constraint_utils import box_constraint, box_constraint_direct, combine_box
from common.plotting_utils import generate_fine_grid
from models import GP_Model
import matplotlib.pyplot as plt
from IPython.display import display, Math
import os
import cdd

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


def check_delta_inp_dep(delta_control_variables, delta_input_mask, state_dim, keep_assert=True):
    # Currently only can handle for state based regions. To extend to input based regions we'd need to rig up
    # a nominal MPC and generate inputs for the RRT* generated path to initialize the deltas dependent on input.
    if delta_control_variables == "state_only":
        pass
    else:
        # if control variables specified to be state+input ensure that the columns only have 1's in positions
        # corresponding to the state in the joint state-input vector i.e. assert that we only consider the columns corresponding
        # to the state when computing deltas since we don't have inputs to condition the deltas on.
        u_limd_delta_input_mask = delta_input_mask[:, state_dim:]
        if keep_assert:
            assert np.count_nonzero(u_limd_delta_input_mask) == 0, "Can't have input dependence on deltas at the moment"
        else:
            if np.count_nonzero(u_limd_delta_input_mask) != 0:
                return False
    return True


# Ref: https://stackoverflow.com/questions/53561897/external-library-uses-print-instead-of-return
class redirected_stdout:
    def __init__(self):
        self._stdout = None
        self._string_io = None

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._string_io = StringIO()
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout

    @property
    def string(self):
        return self._string_io.getvalue()


# Taken from https://gist.github.com/KMChris/8fd878826453c3d55814b3293c7b084c
def np2bmatrix(arrays, return_list=False):
    matrices = ''
    if return_list:
        matrices = []
    for array in arrays:
        matrix = ''
        temp_arr = np.round(array, 5)
        for row in temp_arr:
            try:
                for number in row:
                    matrix += f'{number}&'
            except TypeError:
                matrix += f'{row}&'
            matrix = matrix[:-1] + r'\\'
        if not return_list:
            matrices += r'\begin{bmatrix}'+matrix+r'\end{bmatrix}'
        else:
            matrices.append(r'\begin{bmatrix}'+matrix+r'\end{bmatrix}')
    return matrices


class FeasibleTraj:
    def __init__(self, U: box_constraint, GP_MPC_inst, verbose=False, max_iter=5):
        """
        Parameters
        ----------
        x_init initial starting state assumed to be known with certainty
        U input constraint set
        N MPC O.L. horizon
        GP_MPC_inst: GP_MPC_Global instance with appropriate system matrices passed during instantiations

        Returns
        -------
        A trajectory that obeys the system dynamics
        """
        self.A, self.B, self.Bd, self.gp_fns, self.gp_input_type, self.input_mask, self.N = GP_MPC_inst.get_info_for_traj_gen()
        self.verbose = verbose
        self.max_iter = max_iter
        self.U = U

    def get_traj(self, x_init):
        x_traj = np.zeros(shape=(self.A.shape[0], self.N+1))
        x_traj[:, [0]] = x_init
        u_traj = self.U.get_random_vectors(num_samples=self.N)
        gp_inputs = np.zeros(shape=(np.linalg.matrix_rank(self.input_mask), self.N))
        for i in range(self.N):
            # Note: Even though vertcat is used for gp_input in the state+input case, this is converted to a horizontal vector
            # by Casadi (check the get_sparsity_in specification for the GPR_Callback) since the input vector for inference is
            # expected to be of shape [num_samples, input_dim] as the GP training inputs are expected to be of the same shape.
            if self.gp_input_type == "state_only":
                gp_inputs[:, [i]] = self.input_mask @ x_traj[:, [i]]
            else:
                gp_inputs[:, [i]] = (self.input_mask @ np.vstack([x_traj[:, [i]], u_traj[:, [i]]]))
            if self.verbose and i == 0:
                print(x_traj[:, [i]], "\n", u_traj[:, [i]], "\n", gp_inputs[:, [i]])
                print(self.A, x_traj[:, [i]], self.B, u_traj[:, [i]], self.Bd, self.gp_fns(gp_inputs[:, [i]]), sep="\n")
            x_traj[:, [i+1]] = self.A @ x_traj[:, [i]] + self.B @ u_traj[:, [i]] + self.Bd @ self.gp_fns(gp_inputs[:, [i]])[0]

        return x_traj, u_traj


class FeasibleTraj_Piecewise:
    def __init__(self, U: box_constraint, GP_MPC_inst, verbose=False, max_iter=5):
        """
        Parameters
        ----------
        x_init initial starting state assumed to be known with certainty
        U input constraint set
        N MPC O.L. horizon
        GP_MPC_inst: GP_MPC_Global instance with appropriate system matrices passed during instantiations

        Returns
        -------
        A trajectory that obeys the system dynamics
        """
        self.A, self.B, self.Bd, self.gp_fns, self.gp_input_type, self.input_mask, self.N = GP_MPC_inst.get_info_for_traj_gen()
        self.regions = GP_MPC_inst.regions
        self.delta_control_variables, self.delta_input_mask = GP_MPC_inst.delta_control_variables, GP_MPC_inst.delta_input_mask
        self.gp_input_type, self.gp_input_mask = GP_MPC_inst.gp_inputs, GP_MPC_inst.input_mask
        self.get_mu_d = GP_MPC_inst.get_mu_d
        self.res_dim = GP_MPC_inst.gp_fns.output_dims
        self.get_Sigma_d = GP_MPC_inst.get_Sigma_d
        self.computesigma_wrapped = GP_MPC_inst.computesigma_wrapped
        self.verbose = verbose
        self.K = GP_MPC_inst.K
        self.affine_transform = GP_MPC_inst.affine_transform
        self.max_iter = max_iter
        self.delta_constraint_obj = GP_MPC_inst.delta_constraint_obj
        self.U = U
        self.X = GP_MPC_inst.X

    def get_traj(self, x_init):
        n_x = self.A.shape[-1]
        n_u = self.B.shape[-1]
        x_traj = np.zeros(shape=(n_x, self.N+1))
        x_traj[:, [0]] = x_init
        u_traj = self.U.get_random_vectors(num_samples=self.N)
        num_delta_inp = np.linalg.matrix_rank(self.delta_input_mask)
        delta_controls = cs.DM.zeros(num_delta_inp, self.N)
        gp_inputs = np.zeros(shape=(np.linalg.matrix_rank(self.input_mask), self.N))
        hld_mat = cs.DM.zeros(len(self.regions), self.N)
        lld_mat = [cs.DM.zeros(2*num_delta_inp, len(self.regions)) for _ in range(self.N)]
        mu_d = np.zeros(shape=(self.res_dim, self.N))
        Sigma_d = [np.zeros((self.res_dim, self.res_dim)) for _ in range(self.N)]
        Sigma_x = [np.zeros((n_x, n_x)) for _ in range(self.N+1)]
        Sigma_u = [np.zeros((n_u, n_u)) for _ in range(self.N)]
        Sigma = []
        for i in range(self.N):
            if self.verbose:
                print("Timestep: %s" % i)

            # Generating hld vector and lld array elements
            joint_vec = np.vstack([x_traj[:, [i]], u_traj[:, [i]]])
            control_vec = x_traj[:, [i]] if self.delta_control_variables == "state_only" else joint_vec
            delta_controls[:, [i]] = self.delta_input_mask @ control_vec
            if self.verbose:
                display(Math(r'\delta_{} = {} = {}'.format('{ctrl, %s}' % i,
                                                           "\,\, \,\,".join(np2bmatrix([self.delta_input_mask, control_vec], return_list=True)),
                                                           np2bmatrix([delta_controls[:, [i]]]))))
            lld_mat[i] = simulate_hld_lld(self.delta_constraint_obj, self.regions, state_dim=3, eps=1e-5,
                                          samples=delta_controls[:, [i]], verbose=False, ret_lld=True, unsqueeze=True)
            for region_idx, region in enumerate(self.regions):
                hld_mat[region_idx, i] = 1 if region.check_satisfaction(delta_controls[:, i].T).item() is True else 0
            if self.verbose:
                print("LLD Mat")
                display(Math(r'\delta_{} = {}'.format("{:, :, %s}" % i, np2bmatrix([lld_mat[i]]))))
                print("HLD Row")
                display(Math(r'\delta_{} = {}'.format("{:, %s}" % i, np2bmatrix([hld_mat[:, [i]]]))))

            # Getting output means and covs from piecewise gp class for all regions.
            gp_vec = x_traj[:, [i]] if self.gp_input_type == "state_only" else joint_vec
            gp_inputs[:, [i]] = self.input_mask @ gp_vec
            hybrid_means, *hybrid_covs = self.gp_fns(gp_inputs[:, i])
            if self.verbose:
                display(Math(r'g_{} = {} = {}'.format('{inp, %s}' % i,
                                                      "\,\, \,\,".join(np2bmatrix([self.gp_input_mask, gp_vec], return_list=True)),
                                                      np2bmatrix([gp_inputs[:, [i]]]))))
                print("Region-wise Means")
                display(Math(r'{}'.format(np2bmatrix([hybrid_means]))))
                print("Region-wise Covariances")
                for region_idx in range(len(self.regions)):
                    display(Math(r'Region {}'.format(region_idx+1, np2bmatrix([hybrid_covs[region_idx]]))))


            # Applying deltas to select correct mean and cov.
            mu_d[:, [i]] = self.get_mu_d(hybrid_means, hld_mat[:, [i]])
            Sigma_d[i] = self.get_Sigma_d(hld_mat[:, [i]], *hybrid_covs)
            if self.verbose:
                print("Selected Mean")
                display(Math(r'\mu^d_{} = {}'.format(i, np2bmatrix([mu_d[:, [i]]]))))
                print("Selected Cov")
                display(Math(r'\Sigma^d_{} = {}'.format(i, np2bmatrix([Sigma_d[i]]))))
            if self.verbose and i == 0:
                print(x_traj[:, [i]], "\n", u_traj[:, [i]], "\n", gp_inputs[:, [i]])
                print(self.A, x_traj[:, [i]], self.B, u_traj[:, [i]], self.Bd, self.gp_fns(gp_inputs[:, [i]]), sep="\n")


            # Final dynamics equations for x and Sigma x.
            # Dynamics for x
            x_traj[:, [i+1]] = self.A @ x_traj[:, [i]] + self.B @ u_traj[:, [i]] + self.Bd @ mu_d[:, [i]]
            # Dynamics for Sigma x
            Sigma_u[i] = self.K @ Sigma_x[i] @ self.K.T
            Sigma_i = self.computesigma_wrapped(Sigma_x[i], Sigma_u[i], Sigma_d[i])
            Sigma.append(Sigma_i)
            Sigma_x[i+1] = self.affine_transform @ Sigma_i @ self.affine_transform.T

        return x_traj, u_traj, Sigma_x, hld_mat, lld_mat


def get_user_attributes(cls, exclude_methods=True):
    base_attrs = dir(type('dummy', (object,), {}))
    this_cls_attrs = dir(cls)
    res = []
    for attr in this_cls_attrs:
        if base_attrs.count(attr) or (callable(getattr(cls,attr)) and exclude_methods):
            continue
        res += [attr]
    return res


class Sigma_u_Callback(cs.Callback):
    def __init__(self, name, K, opts={"enable_fd": True}):
        cs.Callback.__init__(self)
        self.K = K
        self.n_u, self.n_x = self.K.shape # K is an mxn matrix since BKx is in R^n, B in R^(nxm), K in R^(mxn) and x in R^n
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return cs.Sparsity.dense(self.n_x, self.n_x)

    def get_sparsity_out(self, i):
        return cs.Sparsity.dense(self.n_u, self.n_u)

    def eval(self, arg):
        # https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/gp_mpc.py#L308
        # Sigma_u = self.K.T @ arg[0] @ self.K
        Sigma_u = self.K @ arg[0] @ self.K.T
        return [Sigma_u]


class Sigma_x_dynamics_Callback_LTI(cs.Callback):
    def __init__(self, name, affine_transform, n_in, n_x, opts={"enable_fd": True}):
        cs.Callback.__init__(self)
        self.affine_transform = affine_transform
        self.n_in = n_in
        self.n_x = n_x
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return cs.Sparsity.dense(self.n_in, self.n_in)

    def get_sparsity_out(self, i):
        return cs.Sparsity.dense(self.n_x, self.n_x)

    def eval(self, arg):
        Sigma_k = arg[0]
        Sigma_x = self.affine_transform @ Sigma_k @ self.affine_transform.T
        return [Sigma_x]


class GPR_Callback(cs.Callback):
    def __init__(self, name, likelihood_fn, model, state_dim=1, output_dim=None, opts={}):
        """
        Parameters
        ----------
        name Name is necessary for Casadi initialization using construct.
        likelihood_fn
        model
        state_dim size of the input dimension.
        opts
        """
        cs.Callback.__init__(self)
        self.likelihood = likelihood_fn
        self.model = model
        self.input_dims = state_dim
        self.output_dims = output_dim
        if self.output_dims is None:
            self.output_dims = self.input_dims
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 2

    def get_num_samples(self):
        return self.model.train_x.shape[0]

    def __len__(self): return 1

    def get_sparsity_in(self, i):
        # If this isn't specified then it doesn't accept the full input vector and only limits itself to the first element.
        # return cs.Sparsity.dense(self.state_dim, 1)
        return cs.Sparsity.dense(1, self.input_dims)

    def eval(self, arg):
        # likelihood will return a mean and variance but out differentiator only needs the mean
        # print(arg[0])
        mean, cov = self.postproc(self.likelihood(self.model(self.preproc(arg[0]))))
        return [mean, cov]

    @staticmethod
    def preproc(inp):
        return torch.from_numpy(np.array(inp).astype(np.float32))

    @staticmethod
    def postproc(op):
        # print(get_user_attributes(op))
        return op.mean.detach().numpy(), op.covariance_matrix.detach().numpy()


class Test_2D_Callback(cs.Callback):
    def __init__(self, name, state_dim=2, opts={}, cov_input=0.3, linear=False):
        cs.Callback.__init__(self)
        self.state_dim = state_dim
        self.cov_input = cov_input
        self.linear = linear
        self.create_cov_mat()
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 2

    def __len__(self): return 2

    def create_cov_mat(self):
        cov = np.eye(self.state_dim) * self.cov_input
        valid_cov = False
        # Adding some randomness to anti-diagonal elements of cov mat.
        while not valid_cov:
            rand_cov_term = np.random.rand() * self.cov_input
            cov[0, 1] = rand_cov_term
            cov[1, 0] = rand_cov_term
            if (np.array(np.linalg.eig(cov)[0]) >= 0).all():
                valid_cov = True
        self.cov = cov

    def get_num_samples(self):
        return "Function is continuous for testing."

    def get_sparsity_in(self, i):
        # If this isn't specified then it doesn't accept the full input vector and only limits itself to the first element.
        return cs.Sparsity.dense(1, self.state_dim)

    def get_sparsity_out(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.state_dim, 1)
        elif i == 1:
            return cs.Sparsity.dense(self.state_dim, self.state_dim)

    def eval(self, arg):
        # mean = np.vstack(np.sin(arg[0][0, :]), np.cos(arg[0][1, :]))
        mean = [0, 0]
        if self.linear:
            mean[0] = 3*arg[0][:, 0] + 2*arg[0][:, 1]
            mean[1] = 2*arg[0][:, 0] + 5*arg[0][:, 1]
        else:
            mean[0] = 3*np.power(arg[0][:, 0], 2) + 2*np.power(arg[0][:, 1], 2)
            mean[1] = 2*np.power(arg[0][:, 0], 2) + 5*np.power(arg[0][:, 1], 2)
        mean = np.vstack(mean)
        return [mean, self.cov]


class Piecewise_Test_1D_Callback(cs.Callback):
    def __init__(self, name, state_dim=1, num_regions=2, cov_input=(0.3, 0.2), linear=False, opts={}):
        assert num_regions == len(cov_input), "Number of covariance must be equal to number of regions."
        cs.Callback.__init__(self)
        self.state_dim = state_dim
        self.cov_input = cov_input
        self.num_regions = num_regions
        self.linear = linear
        self.cov = cov_input
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1+self.num_regions

    def get_num_samples(self):
        return "Function is continuous for testing."

    def get_sparsity_in(self, i):
        # If this isn't specified then it doesn't accept the full input vector and only limits itself to the first element.
        return cs.Sparsity.dense(1, 1)

    def get_sparsity_out(self, i):
        if i == 0:
            return cs.Sparsity.dense(1, self.num_regions)
        elif i >= 1:
            return cs.Sparsity.dense(1, 1)

    def eval(self, arg):
        mean = [0, 0]
        if self.linear:
            # Region 1 dynamics
            mean[0] = 0.3*arg[0][:, 0]
            # Region 2 dynamics
            mean[1] = 2*arg[0][:, 0]
        else:
            mean[0] = 3*np.power(arg[0][0, 0], 2)
            mean[1] = 2*np.power(arg[0][0, 0], 2) + 4*arg[0][0, 0]
        mean = cs.horzcat(*mean)
        return [mean, *self.cov]


def setup_terminal_costs(A, B, Q, R):
    Q_lqr = Q
    R_lqr = R
    from scipy.linalg import solve_discrete_are
    P = solve_discrete_are(A, B, Q_lqr, R_lqr)
    btp = np.dot(B.T, P)
    K = -np.dot(np.linalg.inv(R + np.dot(btp, B)), np.dot(btp, A))
    return K, P


def get_inv_cdf(n_i, satisfaction_prob, boole_deno_override=False):
    # \overline{p} from the paper
    if not boole_deno_override:
        p_bar_i = 1 - (1 / n_i - (satisfaction_prob + 1) / (2 * n_i))
    else:
        p_bar_i = 1 - ((1 - satisfaction_prob) / 2)
    # \phi^-1(\overline{p})
    inverse_cdf_i = scipy.stats.norm.ppf(p_bar_i)
    return inverse_cdf_i


class Piecewise_GPR_Callback(GPR_Callback):
    def __init__(self, name, likelihood_fns, models, output_dim, input_dim, num_regions, opts={}):
        """

        Parameters
        ----------
        name
        likelihood_fns, models: For the multidim piecewise case the ordering would be all models belonging to one state dim
        first before moving to the next. Ex: for the 2-D case with 4 regions we have 8 models [model_0 ... model_7] where
        model_0-model_3 correspond to the 4 piecewise models for the first state dim and model_4-model_7 correspond to those for the
        second state dim
        output_dim: Dimension of the GP output. This must match with the number of models passed obeying the formula
        output_dim*num_regions = len(models)
        input_dim: Dimension of the GP input. Must match with the input dimension of each GP.
        num_regions: number of regions in the piecewise/hybrid model
        opts

        Returns
        Note that this function only returns a (horizontal) concatenation of the means output from each GP. The
        application of delta to select the right mean is left to a Casadi function defined in the piecewise MPC class.
        """
        cs.Callback.__init__(self)
        assert output_dim*num_regions == len(models), "The models must have length = output_dim*num_regions = %s but got %s instead" %\
                                                      (output_dim*num_regions, len(models))
        for model in models:
            assert input_dim == model.train_x.shape[-1], "The value of input_dim must match the number of columns of the model's train_x set. input_dim: %s, train_x_shape: %s" % (input_dim, model.train_x.shape)
        self.likelihoods = likelihood_fns
        self.models = models
        self.output_dims = output_dim
        self.input_dims = input_dim
        self.num_models = len(self.models)
        self.num_regions = num_regions
        self.construct(name, opts)
        self.organize_models()

    def get_n_in(self): return 1
    # Can't return covariances as list or 3-D tensor-like array. Instead return all cov matrices separately. There are
    # num_regions number of cov matrices to return.
    def get_n_out(self): return 1+self.num_regions

    def get_num_samples(self):
        # Only need to sum over 1 dimension of the output since the samples must be summed regionwise.
        return np.sum([self.models[idx].train_x.shape[0] for idx in range(len(self.models)//self.output_dims)])

    def get_sparsity_out(self, i):
        if i == 0:
            # Output mean shape is (self.output_dims, 1) and 1 for every region stacked horizontally to get the below
            return cs.Sparsity.dense(self.output_dims, self.num_regions)
        else:
            # output residual covariance matrices. One of these for every region for a total of self.num_regions covariance
            # matrices as evidenced by get_n_out
            return cs.Sparsity.dense(self.output_dims, self.output_dims)

    def __len__(self): return self.num_models

    def organize_models(self):
        # self.dimwise_region_models, self.dimwise_region_likelihoods = [[] for _ in range(self.num_regions)], [[] for _ in range(self.num_regions)]
        self.dimwise_region_models, self.dimwise_region_likelihoods = [[] for _ in range(self.output_dims)], [[] for _ in range(self.output_dims)]
        # Partition models per dimension. Ex: For 2-D case with 4 regions dimwise_models[0] has models[0]->models[3]
        # and dimwise_models[1] has models[4]->models[7]
        for output_dim in range(self.output_dims):
            self.dimwise_region_models[output_dim] = self.models[output_dim*(self.num_regions):
                                                                 (output_dim+1)*(self.num_regions)]
            self.dimwise_region_likelihoods[output_dim] = self.likelihoods[output_dim*(self.num_regions):
                                                                           (output_dim+1)*(self.num_regions)]


    def eval(self, arg):
        # Note regarding the covariances.
        # Regionwise_covs is going to be a list of single values corresponding to the covariance output in 1 region
        # of a single output GP. In the same way as the MultidimGPR callback, the covariance outputs from the same region
        # must be stored together in a diag matrix. Thus, the final covs list is going be a list of length = num_regions
        # with each element being a *diagonal* (because of independence across dims assumption in residual terms) matrix
        # of size (n_d, n_d)
        dimwise_means, dimwise_covs = [], []
        for output_dim in range(self.output_dims):
            regionwise_means, regionwise_covs = [], []
            dim_likelihoods, dim_models = self.dimwise_region_likelihoods[output_dim], self.dimwise_region_models[output_dim]
            for likelihood, model in zip(dim_likelihoods, dim_models):
                gp_op = self.postproc(likelihood(model(self.preproc(arg[0]))))
                regionwise_means.append(gp_op[0])
                regionwise_covs.append(gp_op[1])
            # Note the gp output mean and covariances are of the shape (1,) (1,). When calling horzcat on these shapes, casadi ends up
            # up vertstacking them instead. So just use horzcat to generate a vector of shape (num_regions, 1) and then transpose. to
            # get the desired row vector instead of column vector.
            dimwise_means.append(cs.horzcat(regionwise_means).T)
            dimwise_covs.append(regionwise_covs)
        covs = [cs.diag([dim_cov[region_idx] for dim_cov in dimwise_covs]) for region_idx in range(self.num_regions)]
        means = cs.vertcat(*dimwise_means)
        return [means, *covs]


def piecewise_callback_test(models, likelihoods, input_dim, num_regions, inp_vec=np.array([[2.5], [0.1]])):
    output_dim = int(len(models) / num_regions)
    testcallback = Piecewise_GPR_Callback('f',  likelihoods, models, output_dim, input_dim, num_regions,
                                          opts={"enable_fd": True})
    # Because of the way Casadi works, the covariances can't be returned as a list but must be a bunch of DM matrices.
    # Using *covs accumulates all the individual covariance matrices back into 1 list.
    means, *covs = testcallback(inp_vec)
    print("output from callback")
    #
    # '\,\, ,\,\, '.join([np2bmatrix([covs[0]])])
    display(Math(r'\text{}: {} ; \text{}: {}'.format("{region-wise means}", np2bmatrix([means]), "{region-wise covs}",
                                                     ' \,\,,\,\, '.join(np2bmatrix(covs, return_list=True)))))
    print("means and covariances by iteration through models")
    for i in range(len(models)):
        temp_callback = GPR_Callback('g', likelihoods[i], models[i], state_dim=2, opts={"enable_fd": True})
        mean, cov = temp_callback(inp_vec)
        print("idx: %s ; mean: %s ; cov: %s" % (i, mean, cov))


class delta_matrix_mult(cs.Callback):
    def __init__(self, name, n_in_rows, n_in_cols, num_regions, N, opts={}, delta_tol=0, test_softplus=False):
        cs.Callback.__init__(self)
        self.n_in_rows = n_in_rows
        # number of columns in a single matrix corresponding to 1 region. Total columns when they're hstacked will be
        # num_regions * n_in_cols_single
        self.n_in_cols = n_in_cols
        self.num_regions = num_regions
        self.N = N
        self.delta_tol = delta_tol
        self.test_softplus = test_softplus
        self.sharpness_param = 75
        self.construct(name, opts)

    def get_n_in(self):
        # 1 for delta array followed by num_regions number of matrices
        return 1+self.num_regions

    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.num_regions, 1)
        else:
            return cs.Sparsity.dense(self.n_in_rows, self.n_in_cols)

    def get_sparsity_out(self, i):
        return cs.Sparsity.dense(self.n_in_rows, self.n_in_cols)

    def eval(self, arg):
        delta_k, matrix_arr = arg[0], arg[1:]
        matrix_summation = cs.DM.zeros(self.n_in_rows, self.n_in_cols)
        for region_idx in range(len(matrix_arr)):
            if not self.test_softplus:
                delta_vec = delta_k[region_idx, 0]
            else:
                delta_vec = np.log(1+np.exp(self.sharpness_param*delta_k[region_idx, 0]))/self.sharpness_param
            matrix_summation += matrix_arr[region_idx] * (delta_vec + self.delta_tol)
        return [matrix_summation]


class hybrid_res_covar(delta_matrix_mult):
    def __init__(self, name, n_d, num_regions, N, opts={}, delta_tol=0, test_softplus=False):
        super().__init__(name, n_d, n_d, num_regions, N, opts, delta_tol, test_softplus)


class MultidimGPR_Callback(GPR_Callback):
    def __init__(self, name, likelihood_fns, models, state_dim=1, output_dim=None, opts={}):
        cs.Callback.__init__(self)
        self.likelihoods = likelihood_fns
        self.models = models
        self.n_d = len(self.models)
        self.state_dim = state_dim
        self.input_dims = self.state_dim
        self.output_dim = output_dim
        self.output_dims = output_dim
        if self.output_dim is None:
            self.output_dim = self.state_dim
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 2

    def __len__(self): return self.n_d

    def get_num_samples(self):
        # Number of samples is constant across all models for the multidimensional case unlike the piecewise one.
        return self.models[0].train_x.shape[0]

    def get_sparsity_out(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.output_dim, 1)
        elif i == 1:
            return cs.Sparsity.dense(self.output_dim, self.output_dim)

    def eval(self, arg):
        means, covs = [], []
        for likelihood, model in zip(self.likelihoods, self.models):
            gp_op = self.postproc(likelihood(model(self.preproc(arg[0]))))
            means.append(gp_op[0])
            covs.append(gp_op[1])
        # Simplifying assumption that the residuals output in each dimension are independent of others and hence off-diagonal elements are 0.
        return [cs.vertcat(means), cs.diag(covs)]


class Multidim_PiecewiseGPR_Callback(GPR_Callback):
    def __init__(self, name, likelihood_fns, models, state_dim=1, output_dim=None,
                 opts={}, test_softplus=False, add_delta_tol=True):
        cs.Callback.__init__(self)
        self.likelihoods = likelihood_fns
        self.models = models
        self.state_dim = state_dim
        self.n_d = len(self.models) // state_dim
        self.input_dims = self.state_dim
        self.output_dim = output_dim
        self.output_dims = output_dim
        self.test_softplus = test_softplus
        self.add_delta_tol = add_delta_tol
        hybrid_means_sym = cs.MX.sym('hybrid_means_sym', self.n_d, self.num_regions)
        delta_k_sym = cs.MX.sym('delta_k_sym', self.num_regions, 1)
        self.get_mu_d = cs.Function('get_mu_d', [hybrid_means_sym, delta_k_sym],
                                    [(delta_k_sym.T @ hybrid_means_sym.T).T],
                                    {'enable_fd': True})
        self.get_Sigma_d = hybrid_res_covar('hybrid_res_cov', self.n_d, self.num_regions,
                                            self.N, opts={'enable_fd': True},
                                            delta_tol=(1e-2 if self.add_delta_tol else 0), test_softplus=self.test_softplus)

        if self.output_dim is None:
            self.output_dim = self.state_dim
        self.construct(name, opts)

    def get_n_in(self): return 2
    def get_n_out(self): return 2

    def __len__(self): return self.n_d

    def get_num_samples(self):
        # Number of samples is constant across all models for the multidimensional case unlike the piecewise one.
        return self.models[0].train_x.shape[0]

    def get_sparsity_out(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.output_dim, 1)
        elif i == 1:
            return cs.Sparsity.dense(self.output_dim, self.output_dim)

    def eval(self, arg):
        means, covs = [], []
        # mean_vec = self.get_mu_d(hybrid_means, high_level_deltas[:, [k]])
        for likelihood, model in zip(self.likelihoods, self.models):
            gp_op = self.postproc(likelihood(model(self.preproc(arg[0]))))
            means.append(gp_op[0])
            covs.append(gp_op[1])
        # Simplifying assumption that the residuals output in each dimension are independent of others and hence off-diagonal elements are 0.
        return [cs.vertcat(means), cs.diag(covs)]


def test_redundant_constr():
    class throwaway_callback(cs.Callback):
        def __init__(self, name, idx, opts={}):
            cs.Callback.__init__(self)
            name = name+"_"+str(idx)
            self.idx = idx
            self.construct(name, opts)

        def get_n_in(self): return 1
        def get_n_out(self): return 1

        def get_sparsity_out(self, i):
            if i == 0:
                return cs.Sparsity.dense(2, 1)

        def get_sparsity_in(self, i):
            if i == 0:
                return cs.Sparsity.dense(2, 1)
            else:
                return cs.Sparsity.dense(3, 1)

        def eval(self, arg):
            inp = arg[0]
            op = inp*self.idx
            print("Calling throwaway region: %s, o/p: %s" % (self.idx, op))
            return [op]


    class compute_mean(cs.Callback):
        def __init__(self, name, opts={}):
            cs.Callback.__init__(self)
            name = name
            self.construct(name, opts)

        def get_n_in(self): return 2
        def get_n_out(self): return 1

        def get_sparsity_out(self, i):
            if i == 0:
                return cs.Sparsity.dense(2, 1)

        def get_sparsity_in(self, i):
            if i == 0:
                return cs.Sparsity.dense(2, 3)
            else:
                return cs.Sparsity.dense(3, 1)

        def eval(self, arg):
            region_means, delta = arg[0], arg[1]
            print(delta.T, region_means.T, delta.T @ region_means.T)
            op = (delta.T @ region_means.T).T
            print("Calling mean compute, o/p: %s" % (op))
            return [op]

    opti = cs.Opti()
    # x takes the place of the gp input that remains constant across all models.
    x = opti.variable(2, 1)
    # opti.set_initial(x, np.array([[-3, 2]]).T)
    # y takes the place of the gp residual output. Here for simplicity it is assumed that each callback returns the full state vector
    # but ofcourse when working with a multi-dim residual, we will need to handle for each dimension's residual separately.
    y = opti.variable(2, 1)
    delta = opti.parameter(3, 1)

    # Length of the throaway callbacks corresponds to the number of regions
    throwaway_callbacks = [throwaway_callback('f', i, {"enable_fd": True}) for i in range(3)]
    # Stack all the callback mean outputs (given x as input) together in preparation for multiplication with delta vector
    hstacked_means = cs.horzcat(*[throwaway_callback(x) for throwaway_callback in throwaway_callbacks])
    # Just split the computation.
    # opti.subject_to(y - (delta.T @ hstacked_means.T).T == 0)
    # multiplier = cs.MX.zeros(3, 1)
    multiplier = cs.DM.zeros(3, 1)
    # multiplier[2, 0] = 1
    opti.subject_to(y - (multiplier.T @ hstacked_means.T).T == 0)
    # mean_compute_fn = compute_mean('f', {"enable_fd": True})
    # opti.subject_to(y - mean_compute_fn(hstacked_means, delta) == 0)

    # print(opti.debug.g_describe(0))
    # opti.set_value(delta, np.array([[0, 1]]).T)
    # print(opti.debug.g_describe(0))
    # opti.subject_to(y == (np.zeros([1, 2]) @ x.T).T)
    # callback_inst = redundant_callback('f', {'enable_fd': True})
    # selected_mean = callback_inst(delta, x)
    # opti.subject_to(y == selected_mean)
    # print(opti.g)
    # print("Starting callback test")
    # test_eval_fn = cs.Function('f', [opti.x, opti.p], [opti.g])
    # assignment to opti's optimization variables. First 2 values correspond to assignment to x and second 2 values
    # correspond to assignment to y. With the delta assignment shown below (0, 0, 1), the true expected y value is
    # -6, 4 but we pass in -5, 4 to check if the constr viol shows up as expected.
    print(opti.x, opti.g)
    opti_x_assgt = np.array([-3, 2, -6, 4])
    # assignment to deltas
    # opti_p_assgt = np.array([0, 0, 1])
    # opti_p_assgt = cs.DM.zeros(3)
    # opti_p_assgt[-1] = 1
    # Interesting. When using MX.zeros since the zeros are structural it correctly doesn't print the statements within the callbacks.
    # But at the same time, the constr viol output is a function instead of a numeric value.
    opti_p_assgt = cs.DM.zeros(3, 1)
    opti_p_assgt[-1] = 1
    # constr_viol = test_eval_fn(opti_x_assgt, opti_p_assgt)
    # print(constr_viol)
    opti.set_value(delta, opti_p_assgt)
    opti.subject_to(x - np.ones([2, 1])*2 >= 0)

    opti.minimize(np.array([[1, 1]]) @ y)

    opti.solver('ipopt')
    sol = opti.solve()
    print(opti.debug.value(x))
    print(opti.debug.value(y))


def test_redundant_constr_const_delta():
    class throwaway_callback(cs.Callback):
        def __init__(self, name, idx, opts={}):
            cs.Callback.__init__(self)
            name = name+"_"+str(idx)
            self.idx = idx
            self.construct(name, opts)

        def get_n_in(self): return 1
        def get_n_out(self): return 1

        def get_sparsity_out(self, i):
            if i == 0:
                return cs.Sparsity.dense(2, 1)

        def get_sparsity_in(self, i):
            if i == 0:
                return cs.Sparsity.dense(2, 1)
            else:
                return cs.Sparsity.dense(3, 1)

        def eval(self, arg):
            inp = arg[0]
            op = inp*self.idx
            print("Calling throwaway region: %s, o/p: %s" % (self.idx, op))
            return [op]

    opti = cs.Opti()
    # x takes the place of the gp input that remains constant across all models.
    x = opti.variable(2, 1)
    y = opti.variable(2, 1)

    # Length of the throaway callbacks corresponds to the number of regions
    throwaway_callbacks = [throwaway_callback('f', i, {"enable_fd": True}) for i in range(3)]
    # Stack all the callback mean outputs (given x as input) together in preparation for multiplication with delta vector
    hstacked_means = cs.horzcat(*[throwaway_callback(x) for throwaway_callback in throwaway_callbacks])
    # Split the computation so that zeros are respected and unnecessary callbacks are not evaluated
    final_mean = 0
    const_delta = cs.DM.zeros(3)
    const_delta[-2] = 1
    for i in range(3):
        final_mean += const_delta[i] * hstacked_means[:, i]

    opti.subject_to(y - final_mean == 0)
    opti.subject_to(x - np.ones([2, 1])*2 >= 0)

    opti.minimize(np.array([[1, 1]]) @ y)

    opti.solver('ipopt')
    sol = opti.solve()
    print(opti.debug.value(x))
    print(opti.debug.value(y))


def test_redundant_constr_param_delta():
    class throwaway_callback(cs.Callback):
        def __init__(self, name, idx, opts={}):
            cs.Callback.__init__(self)
            name = name+"_"+str(idx)
            self.idx = idx
            self.construct(name, opts)

        def get_n_in(self): return 1
        def get_n_out(self): return 1

        def get_sparsity_out(self, i):
            if i == 0:
                return cs.Sparsity.dense(2, 1)

        def get_sparsity_in(self, i):
            if i == 0:
                return cs.Sparsity.dense(2, 1)
            else:
                return cs.Sparsity.dense(3, 1)

        def eval(self, arg):
            inp = arg[0]
            op = inp*self.idx
            print("Calling throwaway region: %s, o/p: %s" % (self.idx, op))
            return [op]

    opti = cs.Opti()
    # x takes the place of the gp input that remains constant across all models.
    x = opti.variable(2, 1)
    y = opti.variable(2, 1)
    delta = opti.parameter(3)

    # Length of the throaway callbacks corresponds to the number of regions
    throwaway_callbacks = [throwaway_callback('f', i, {"enable_fd": True}) for i in range(3)]
    # Stack all the callback mean outputs (given x as input) together in preparation for multiplication with delta vector
    hstacked_means = cs.horzcat(*[throwaway_callback(x) for throwaway_callback in throwaway_callbacks])
    # Split the computation to try to see that zeros are respected and unnecessary callbacks are not evaluated
    final_mean = 0
    for i in range(3):
        final_mean += delta[i] * hstacked_means[:, i]
    opti.subject_to(y - final_mean == 0)

    # assignment to deltas
    opti_p_assgt = cs.DM([0, 0, 1])
    opti.set_value(delta, opti_p_assgt)
    print(opti.debug.value(delta))
    opti.subject_to(x - np.ones([2, 1])*2 >= 0)

    opti.minimize(np.array([[1, 1]]) @ y)

    opti.solver('ipopt')
    sol = opti.solve()
    print(opti.debug.value(x))
    print(opti.debug.value(y))


class computeSigmaglobal_meaneq(cs.Callback):
    # flag for taylor approx
    def __init__(self, name, feedback_mat, residual_dim, opts={}):
        cs.Callback.__init__(self)
        self.K = feedback_mat
        self.n_u, self.n_x = self.K.shape
        self.n_d = residual_dim
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 3
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.n_x, self.n_x)
        elif i == 1:
            return cs.Sparsity.dense(self.n_u, self.n_u)
        elif i == 2:
            return cs.Sparsity.dense(self.n_d, self.n_d)

    # This method is needed to specify Sigma's output shape matrix (as seen from casadi's callback.py example). Without it,
    # the symbolic output Sigma is treated as (1, 1) instead of (mat_dim, mat_dim)
    def get_sparsity_out(self, i):
        # Forward sensitivity
        mat_dim = self.n_x+self.n_u+self.n_d
        return cs.Sparsity.dense(mat_dim, mat_dim)

    def eval(self, arg):
        Sigma_x, Sigma_u, Sigma_d = arg[0], arg[1], arg[2]
        # print(Sigma_x.shape, Sigma_u.shape, Sigma_d.shape)
        assert Sigma_d.shape == (self.n_d, self.n_d), "Shape of Sigma_d must match with n_d value specified when creating instance"
        # https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/gp_mpc.py#L310
        Sigma_xu = Sigma_x @ self.K.T
        # Sigma_xu = Sigma_x @ self.K
        # Sigma_zd is specific to mean equivalence
        Sigma_xd = np.zeros((self.n_x, self.n_d))
        Sigma_ud = np.zeros((self.n_u, self.n_d))

        Sigma_z = cs.vertcat(cs.horzcat(Sigma_x, Sigma_xu),
                             cs.horzcat(Sigma_xu.T, Sigma_u)
                            )

        Sigma = cs.vertcat(cs.horzcat(Sigma_x, Sigma_xu, Sigma_xd),
                           cs.horzcat(Sigma_xu.T, Sigma_u, Sigma_ud),
                           cs.horzcat(Sigma_xd.T, Sigma_ud.T, Sigma_d))

        return [Sigma]


class computeSigma_nofeedback(cs.Callback):
    # flag for taylor approx
    def __init__(self, name, n_x, n_u, n_d, opts={}):
        cs.Callback.__init__(self)
        self.n_x, self.n_u = n_x, n_u
        self.n_d = n_d
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.n_x, self.n_x)
        elif i == 1:
            return cs.Sparsity.dense(self.n_d, self.n_d)

    # This method is needed to specify Sigma's output shape matrix (as seen from casadi's callback.py example). Without it,
    # the symbolic output Sigma is treated as (1, 1) instead of (mat_dim, mat_dim)
    def get_sparsity_out(self, i):
        # Forward sensitivity
        mat_dim = self.n_x+self.n_u+self.n_d
        return cs.Sparsity.dense(mat_dim, mat_dim)

    def eval(self, arg):
        Sigma_x, Sigma_d = arg[0], arg[1]
        # print(Sigma_x.shape, Sigma_u.shape, Sigma_d.shape)
        assert Sigma_d.shape == (self.n_d, self.n_d), "Shape of Sigma_d must match with n_d value specified when creating instance"
        # https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/gp_mpc.py#L310

        Sigma_xu = np.zeros((self.n_x, self.n_u))
        Sigma_u = np.zeros((self.n_u, self.n_u))
        Sigma_xd = np.zeros((self.n_x, self.n_d))
        Sigma_ud = np.zeros((self.n_u, self.n_d))

        Sigma = cs.vertcat(cs.horzcat(Sigma_x, Sigma_xu, Sigma_xd),
                           cs.horzcat(Sigma_xu.T, Sigma_u, Sigma_ud),
                           cs.horzcat(Sigma_xd.T, Sigma_ud.T, Sigma_d))

        return [Sigma]


def gen_boxconstr_from_poly(minimalpoly_intersection: cdd.Polyhedron):
    minHRep = np.array(minimalpoly_intersection.get_inequalities(), dtype=np.float32)
    minimal_poly_b, minimal_poly_H = minHRep[:, 0], -minHRep[:, 1:]
    # IGNORE THIS COMMENT. IT IS NOW IRRELEVANT. The method is now general for any intersections since we
    # don't need to reconstruct the bounds unless we need to sample random vectors from the box_constraint instance

    # # This assertion assumes we are taking an intersection between 2 box constraints i.e. hyperrectangles
    # # with no rotation about any of the axes. Under this assumption of no rotation, the resultant intersection
    # # is also a hyperrectangle of the same form. Thus, the rows of the H matrix (and hence b vector) can be re-arranged
    # # to get the matrices to be in the order expected by the box_constraint_direct class since that class
    # # tries to reconstruct the lb and ub based on the specified ordering.
    box_constr_inst = box_constraint_direct(H_np=minimal_poly_H, b_np=minimal_poly_b, skip_bound_construction=True)
    return box_constr_inst


def planar_region_gen_and_viz(viz=True, s_start_limit=np.array([[-2, -2]]).T, s_end_limit=np.array([[2, 2]]).T,
                              x0_delim=-0.5, x1_delim=0.5, ax=None):
    # print("delimiters", x0_delim, x1_delim)
    # r1 spans full x1 but x0 \in [-2, -0.5]
    r1_start, r1_end = np.array([[s_start_limit[0, :].item(), s_start_limit[1, :].item()]]).T,\
                       np.array([[x0_delim, s_end_limit[1, :].item()]]).T
    # r2 spans the remainder of x0 and x1 is limited to be 0.5 -> 2
    r2_start, r2_end = np.array([[x0_delim, x1_delim]]).T,\
                       np.array([[s_end_limit[0, :].item(), s_end_limit[1, :].item()]]).T
    # r3 also spans the remainder of x0 and now x1 too [-2, 0.5].
    r3_start, r3_end = np.array([[x0_delim, s_start_limit[1, :].item()]]).T,\
                       np.array([[s_end_limit[0, :].item(), x1_delim]]).T
    regions = [box_constraint(r1_start, r1_end), box_constraint(r2_start, r2_end), box_constraint(r3_start, r3_end)]
    if viz:
        visualize_regions(s_start_limit=s_start_limit, s_end_limit=s_end_limit, regions=regions, ax=ax)
    return regions


def visualize_regions(s_start_limit, s_end_limit, regions, ax=None):
    # Add values to generate samples that lie outside of the constraint set to test those too
    grid_check = generate_fine_grid(s_start_limit-1, s_end_limit+1, fineness_param=(45, 15), viz_grid=False)
    # print(grid_check.shape)
    mask = [[], [], []]
    for grid_vec_idx in range(grid_check.shape[-1]):
        grid_vec = grid_check[:, grid_vec_idx]
        for region_idx in range(len(regions)):
            test_constraint = regions[region_idx]
            mask[region_idx].append((test_constraint.sym_func(grid_vec) <= 0).all().item())
    passed_vecs = [0, 0, 0]
    colours = ['r', 'g', 'b']
    if ax is None:
        plt.figure()
        for i in range(len(regions)):
            passed_vecs[i] = grid_check[:, mask[i]]
            plt.scatter(passed_vecs[i][0], passed_vecs[i][1], c=colours[i], alpha=0.3)
    else:
        for i in range(len(regions)):
            passed_vecs[i] = grid_check[:, mask[i]]
            ax.scatter(passed_vecs[i][0], passed_vecs[i][1], c=colours[i], alpha=0.1)
    # print(grid_check)


def gen_poly_from_box(box_in: box_constraint):
    # print(box_in.b_np, box_in.b_np.shape)
    # print(box_in.H_np, box_in.H_np.shape)
    mat = cdd.Matrix(np.hstack([box_in.b_np, -box_in.H_np]), number_type='fraction')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    return poly


def compute_intersection(constr_set1: box_constraint, constr_set2: box_constraint, verbose=False) -> box_constraint:
    poly1 = gen_poly_from_box(constr_set1)
    poly2 = gen_poly_from_box(constr_set2)

    # H-representation of the first hypercube
    h1 = poly1.get_inequalities()
    # H-representation of the second hypercube
    h2 = poly2.get_inequalities()

    # join the two sets of linear inequalities; this will give the intersection
    hintersection = np.vstack((h1, h2))
    mat = cdd.Matrix(hintersection, number_type='fraction')
    mat.rep_type = cdd.RepType.INEQUALITY
    mat.canonicalize()
    # However, also generates redundant linear inequalities. To fix this, invoke the canonicalize method to get
    # rid of redundant inequalities in the matrix and generate the minimal H-representation of the intersection.
    minimalpoly_intersection = cdd.Polyhedron(mat)
    intersected_box_inst = gen_boxconstr_from_poly(minimalpoly_intersection)
    if verbose:
        print(intersected_box_inst)

    return intersected_box_inst


def gen_square_coords(lbs):
    dim_0_coord, dim_1_coord = lbs[0, :].item(), lbs[1, :].item()
    return (dim_0_coord, dim_0_coord, -dim_0_coord, -dim_0_coord), (dim_1_coord, -dim_1_coord, -dim_1_coord, dim_1_coord)


def gen_square_coords2(lbs, ubs):
    dim_1_ub, dim_1_lb, dim_0_ub, dim_0_lb = ubs[1, :].item(), lbs[1, :].item(), ubs[0, :].item(), lbs[0, :].item(),
    return (dim_0_ub, dim_0_ub, dim_0_lb, dim_0_lb), (dim_1_ub, dim_1_lb, dim_1_lb, dim_1_ub)


def simulate_hld_lld(X_test, regions, state_dim=2, eps=1e-5,
                     samples=np.array([[-2, -3]]).T, num_samples=1, verbose=True, ret_lld=False, unsqueeze=False):
    num_samples = samples.shape[-1]
    delta_r_k = np.zeros((2*state_dim, len(regions), num_samples))

    if verbose:
        display(Math(r'\text{}\,\,: {}{}<={}'.format("{State constraints}",
                                                     np2bmatrix([X_test.H_np]),
                                                     np2bmatrix([samples]),
                                                     np2bmatrix([X_test.b_np]))))

    for region_idx, region in enumerate(regions):
        region_H, region_b = region.H_np, region.b_np
        if verbose:
            display(Math(r'\text{}\,\,{}\,\,\text{}: {}{}<={}'.format("{Region}", region_idx+1, "{constraints}",
                                                                      np2bmatrix([region_H]),
                                                                      np2bmatrix([samples]),
                                                                      np2bmatrix([region_b]))))
        dim_idx = 0
        for inequality_idx in range(region_H.shape[0]):
            if verbose:
                print("Inequality %s" % inequality_idx)
            b = region_b[inequality_idx, :]
            # First half of inequalities correspond to lower bounds
            if inequality_idx < region_H.shape[0]//2:
                m = -(X_test.ub[dim_idx, :] + b)
                M = -(X_test.lb[dim_idx, :] + b)
            # Second half of inequalities correspond to upper bounds
            else:
                m = X_test.lb[dim_idx, :] - b
                M = X_test.ub[dim_idx, :] - b
            if verbose:
                print("delta_1")
                print(-X_test.ub[dim_idx, :], -b, m)
                display(Math(r'{} <= {} {} <= {}'.format(np2bmatrix([b+m]), np2bmatrix([region_H[[inequality_idx], :]]),
                                                         np2bmatrix([samples]), np2bmatrix([b]))))
                print("delta_0")
                print(-X_test.lb[dim_idx, :], b, M)
                display(Math(r'{} <= {} {} <= {}'.format(np2bmatrix([(b+eps)]), np2bmatrix([region_H[[inequality_idx], :]]),
                                                         np2bmatrix([samples]), np2bmatrix([b+M]))))
            ineq_row = region_H[inequality_idx, :]
            if unsqueeze:
                ineq_row = region_H[[inequality_idx], :]
            delta_1_bool = ((b+m) <= (ineq_row @ samples) <= b)
            delta_0_bool = ((b+eps) <= (ineq_row @ samples) <= (b+M))
            delta_assgt = 0 if delta_0_bool else 1
            if verbose:
                print("Delta Assignment: %s" % delta_assgt)
            assert delta_1_bool != delta_0_bool, "delta0 = delta1"
            delta_r_k[inequality_idx, region_idx, :] = delta_assgt
            # Circular rotation of dim_idx going from 0->state_dim for the lbs and then repeating from 0 for the ubs
            dim_idx = (dim_idx+1) % state_dim

    if ret_lld:
        # This function is called for warmstarting. When warmstarting, we call it using 1 sample only. and override a 2-D lld DM array. Thus
        # we can squeeze to remove the unnecessary 3rd dimension which casadi can't handle for opt vars/params anyway.
        return delta_r_k.squeeze()

    valid = 1
    for inequality_idx in range(regions[0].H_np.shape[0]):
        valid = cs.logic_and(valid, delta_r_k[inequality_idx, 0, 0])
    print("Region mask from deltas: %s" % False if not valid else True)


def construct_delta_constraint(delta_input_mask, X, U=None):
    constraint_obj = X
    # If U is passed -> delta controls is joint state and input.
    if U is not None:
        constraint_obj = combine_box(X, U, verbose=False)
    masked_lb, masked_ub = delta_input_mask @ constraint_obj.lb, delta_input_mask @ constraint_obj.ub
    delta_constraint_obj = box_constraint(masked_lb, masked_ub)
    return delta_constraint_obj
