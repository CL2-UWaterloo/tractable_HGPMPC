import math
import os.path

import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List

from common.box_constraint_utils import box_constraint
from common.plotting_utils import generate_fine_grid
from .residual_blocks import sum_of_sinusoids, polynomial_1d
from common.plotting_utils import plot_uncertainty_bounds_1d
# from mpl_toolkits import mplot3d


class DS_Sampler:
    def __init__(self, dims, method, num_samples, min_max_range, **kwargs):
        """
        Note: Sample methods are implemented assuming each dim has a min, max range that is independent of the others
        i.e. there is no additional p-norm (p \neq \infty) constraint on the vector itself which is why each
        dimension's 0->1 output range can be scaled independently of the others.

        :param dims: dimension of the vector to be sampled.
        :param method: sampling method to be used.
        Current methods: uniform random, clustering (pre-specified means),
        :param num_samples: number of samples to be generated.
        :param min_max_range: list of tuples. assert len(list) == dims and list[i][0], list[i][1] gives the min and max value
        for the ith dim respectively.
        """
        self.dims = dims
        self.method = method
        self.num_samples = num_samples
        self.min_max_range = min_max_range
        self.config_options = kwargs.get('config_options', {})

        self.sample_variances = kwargs.get('sample_variances', None)

    def normalize(self, x):
        return (x / np.sum(x))

    def get_samples(self, means=None, covariances=None, n_samples=None):
        if self.method == 'uniform random':
            return self._ur_sampling()
        if self.method == 'clustering':
            return self._clustering_sampling(means, covariances, n_samples)[0]

    def _clustering_sampling(self, means, covariances, n_samples):
        """
        Generate data by sampling from Gaussians with given mean vectors and covariances.

        Args:
            means (numpy array): Array of mean vectors. Each column is a mean vector.
            covariances (numpy array): Array of covariance matrices. Each 2D array is a covariance matrix.
            n_samples (numpy array or int): Number of samples to generate for each mean vector.

        Returns:
            X (numpy array): Array of generated data. Each row is a sample.
            y (numpy array): Array of labels corresponding to mean vectors.
        """
        self.means = means
        self.covariances = covariances

        # Determine the number of mean vectors
        n_means = means.shape[1]

        # Determine the number of dimensions (assumes all mean vectors have same number of dimensions)
        n_dims = means.shape[0]

        # If n_samples is int => same samples for each mean. Broadcast.
        if type(n_samples) == int:
            n_samples = np.ones((1, n_means)) * n_samples
        else:
            assert n_samples.shape[1] == n_means, "n_samples must either be int or n_samples.shape[1] == means.shape[1]"
        n_samples = n_samples.astype(np.int32).squeeze()
        self.n_samples = n_samples

        # Initialize arrays to store data and labels
        total_samples = (int(np.sum(n_samples).item()))
        X = np.zeros((n_dims, total_samples))
        y = np.zeros((1, total_samples))  # Cluster mask
        idxs = np.hstack([np.array([0]), np.cumsum(n_samples)]).astype(np.int32).squeeze()

        # Generate data for each mean vector
        for i in range(n_means):
            # Generate samples from Gaussian with given mean and covariance
            X[:, idxs[i]:idxs[i+1]] = np.random.multivariate_normal(mean=means[:, i], cov=covariances[i, :, :], size=n_samples[i]).T

            # Label data with index of mean vector
            y[:, idxs[i]:idxs[i+1]] = i

        self.cluster_mask = y

        return X, y

    def _clip_samples(self, samples):
        for i in range(len(self.min_max_range)):
            idx_range = self.min_max_range[i]
            np.clip(samples[i, :], idx_range[0], idx_range[1], out=samples[i, :])

    def _ur_sampling(self):
        # To move from [0, 1) to [start_limit, end_limit) multiply by end_limit-start_limit and then shift by start_limit
        dimwise_vectors = np.vstack([np.random.rand(self.num_samples)*(self.min_max_range[i][1] - self.min_max_range[i][0])+self.min_max_range[i][0]
                                     for i in range(len(self.min_max_range))])
        # print(self.min_max_range)
        # print(dimwise_vectors)
        return dimwise_vectors

    @staticmethod
    def viz_2d_ur(samples):
        plt.figure()
        plt.scatter(samples[0, :], samples[1, :], c='r')
        plt.show()


class GP_DS:
    def __init__(self, train_x, callables, state_space_polytope, regions, noise_vars, output_dims=None, **kwargs):
        """
        :param train_x => input data generated by DS_sampler class
        :param callables => number of callables = size of output vector

        Outputs
        train_y => (train_x, train_y) used to train the GP model
        Extensions: Accept sympy functions as inputs instead of hardcoding them and use lambdify to be able
        to apply it to numpy arrays.
        Refs:
        https://stackoverflow.com/questions/58784150/how-to-integrate-sympy-and-numpy-in-python-numpy-array-of-sympy-symbols
        https://stackoverflow.com/questions/58738051/add-object-has-no-attribute-sinh-error-in-numerical-and-symbolic-expression/58744816#58744816
        https://docs.sympy.org/latest/modules/utilities/lambdify.html
        """
        self.train_x = train_x
        self.train_tensor = self._convert_to_2d_tensor(self.train_x)
        self.input_dims, self.num_samples = self.train_tensor.shape
        self.output_dims = output_dims
        # input_dims specifies the dimension of the input vector and output_dims specifies the dimension of the residual vector.
        if self.output_dims is None:
            self.output_dims = self.input_dims
        self.state_space_polytope = state_space_polytope
        self.regions: List[box_constraint] = regions
        self.num_regions = len(self.regions)
        self.noise_vars = noise_vars
        self.callables = callables
        self.num_funcs = len(self.callables)
        assert self.num_funcs == self.output_dims*self.num_regions, "Number of func callables must be = dims*num_regions. If functions repeated across multiple positions then pass them multiple times"

        self.region_gpinp_exclusive = self.check_exclusivity(kwargs.get("gp_input_mask"), kwargs.get("delta_input_mask"))
        # Repetition from problem_setups.py. Unfortunately required for legacy use. Remove when possible.
        if kwargs.get("gpinp_subset_lb") is not None:
            self.gpinp_subset_lb, self.gpinp_subset_ub = kwargs.get("gpinp_subset_lb"), kwargs.get("gpinp_subset_ub")
        else:
            if kwargs.get("gp_input_mask") is not None:
                self.gpinp_subset_lb, self.gpinp_subset_ub = kwargs.get("gp_input_mask") @ np.vstack([self.state_space_polytope.lb, np.zeros((kwargs.get("n_u"), 1))]),\
                                                             kwargs.get("gp_input_mask") @ np.vstack([self.state_space_polytope.ub, np.zeros((kwargs.get("n_u"), 1))])
            else:
                self.gpinp_subset_lb, self.gpinp_subset_ub = self.state_space_polytope.lb, self.state_space_polytope.ub

    @staticmethod
    def check_exclusivity(gp_inp_mask, delta_input_mask):
        if gp_inp_mask is None or delta_input_mask is None:
            return False
        else:
            idxs = []
            for mask in (delta_input_mask, gp_inp_mask):
                idxs.append([])
                for row_idx in range(mask.shape[0]):
                    idxs[-1].append(np.nonzero(mask[row_idx, :])[0].item())

            var_intersect = np.intersect1d(idxs[0], idxs[1])
            if len(var_intersect) == 0:
                return True
            return False

    @staticmethod
    def _convert_to_2d_tensor(input_arr):
        return torch.Tensor(np.array(input_arr, ndmin=2))

    def _generate_regionspec_mask(self, input_arr=None, regions=None, delta_var_inp=False):
        """
        :param input_samples: Data samples generated by custom random sampling method. Num samples = num columns.
        Num rows = dim of vector space from which samples are drawn
        :param regions: List of box constraints
        :param delta_var_inp: If delta_var_inp is true that means the input array used is no longer the gp inputs but rather the
        delta inputs. In this case we need to check satisfaction for each region.
        :return:
        """
        # If no input array then we're working with the created dataset instead of some passed in array and
        # so we can just pull num_samples from the class vars
        num_samples = self.num_samples if (input_arr is None) else input_arr.shape[-1]
        init_mask = np.zeros((1, num_samples))
        # All samples are initialized to not be in any region. Then we iterate through all samples and assign
        # 1 to the region containing the sample while the rest retain the default 0.
        region_masks = defaultdict(init_mask.copy)
        if not self.region_gpinp_exclusive or delta_var_inp:
            for sample_idx in range(num_samples):
                if input_arr is None:
                    sample = self.train_tensor[:, sample_idx]
                    # print("Train tensor shape: %s" % list(self.train_tensor.shape))
                else:
                    sample = input_arr[:, sample_idx]
                    # print("Input tensor shape: %s" % list(input_arr.shape))
                for region_idx, region in enumerate(self.regions if regions is None else regions):
                    if region.check_satisfaction(sample):
                        region_masks[region_idx][:, sample_idx] = 1
                    else:
                        region_masks[region_idx][:, sample_idx] = 0
            # print(region_masks)
            # print([region_mask.shape for region_mask in self.region_masks.values()])
        else:
            idx_gen_arr = np.ones((1, num_samples))
            # Samples are uniformly distributed across all regions
            region_samples = num_samples // self.num_regions
            for region_idx in range(self.num_regions):
                nonzero_idxs = np.nonzero(idx_gen_arr)[1]
                # Last region gets remaining idxs.
                if region_idx == (self.num_regions - 1):
                    random_idxs = nonzero_idxs
                else:
                    random_idxs = np.random.choice(nonzero_idxs, region_samples, replace=False)
                # print(random_idxs.shape)
                idx_gen_arr[0, random_idxs] = 0
                region_masks[region_idx][0, random_idxs] = 1
                # print(region_masks[region_idx])

        # Assert that all samples have been assigned a region label.
        idx_union = np.union1d(np.nonzero(region_masks[0])[1], np.nonzero(region_masks[1])[1])
        for region_idx in range(2, self.num_regions):
            idx_union = np.union1d(idx_union, np.nonzero(region_masks[region_idx])[1])
        # print(input_arr)
        assert len(idx_union) == num_samples, "Not all samples have been assigned a region label"

        if input_arr is not None or regions is not None:
            return region_masks
        # If dealing with the dataset then overwrite the class variable region_masks
        self.region_masks = region_masks

    def generate_random_delta_var_assignments(self, mask_dict_inp, delta_dim):
        delta_var_assgts = np.zeros((delta_dim, mask_dict_inp[0].shape[-1]))
        for region_idx in mask_dict_inp.keys():
            idxs = np.nonzero(mask_dict_inp[region_idx].squeeze() > 0)[0]
            num_idxs = len(idxs)
            region = self.regions[region_idx]
            assert region.dim == delta_dim, "Region dim %d does not match delta dim %d" % (region.dim, delta_dim)
            delta_var_assgts[:, idxs] = region.get_random_vectors(num_idxs)
        return delta_var_assgts

    def get_region_idxs(self, mask_input=None):
        # dict of numpy arrays. array at key i tells us which indices of train_x belong to region_i
        if mask_input is None:
            mask_input = self.region_masks
        regionwise_sample_idxs = {}
        for region_idx, region in mask_input.items():
            # nonzero pulls out those indices of the mask which have values 1 and hence correspond
            # to those samples belonging to this region_idx in the input_arr that generate the mask_input
            # array.
            regionwise_sample_idxs[region_idx] = np.nonzero(mask_input[region_idx].squeeze() > 0)
            # print(regionwise_sample_idxs[region_idx])
        return regionwise_sample_idxs

    def _gen_white_noise(self, num_samples=None, region_masks=None, noise_verbose=False, ax=None):
        """
        :param dims: Dim of input vector (= train_x.shape[0] Inputs can be over joint state and input space
        of the control system. But writing train_*x* is just to keep convention with normal notation for ML training inputs)
        :param num_samples: number of training samples generated. Number of white noise vectors generated must be equal to this.
        :param noise_vars: List of noise variances for each region
        :param region_masks: Dictionary of masks that map samples to regions that are within. Masks are output from the
        generate_regionspec_mask function
        """
        if num_samples is None:
            num_samples = self.num_samples
            region_masks = self.region_masks
        noise_samples = torch.zeros((self.output_dims, num_samples))
        colours = ['r', 'g', 'b']
        for region_idx, region_mask in region_masks.items():
            noise_var = self.noise_vars[region_idx]
            region_noise = ((torch.randn((self.output_dims, num_samples)) * math.sqrt(noise_var)) * region_mask)
            noise_samples += region_noise
            if ax is not None:
                ax.hist(region_noise, bins=16, color=colours[region_idx])
            # print(noise_var)
        # print(noise_samples.shape)
        return noise_samples

    def generate_outputs(self, input_arr=None, no_noise=False, ret_noise_sample=False,
                         return_op=False, noise_verbose=False, noise_plot_ax=None, mask_dict_override=None):
        # Will be none when generating noisy samples and not None when passing a fine grained equispaced set of points
        # in the state space to visualize the true function mean (note the true function also has a covariance because of the stochasticity estimates.
        if input_arr is None:
            input_arr = self.train_tensor
        if mask_dict_override is None:
            if no_noise or return_op:
                mask = self._generate_regionspec_mask(input_arr=input_arr)
                # print(mask)
            else:
                self._generate_regionspec_mask()
                mask = self.region_masks
        else:
            mask = mask_dict_override
            if input_arr.shape[-1] > 1:
                assert input_arr[0].shape == mask[0].squeeze().shape, "Mask shape mismatch: %s, %s" % (input_arr[0].shape, mask[0].squeeze().shape)
            else:
                assert mask[0].shape[0] == 1, "Mask shape mismatch: %s, %s" % (input_arr[0].shape, mask[0].squeeze().shape)
        # print(mask)

        # Ex: For 2D case with 2 regions we have f1->f4. The mask remains same for output of (f1, f2) and (f3, f4) since f1, f2 are always active in region 1
        # and f3, f4 are always active in region 2.
        segregated_ops = [func(input_arr) * mask[idx // self.output_dims] for idx, func in enumerate(self.callables)]
        # print(mask)
        # print(segregated_ops)
        # if no_noise:
        #     print((mask[0//self.dims]+mask[2//self.dims] > 0).all())
        concd_ops_dims = []
        for i in range(self.num_regions):
            concd_ops_dims.append(np.vstack(segregated_ops[i*self.output_dims: (i+1)*self.output_dims]))
        # print(concd_ops_dims)

        train_outputs = torch.sum(torch.from_numpy(np.stack(concd_ops_dims)), 0)
        # print(train_outputs)
        if no_noise or return_op:
            if no_noise:
                return train_outputs
            else:
                noise_samples = self._gen_white_noise(num_samples=input_arr.shape[-1], region_masks=mask,
                                                      noise_verbose=noise_verbose, ax=noise_plot_ax)
                # print("sampled noise: %s" % noise_samples)
                if ret_noise_sample:
                    return noise_samples
                else:
                    if noise_verbose:
                        print("Residual mean: %s, residual sampled noise: %s" % (train_outputs, noise_samples))
                    return train_outputs + noise_samples

        noise_samples = self._gen_white_noise()
        self.train_y = train_outputs + noise_samples

    def viz_outputs_1d(self, fineness_param=(51, ), ax1=None, true_only=False):
        if ax1 is None:
            f, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        start_limit, end_limit = self.gpinp_subset_lb, self.gpinp_subset_ub
        fine_x = torch.linspace(start_limit.item(), end_limit.item(), fineness_param[0]*(end_limit-start_limit).item())
        true_y = self.generate_outputs(input_arr=self._convert_to_2d_tensor(fine_x), no_noise=True)
        ax1.plot(fine_x, true_y.squeeze(), 'b')
        if true_only:
            ax1.legend(["True function"], fontsize=25)
        else:
            ax1.scatter(self.train_x, self.train_y.squeeze(), color='r')
            ax1.legend(["True function", "Noisy samples"], fontsize=25)

        # ax1.set_title('True function w/ noisy samples', fontsize=25)

    @staticmethod
    def gen_excl_fine_mask(fine_x, region_idx):
        init_mask = np.zeros((1, fine_x.shape[-1]))
        region_masks = defaultdict(init_mask.copy)
        region_masks[region_idx] = np.ones((1, fine_x.shape[-1]))
        return region_masks

    def viz_outputs_1d_excl(self, fineness_param=(51, ), ax=None, true_only=False):
        """
        This method is intended to be used for a single gp input variable generating a single residual output term. It is to be
        used only when the region_gpinp_excl is set to True.
        :param fineness_param: Controls the number of samples in the fine grid generated to visualize the true function
        :param ax: Input ax for plotting if this method is called by an external function that already has a figure defined.
        :param true_only: If true, display only the true function and not the noisy samples.
        :return:
        """
        assert self.region_gpinp_exclusive is True, "This method is intended to be used only when the gp input and delta" \
                                                    " variables are mutually exclusive"
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(12, 8))
        # fine_y = \[\]\*num_regions
        # for region_idx in num_regions:
        #     fine grid gets passed through all function callables.
        #     fine_y\[region_idx\] = callables\[region_idx\](fine_x)
        #     plot(fine_x, fine_y\[region_idx\])
        start_limit, end_limit = self.gpinp_subset_lb, self.gpinp_subset_ub
        assert start_limit.shape[-1] == 1 and end_limit.shape[-1] == 1, "This method is intended to be used only when " \
                                                                        "the gp input is a single variable"
        fine_x = torch.linspace(start_limit.item(), end_limit.item(), fineness_param[0]*(end_limit-start_limit).astype(np.uint8).item())
        true_y_arr = [0]*self.num_regions
        colours = ['grey', 'cyan', 'mediumvioletred', 'y', 'k']
        scatter_colours = ['black', 'mediumturquoise', 'red']
        for region_idx in range(self.num_regions):
            region_masks = self.gen_excl_fine_mask(fine_x=fine_x, region_idx=region_idx)
            true_y_arr[region_idx] = self.generate_outputs(input_arr=self._convert_to_2d_tensor(fine_x), no_noise=True, mask_dict_override=region_masks)
            y_lower = true_y_arr[region_idx] - 2*math.sqrt(self.noise_vars[region_idx])
            y_upper = true_y_arr[region_idx] + 2*math.sqrt(self.noise_vars[region_idx])
            ax.plot(fine_x, true_y_arr[region_idx].squeeze(), colours[region_idx], label="Residual for region: %s" % (region_idx+1))
            ax.fill_between(fine_x, y_lower.squeeze(), y_upper.squeeze(), alpha=0.5, color=colours[region_idx])
            if not true_only:
                bool_select = self.region_masks[region_idx].squeeze() == 1
                ax.scatter(self.train_x[:, bool_select],
                           self.train_y[:, bool_select].squeeze(),
                           color=scatter_colours[region_idx])
                # ax.legend(["True function", "Noisy samples"], fontsize=25)
            # ax.set_ylim([-0.6, 0.9])
            # ax.legend(fontsize=20, loc='upper center')

    def generate_fine_grid(self, fineness_param, with_mask=False, regions=None, num_samples=None):
        if fineness_param is not None:
            fine_grid = generate_fine_grid(self.gpinp_subset_lb, self.gpinp_subset_ub, fineness_param=fineness_param)
        else:
            assert num_samples is not None, "Either fineness_param or num_samples must be specified"
            fine_grid = generate_fine_grid(self.gpinp_subset_lb, self.gpinp_subset_ub, fineness_param=None, num_samples=num_samples)
        if with_mask:
            mask = self._generate_regionspec_mask(input_arr=fine_grid, regions=regions)
            return fine_grid, mask
        else:
            return fine_grid

    @staticmethod
    def ret_regspec_samples(input_arrs, region_idx, regionwise_idxs_dict):
        return [input_arr[:, regionwise_idxs_dict[region_idx]] for input_arr in input_arrs]

    @staticmethod
    def get_colours(num_regions, cm_name='Spectral'):
        cmap = plt.cm.get_cmap('Spectral')
        intervals = list(np.linspace(0, 1, num_regions))
        colours = [cmap(intervals[i]) for i in range(len(intervals))]
        return colours

    def plot_true_func_2d(self, axes, fineness_param, true_plot=False, samples_plot=False, op_arr=None, colours=None, io_only=False):
        assert len(axes) == self.output_dims, "Number of axes must match the dimension of the state vector " \
                                             "(ex: 2-dim residual output state vectors has 2 separate functions governing the dynamics of the output)"
        fine_grid, mask = self.generate_fine_grid(fineness_param=fineness_param, with_mask=True)
        true_y = self.generate_outputs(input_arr=torch.from_numpy(fine_grid), no_noise=True)
        if io_only:
            return fine_grid, true_y
        fine_regionwise_idxs = self.get_region_idxs(mask_input=mask)
        sample_regionwise_idxs = self.get_region_idxs()
        if colours is None:
            colours = self.get_colours(self.num_regions)
        if op_arr is None:
            op_arr = self.train_y
        colours = ['lightsalmon', 'cyan', 'limegreen']
        for ax_idx, ax in enumerate(axes):
            for region_idx in range(len(self.regions)):
                region_finex, region_fineop = self.ret_regspec_samples([fine_grid, true_y], region_idx, fine_regionwise_idxs)
                region_samples, op_scalars = self.ret_regspec_samples([self.train_x, op_arr], region_idx, sample_regionwise_idxs)
                if true_plot:
                    ax.scatter3D(region_finex[0, :], region_finex[1, :], region_fineop[ax_idx, :].squeeze(), color=colours[region_idx],
                                 label="Region %s" % (region_idx+1))
                if samples_plot:
                    ax.scatter3D(region_samples[0, :], region_samples[1, :],
                                 op_scalars[ax_idx, :].squeeze(), color=colours[-region_idx-1])
            ax.xaxis.set_tick_params(labelsize=20)
            ax.yaxis.set_tick_params(labelsize=20)
            ax.zaxis.set_tick_params(labelsize=20)
            ax.set_xlabel("State 1", fontsize=20, labelpad=20)
            ax.set_ylabel("State 2", fontsize=20, labelpad=20)
            ax.set_zlabel("Residual (g(x))", fontsize=20, labelpad=20)
            ax.legend(loc="upper center", fontsize=20)
        return fine_grid, fine_regionwise_idxs

    def viz_outputs_2d(self, fineness_param=(51,), true_only=False, ax=None, io_only=False):
        """
        Parameters
        ----------
        fineness_param Specifies how many samples are generated across the range of the input space. Higher param for
        the dimension => more unique values for that dimension get generated for the cartesian product.
        true_only Only plot the true function
        dim_limit, dim_idxs If true then the input space has more than 3 dims so to visualize we limit x-y plane of the plot
        to only be 2 selected dimensions of the input space. THe selected dimensions are specified in dim_idxs
        ax if passing in an axis object then one is not created.
        """
        assert self.input_dims == 2, "Dimension of the input space must be 2 to use this function"
        if ax is None:
            fig = plt.figure(figsize=(16, 32))
            axes = []
            for i in range(self.output_dims):
                ax = fig.add_subplot(self.output_dims, 1, i+1, projection='3d')
                axes.append(ax)
            # ax1, ax2 = fig.add_subplot(2, 1, 1, projection='3d'), fig.add_subplot(2, 1, 2, projection='3d')
            # axes = [ax1, ax2]
        else:
            axes = [ax]
        # fine_grid = self.generate_fine_grid(fineness_param=fineness_param)
        # print("Fine grid shape: %s" % list(fine_grid.shape))
        # true_y = self.generate_outputs(input_arr=fine_grid, no_noise=True)
        ret_val = self.plot_true_func_2d(axes=axes, fineness_param=fineness_param, true_plot=True,
                                         samples_plot=not true_only, io_only=io_only)
        if io_only:
            return ret_val
        str_append = ""

        for ax_idx, ax in enumerate(axes):
            # ax.scatter3D(fine_grid[0, :], fine_grid[1, :], true_y[ax_idx, :].squeeze(), 'b')
            if not true_only:
                ax.legend(["True function", "Noisy samples"])
                str_append = "w/ noisy samples"
            # ax.set_title('True function for state x%s %s' % (ax_idx+1, str_append))


def test_1d_op_2d_inp_sinusoids(regions, sine_freqs=((1.2, 1, 0.75), (0.7, 0.4, 0.5), (1.2, 1, 1.25)),
                                sine_mults=((0.15, 0.2, 0.1), (0.1, 0.3, 0.2), (0.15, 0.4, 0.1)),
                                start_limit=np.array([[-2, -2]]).T, end_limit=np.array([[2, 2]]).T, num_points=50,
                                noise_vars=(0.05, 0.02, 0.03), fineness_param=(11, 11), no_viz=True, true_only=False):
    assert len(regions) == 3, "Must pass 3 regions for this example."

    # Since the sampler generates train_x samples, the dims arg is 2 here. Samples are generated uniform randomly.
    r1_d1 = lambda x: sum_of_sinusoids(x[0, :], freq_params=sine_freqs[0], scale_params=sine_mults[0])
    r2_d1 = lambda x: sum_of_sinusoids(x[1, :], freq_params=sine_freqs[1], scale_params=sine_mults[1])
    r3_d1 = lambda x: sum_of_sinusoids(x[0, :], freq_params=sine_freqs[2], scale_params=sine_mults[2])

    min_max_range = [(start_limit[i, :], end_limit[i, :]) for i in range(2)]
    sampler_inst = DS_Sampler(dims=2, method="uniform random", num_samples=num_points,
                              min_max_range=min_max_range)
    coarse_x = sampler_inst.get_samples()

    state_space_polytope = box_constraint(start_limit, end_limit)
    callables = [r1_d1, r2_d1, r3_d1]

    dataset = GP_DS(train_x=coarse_x, callables=callables, state_space_polytope=state_space_polytope,
                    regions=regions, noise_vars=noise_vars, output_dims=1)
    dataset.generate_outputs()
    if not no_viz:
        dataset.viz_outputs_2d(fineness_param=fineness_param, true_only=true_only)

    return dataset


def test_1d_op_1d_inp_poly(regions, poly_coeffs, start_limit, end_limit, gp_input_mask, delta_input_mask, n_u,
                           num_points=50, noise_vars=(0.05, 0.02), fineness_param=(11, 11), no_viz=True, true_only=False,
                           method="uniform random", means=None, covs=None, num_samples=1000, scale_poly=1):
    """
    Function takes in:
    -   regions that split the state input space (or, more generally, a subspace of it)
    -   polynomial coefficients (user-specified so that I can test that residual magnitude is reasonable across
        allowed range of gp_inputs. Must be a list of length 2 (i.e. 2 regions).
        Each element must be a tuple whose entries contain the poly coeffs.
    -   fineness_param Amount of discretization used to generate the meshgrid for visualizing the true function.
    -   start and end limit used for generating the input samples for GP_DS.
    """
    gp_inp_dim, num_regions = 1, 3
    assert len(regions) == num_regions, "Must pass %s regions for this example." % num_regions
    assert len(noise_vars) == num_regions, "Must pass %s noise sigma for this example." % num_regions
    assert len(poly_coeffs) == num_regions, "Must pass %s poly coeff lists for this example." % num_regions
    assert len(fineness_param) == gp_inp_dim, "Fine param is a single value since 1-D gp input"
    assert start_limit.shape[-1] == np.linalg.matrix_rank(gp_input_mask), "Start limit must have same dim as rank of gp input mask"

    # Since the sampler generates train_x samples, the dims arg is 2 here. Samples are generated uniform randomly.
    r1_res_fn = lambda x: scale_poly * polynomial_1d(coefficients=poly_coeffs[0], inputs=x)
    r2_res_fn = lambda x: scale_poly * polynomial_1d(coefficients=poly_coeffs[1], inputs=x)
    r3_res_fn = lambda x: scale_poly * polynomial_1d(coefficients=poly_coeffs[2], inputs=x)

    min_max_range = [(start_limit[i, :], end_limit[i, :]) for i in range(1)]
    sampler_inst = DS_Sampler(dims=gp_inp_dim, method=method, num_samples=num_points,
                              min_max_range=min_max_range)
    if method == "clustering":
        assert means is not None, "Must pass means and covs for clustering method"
        num_points = (num_points // means.shape[1])
    coarse_x = sampler_inst.get_samples(means=means, covariances=covs, n_samples=num_points)

    state_space_polytope = box_constraint(start_limit, end_limit)
    callables = [r1_res_fn, r2_res_fn, r3_res_fn]

    dataset = GP_DS(train_x=coarse_x, callables=callables, state_space_polytope=state_space_polytope,
                    regions=regions, noise_vars=noise_vars, output_dims=1,
                    gp_input_mask=gp_input_mask, delta_input_mask=delta_input_mask, n_u=n_u,
                    gpinp_subset_lb=start_limit, gpinp_subset_ub=end_limit)
    dataset.generate_outputs()
    if not no_viz:
        dataset.viz_outputs_1d_excl(fineness_param=fineness_param, true_only=true_only)

    return dataset


def combine_box(box1, box2, verbose=False):
    box1_lb, box1_ub = box1.lb, box1.ub
    box2_lb, box2_ub = box2.lb, box2.ub
    new_lb, new_ub = np.vstack((box1_lb, box2_lb)), np.vstack((box1_ub, box2_ub))
    new_constraint = box_constraint(new_lb, new_ub)
    return new_constraint



