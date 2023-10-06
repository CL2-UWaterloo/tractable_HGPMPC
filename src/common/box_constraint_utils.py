from ctypes import Union

import torch
import matplotlib.pyplot as plt
import numpy as np
import polytope as pc


class box_constraint:
    """
    Bounded constraints lb <= x <= ub as polytopic constraints -Ix <= -b and Ix <= b. np.vstack(-I, I) forms the H matrix from III-D-b of the paper
    """
    def __init__(self, lb=None, ub=None, plot_idxs=None):
        """
        :param lb: dimwise list of lower bounds.
        :param ub: dimwise list of lower bounds.
        :param plot_idxs: When plotting, the box itself might be defined in some dimension greater than 2 but we might only want to
        plot the workspace variables and so plot_idxs allows us to limit the consideration of plot_constraint_set to those variables.
        """
        self.lb = np.array(lb, ndmin=2).reshape(-1, 1)
        self.ub = np.array(ub, ndmin=2).reshape(-1, 1)
        self.plot_idxs = plot_idxs
        self.dim = self.lb.shape[0]
        assert (self.lb < self.ub).all(), "Lower bounds must be greater than corresponding upper bound for any given dimension, %s, %s" % (self.lb, self.ub)
        self.setup_constraint_matrix()

    def __str__(self): return "Lower bound: %s, Upper bound: %s" % (self.lb, self.ub)

    def get_random_vectors(self, num_samples):
        rand_samples = np.random.rand(self.dim, num_samples)
        for i in range(self.dim):
            scale_factor, shift_factor = (self.ub[i] - self.lb[i]), self.lb[i]
            rand_samples[i, :] = (rand_samples[i, :] * scale_factor) + shift_factor
        return rand_samples

    def setup_constraint_matrix(self):
        dim = self.lb.shape[0]
        # Casadi can't do matrix mult with Torch instances but only numpy instead. So have to use the np version of the H and b matrix/vector when
        # defining constraints in the opti stack.
        self.H_np = np.vstack((-np.eye(dim), np.eye(dim)))
        self.H = torch.Tensor(self.H_np)
        # self.b = torch.Tensor(np.hstack((-self.lb, self.ub)))
        self.b_np = np.vstack((-self.lb, self.ub))
        self.b = torch.Tensor(self.b_np)
        # print(self.b)
        self.sym_func = lambda x: self.H @ np.array(x, ndmin=2).T - self.b

    def check_satisfaction(self, sample):
        # If sample is within the polytope defined by the constraints return 1 else 0.
        # print(sample, np.array(sample, ndmin=2).T, self.sym_func(sample), self.b)
        return (self.sym_func(sample) <= 0).all()

    def generate_uniform_samples(self, num_samples):
        n = int(np.round(num_samples**(1./self.lb.shape[0])))

        # Generate a 1D array of n equally spaced values between the lower and upper bounds for each dimension
        coords = []
        for i in range(self.lb.shape[0]):
            coords.append(np.linspace(self.lb[i, 0], self.ub[i, 0], n))

        # Create a meshgrid of all possible combinations of the n-dimensions
        meshes = np.meshgrid(*coords, indexing='ij')

        # Flatten the meshgrid and stack the coordinates to create an array of size (K, n-dimensions)
        samples = np.vstack([m.flatten() for m in meshes])

        # Truncate the array to K samples
        samples = samples[:num_samples, :]

        # Print the resulting array
        return samples

    def plot_constraint_set(self, ax=None, alpha=0.8, colour='r'):
        if type(self.plot_idxs) is list:
            if len(self.plot_idxs) != 2:
                if self.dim != 2:
                    raise NotImplementedError("Plotting is only possible for 2D box constraints")
                else:
                    self.plot_idxs = [0, 1]
        else:
            if self.dim != 2:
                assert type(self.plot_idxs) is list, "plot_idxs must be a list of indices of the dimensions to plot"
            else:
                self.plot_idxs = [0, 1]

        plot_mat = np.zeros((2, self.dim))
        for row_idx, plot_idx in enumerate(self.plot_idxs):
            plot_mat[row_idx, plot_idx] = 1

        H_np = np.vstack((-np.eye(2), np.eye(2)))
        b_np_upper, b_np_lower = self.b_np[:self.dim, :], self.b_np[self.dim:, :]
        b_np = np.vstack([plot_mat @ b_np_upper, plot_mat @ b_np_lower])

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        poly_temp = pc.Polytope(H_np, b_np)
        vertices = pc.extreme(poly_temp)
        ax.fill(vertices[:, 0], vertices[:, 1], color=colour, alpha=alpha, edgecolor='black')

    def clip_to_bounds(self, samples):
        return np.clip(samples, self.lb, self.ub)


class box_constraint_direct(box_constraint):
    """
    Bounded constraints lb <= x <= ub as polytopic constraints -Ix <= -b and Ix <= b. np.vstack(-I, I) forms the H matrix from III-D-b of the paper
    """
    def __init__(self, b_np, skip_bound_construction=False, plot_idxs=None):
        self.plot_idxs = plot_idxs
        self.b_np = np.array(b_np, ndmin=2).reshape(-1, 1)
        self.dim = self.b_np.shape[0] // 2
        self.H_np = np.vstack((-np.eye(self.dim), np.eye(self.dim)))
        self.skip_bound_construction = skip_bound_construction
        if not skip_bound_construction:
            self.retrieve_ub_lb()
        self.setup_constraint_matrix()

    def retrieve_ub_lb(self):
        lb = -self.b_np[:self.dim]
        ub = self.b_np[self.dim:]
        self.lb = np.array(lb, ndmin=2)
        self.ub = np.array(ub, ndmin=2)

    def __str__(self):
        if not self.skip_bound_construction:
            return "Hx<=b ; H: %s, b: %s" % (self.H_np, self.b_np)
        else:
            super().__str__()

    def get_random_vectors(self, num_samples):
        if self.skip_bound_construction:
            assert NotImplementedError, "You chose to skip bound construction. The sampling method currently implemented" \
                                        "requires bounds to generate random samples."
        rand_samples = np.random.rand(self.dim, num_samples)
        for i in range(self.dim):
            scale_factor, shift_factor = (self.ub[i] - self.lb[i]), self.lb[i]
            rand_samples[i, :] = (rand_samples[i, :] * scale_factor) + shift_factor
        return rand_samples

    def setup_constraint_matrix(self):
        self.H = torch.Tensor(self.H_np)
        self.b = torch.Tensor(self.b_np)
        # print(self.b)
        self.sym_func = lambda x: self.H @ np.array(x, ndmin=2).T - self.b


def combine_box(box1, box2, verbose=False):
    box1_lb, box1_ub = box1.lb, box1.ub
    box2_lb, box2_ub = box2.lb, box2.ub
    new_lb, new_ub = np.vstack((box1_lb, box2_lb)), np.vstack((box1_ub, box2_ub))
    new_constraint = box_constraint(new_lb, new_ub)
    if verbose:
        print(new_constraint)
    return new_constraint


class infnorm_constraint(box_constraint):
    def __init__(self, bound, dim=1):
        assert bound > 0, "Bound value must be positive"
        assert dim >= 1, "Dimension of state vector must be at least 1"
        self.lb = np.array(-bound*dim)
        self.ub = np.array(bound*dim)
        self.setup_constraint_matrix()
