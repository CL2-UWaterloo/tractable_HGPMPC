from matplotlib.transforms import Bbox
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from .box_constraint_utils import box_constraint_direct

import polytope as pc
import cdd
import matplotlib.pyplot as plt


def plot_uncertainty_bounds_1d(observed_pred, region_x, ax, colours, idx, custom_text=None):
    lower, upper = observed_pred.confidence_region()
    ax.plot(region_x.squeeze(), observed_pred.mean.numpy(), colours[idx], label='GP mode %s' % (idx+1) if custom_text is None else custom_text)
    ax.fill_between(region_x.squeeze(), lower.numpy(), upper.numpy(), alpha=0.5)


def plot_constraint_sets(plot_idxs, inp_type="shrunk_vec", alpha=0.8, colour='r', ax=None, **kwargs):
    if inp_type == "shrunk_vec":
        reqd_keys = ["shrunk_vec"]
        assert all([key in kwargs for key in reqd_keys]), "Missing required keys for plotting shrunk vec representation"
        box = box_constraint_direct(kwargs["shrunk_vec"], skip_bound_construction=False, plot_idxs=plot_idxs)
    else:
        raise NotImplementedError
    box.plot_constraint_set(ax=ax, alpha=alpha, colour=colour)


def save_fig(axes, fig_name, tick_sizes, tick_skip=1, k_range=None):
    for ax in axes:
        ax.tick_params(axis='both', labelsize=tick_sizes)
        if k_range is not None:
            ax.set_xticks(k_range[::tick_skip])
    plt.savefig(fig_name+'.svg', format='svg', dpi=300)


def save_fig_plt(fig_name, tick_sizes, tick_skip=1, k_range=None):
    plt.tick_params(axis='both', labelsize=tick_sizes)
    # plt.ticklabel_format(axis='both', style='plain')
    if k_range is not None:
        plt.xticks(k_range[::tick_skip])
    plt.savefig(fig_name+'.svg', format='svg', dpi=300)
    plt.show()


def generate_fine_grid(start_limit, end_limit, fineness_param, viz_grid=False, num_samples=200):
    # print(start_limit[0, :])
    # print(start_limit, start_limit.shape[-1])
    fine_coords_arrs = [torch.linspace(start_limit[idx, :].item(), end_limit[idx, :].item(),
                                       (int(fineness_param[idx]*(end_limit[idx, :]-start_limit[idx, :]).item()) if fineness_param is not None else num_samples))
                        for idx in range(start_limit.shape[0])]
    meshed = np.meshgrid(*fine_coords_arrs)
    grid_vectors = np.vstack([mesh.flatten() for mesh in meshed]) # Shape = dims * num_samples where num_samples is controller by fineness_param and start and end lims
    # print(grid_vectors.shape)

    if viz_grid:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(*[grid_vectors[axis_idx, :] for axis_idx in range(grid_vectors.shape[0])], c='b')
    return grid_vectors


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    items += [ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def dir_exist_or_create(base_path, sub_path=None):
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    store_dir = base_path
    if sub_path is not None:
        if not os.path.exists(base_path+sub_path):
            os.mkdir(base_path+sub_path)
        store_dir = base_path+sub_path
    return store_dir


def save_subplot(ax, figure, fig_name=None, base_path="C:\\Users\\l8souza\\PycharmProjects\\GP4MPC\\src\\images\\",
                 sub_path="scalar_motivating_example\\", extension='.svg'):
    assert fig_name is not None, "Need to set the fig_name attribute"
    # Save just the portion _inside_ the second axis's boundaries
    extent = full_extent(ax).transformed(figure.dpi_scale_trans.inverted())
    store_dir = dir_exist_or_create(base_path, sub_path=sub_path)
    figure.savefig(store_dir+fig_name+extension, bbox_inches=extent)
