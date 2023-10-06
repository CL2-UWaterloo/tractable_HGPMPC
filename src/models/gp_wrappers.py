import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torch

from .utils import train_test, piecewise_train_test, Piecewise_GPR_Callback, MultidimGPR_Callback, GPR_Callback, gp_viz_plotter
from ds_utils import GP_DS
from common.plotting_utils import plot_uncertainty_bounds_1d


def gen_wrapped_pw_gp(ds_ndim_test, return_error_attrs, num_regions, num_iter=400,
                      terminate_by_change=False, save_to_file=False, load_from_file=False, file_save_name=None, file_load_name=None):
    if save_to_file:
        assert file_save_name is not None, "Must provide save file name for GP models if desiring to save the models and likelihoods"
    if load_from_file:
        assert file_load_name is not None, "Must provide load file name for GP models/likelihoods if load_from_file=True"
    if not load_from_file:
        returned_train_ops = piecewise_train_test(ds_ndim_test, no_squeeze=True, return_trained_covs=return_error_attrs,
                                                  num_iter=num_iter, terminate_by_change=terminate_by_change)
        if not return_error_attrs:
            likelihoods_piecewise_nd, piecewise_models_nd = returned_train_ops
        else:
            likelihoods_piecewise_nd, piecewise_models_nd, region_trained_covs = returned_train_ops
        likelihoods, models = likelihoods_piecewise_nd, piecewise_models_nd
        if save_to_file:
            print("SAVING GP MODELS AND LIKELIHOODS TO FILE: %s", file_save_name+".pkl")
            with open(os.getcwd()+"\\"+file_save_name+".pkl", "wb") as pklfile:
                data_dict = {"ds_inst": ds_ndim_test, "likelihoods_pw": likelihoods, "models_pw": models}
                pkl.dump(data_dict, pklfile)
    else:
        print("LOADING GP MODELS AND LIKELIHOODS FROM FILE: %s", file_load_name+".pkl")
        with open(os.getcwd()+"\\"+file_load_name+".pkl", "rb") as pklfile:
            data_dict = pkl.load(pklfile)
            ds_ndim_test, likelihoods, models = data_dict["ds_inst"], data_dict["likelihoods_pw"], data_dict["models_pw"]

    for model_idx in range(len(models)):
        models[model_idx].eval()
        likelihoods[model_idx].eval()
    res_input_dim = ds_ndim_test.input_dims
    res_output_dim = ds_ndim_test.output_dims
    piecewise_gp_wrapped = Piecewise_GPR_Callback('f', likelihoods, models,
                                                  output_dim=res_output_dim, input_dim=res_input_dim, num_regions=num_regions,
                                                  opts={"enable_fd": True})
    if return_error_attrs:
        return piecewise_gp_wrapped, models, likelihoods, region_trained_covs
    else:
        return piecewise_gp_wrapped, models, likelihoods


def construct_gp_wrapped_piecewise(ds_inst_in: GP_DS, regions, viz=True, fineness_param=(20, 20), verbose=True,
                                   return_error_attrs=False, seed=None, true_only=False, num_iter=400, terminate_by_change=False,
                                   save_to_file=False, load_from_file=False,
                                   file_save_name=None, file_load_name=None):
    if seed is not None:
        np.random.seed(seed)

    assert ds_inst_in is not None, "Must provide a dataset instance to construct GP wrapped piecewise"
    ds_ndim_test = ds_inst_in
    if verbose:
        print("Training piecewise GP")

    num_regions = len(regions)
    ops = gen_wrapped_pw_gp(ds_ndim_test, return_error_attrs, num_regions,
                            num_iter=num_iter, terminate_by_change=terminate_by_change,
                            save_to_file=save_to_file, load_from_file=load_from_file,
                            file_save_name=file_save_name, file_load_name=file_load_name)
    if return_error_attrs:
        piecewise_fullgp_wrapped, models, likelihoods, region_trained_covs = ops
    else:
        piecewise_fullgp_wrapped, models, likelihoods = ops

    if viz:
        ops = gp_viz_plotter(ds_ndim_test, piecewise_fullgp_wrapped,
                             fineness_param=fineness_param, true_only=true_only, return_error_attrs=return_error_attrs)
        fig, axes = ops[0], ops[1]
        if return_error_attrs:
            mean_error_list, cov_list = ops[2], ops[3]

        if return_error_attrs:
            ops = [piecewise_fullgp_wrapped, ds_ndim_test, axes, fig, region_trained_covs, mean_error_list, cov_list]
            return ops
        else:
            ops = [piecewise_fullgp_wrapped, ds_ndim_test, axes, fig]
            return ops
    else:
        return piecewise_fullgp_wrapped, ds_ndim_test


def construct_gp_wrapped_global(ds_inst_in, viz=True, ax=None, fineness_param=(20, 20), verbose=True,
                                return_error_attrs=False, num_iter=400, terminate_by_change=False,
                                save_to_file=False, load_from_file=False, file_save_name=None, file_load_name=None):
    assert ds_inst_in is not None, "You must pass an input dataset that is shared across the global and piecewise case and generated by construct_gp_wrapped_piecewise"
    ds_ndim_test = ds_inst_in

    if save_to_file:
        assert file_save_name is not None, "Must provide save file name for GP models if desiring to save the models and likelihoods"
    if load_from_file:
        assert file_load_name is not None, "Must provide load file name for GP models/likelihoods if load_from_file=True"
    if not load_from_file:
        returned_train_ops = train_test(ds_ndim_test, no_squeeze=True, verbose=verbose, return_trained_covs=return_error_attrs,
                                        num_iter=num_iter, terminate_by_change=terminate_by_change)
        if not return_error_attrs:
            likelihoods_2d, models_2d = returned_train_ops
        else:
            likelihoods_2d, models_2d, baseline_trained_covs = returned_train_ops
        likelihoods, models = likelihoods_2d, models_2d

        if save_to_file:
            with open(os.getcwd()+"\\"+file_save_name+".pkl", "rb") as pklfile:
                data_dict = pkl.load(pklfile)
            with open(os.getcwd()+"\\"+file_save_name+".pkl", "wb") as pklfile:
                data_dict.update({"ds_inst": ds_ndim_test, "likelihoods_glob": likelihoods, "models_glob": models})
                pkl.dump(data_dict, pklfile)
    else:
        with open(os.getcwd()+"\\"+file_load_name+".pkl", "rb") as pklfile:
            data_dict = pkl.load(pklfile)
            ds_ndim_test, likelihoods, models = data_dict["ds_inst"], data_dict["likelihoods_glob"], data_dict["models_glob"]

    if verbose:
        print("Training global GP")

    for model_idx in range(len(models)):
        models[model_idx].eval()
        likelihoods[model_idx].eval()
    res_input_dim = ds_ndim_test.input_dims
    res_output_dim = ds_ndim_test.output_dims
    global_gp_wrapped = MultidimGPR_Callback('f', likelihoods, models,
                                             state_dim=res_input_dim, output_dim=res_output_dim, opts={"enable_fd": True})

    if viz:
        if ax is None:
            # First half of plots shows the true mean function and second half shows the GP learnt mean function
            fig, axes = plt.subplots(1, res_output_dim*2, figsize=(10, 13))
            ds_ndim_test.viz_outputs_2d(fineness_param=fineness_param, ax=axes[:res_output_dim])
            ax = axes[res_output_dim:]
        else:
            # ax = ax[-1]
            assert len(ax) == res_output_dim, "The number of axes passed must be the same as the number of residual output dims %s but got %s" % (res_output_dim, len(ax))
        observed_preds = []
        fine_grid = ds_ndim_test.generate_fine_grid(fineness_param=fineness_param)
        if return_error_attrs:
            mean_error_list = []
        for idx in range(len(models)):
            likelihood, model = likelihoods[idx], models[idx]
            observed_pred = likelihood(model(GPR_Callback.preproc(fine_grid.squeeze().T)))
            if return_error_attrs:
                errors = np.zeros([fine_grid.shape[-1]])
            # Check to ensure the callback method is working as intended
            with torch.no_grad():
                # callback sparsity is 1 sample at a time so need to iterate through all 1 at a time
                for sample_idx in range(fine_grid.shape[-1]):
                    sample = fine_grid[:, sample_idx]
                    residual_mean, *residual_covs = global_gp_wrapped(sample)
                    non_callback_mean = observed_pred.mean.numpy()[sample_idx]
                    true_mean = ds_ndim_test.generate_outputs(input_arr=torch.from_numpy(np.array(sample, ndmin=2).T), no_noise=True)
                    errors = None
                    # if return_error_attrs:
                    #     errors[sample_idx] = np.abs(true_mean - non_callback_mean)
                    # assert np.abs(residual_mean[:, idx] - non_callback_mean) <= 1e-3, \
                    #     "GP output mean (%s) and non-callback residual mean (%s) don't match: " % (residual_mean[:, idx], non_callback_mean)
            if return_error_attrs:
                mean_error_list.append({"test_samples": fine_grid, "errors": errors})
            observed_preds.append(observed_pred)
        idx = -1
        colours = ['r', 'g', 'b']
        with torch.no_grad():
            for observed_pred in observed_preds:
                idx += 1
                if ds_ndim_test.input_dims == 2:
                    ax[idx].scatter3D(fine_grid[0, :], fine_grid[1, :],
                                      observed_pred.mean.numpy(), color=colours[idx])
                elif ds_ndim_test.input_dims == 1:
                    plot_uncertainty_bounds_1d(observed_pred, fine_grid, ax[0], colours, idx)
                    # ax[idx].scatter(fine_grid[0, :], observed_pred.mean.numpy(), color=colours[idx])
                else:
                    raise NotImplementedError("Only 1D and 2D inputs are supported")
    if return_error_attrs:
        return global_gp_wrapped, baseline_trained_covs, mean_error_list
    else:
        return global_gp_wrapped


def train_global_n_pw(ds_inst_in, regions, viz=True, verbose=True, return_error_attrs=False,
                      seed=None, fineness_param=(20, 20), num_iter=50, true_only=False,
                      terminate_by_change=False, save_to_file=False, load_from_file=False, file_save_name=None, file_load_name=None):

    pw_ops = construct_gp_wrapped_piecewise(ds_inst_in, regions, viz=viz, fineness_param=fineness_param, verbose=verbose,
                                            return_error_attrs=return_error_attrs, seed=seed,
                                            true_only=true_only, num_iter=num_iter, terminate_by_change=terminate_by_change,
                                            save_to_file=save_to_file, load_from_file=load_from_file,
                                            file_save_name=file_save_name, file_load_name=file_load_name)

    if return_error_attrs:
        piecewise_gp_wrapped, gp_ds_inst, axes, fig, pw_covs, pw_mean_errors, pw_cov_list = pw_ops
    else:
        if not viz:
            piecewise_gp_wrapped, gp_ds_inst = pw_ops
        else:
            piecewise_gp_wrapped, gp_ds_inst, axes, fig = pw_ops

    glob_ops = construct_gp_wrapped_global(ds_inst_in=gp_ds_inst, ax=(None if not viz else [axes[-1]]),
                                           viz=viz, verbose=verbose,
                                           return_error_attrs=return_error_attrs, fineness_param=fineness_param,
                                           num_iter=num_iter, terminate_by_change=terminate_by_change,
                                           save_to_file=save_to_file, load_from_file=load_from_file,
                                           file_save_name=file_save_name, file_load_name=file_load_name)
    if return_error_attrs:
        global_gp_wrapped, glob_covs, glob_mean_errors, glob_cov_list = glob_ops
    else:
        global_gp_wrapped = glob_ops

    for ax in axes:
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_xlim([-1, 1])

    plt.savefig('gp_inp_overlap_base.svg', format='svg', dpi=300)
    plt.show()

    if return_error_attrs:
        ret_list = [piecewise_gp_wrapped, global_gp_wrapped, gp_ds_inst,
                    pw_covs, glob_covs, pw_mean_errors, glob_mean_errors, pw_cov_list, glob_cov_list]
        return ret_list
    else:
        return piecewise_gp_wrapped, global_gp_wrapped, gp_ds_inst
