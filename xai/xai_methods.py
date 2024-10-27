import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from helper_functions import *
from captum.attr import GradientShap, DeepLiftShap, LRP, IntegratedGradients, LimeBase, Lime
from src.models.models import Vanillix, Varix


def captum_importance_values(
    run_id,
    model_type="varix",
    data_types="RNA",
    dimension=0,
    latent_space_explain=False,
    xai_method="deepshap",
    visualize=True,
    return_delta=True,
):
    """
    Function returns attribution values according to parameters specified.
    :param run_id: id to identify model
    :param model_type: 'AE', 'VAE' supported
    :param data_types: 'RNA' etc. TBD
    :param dimension: if latent_space_explain is True, refers to latent dimension; else it refers to feature of model output
    :param latent_space_explain: if True, attribution scores are calculated for a dimension in latent space; else for model output
    :param xai_method: 'deepshap', 'gradientshap', 'lrp' supported
    :param visualize: plots shap.summary_plot of 10 top features
    :param return_delta: if True, returns attributions and delta values, else only attributions
    """

    # get model and data information related to run_id
    model_type = model_type.lower()
    state_dict = get_state_dict(run_id, data_types=data_types, model_type=model_type)

    config_data = get_config(run_id)

    input_data = get_interim_data(run_id, model_type)
    feature_names = input_data.columns.tolist()

    # quick fix solution only
    feature_names_final = []
    for name in feature_names:
        if 'ENSG' in name:
            processed_name = name.replace('RNA_', '')  # Remove the 'RNA_' prefix
            feature_names_final.append(processed_name)
        else:
            feature_names_final.append(name)

    input_dim = get_input_dim(state_dict)
    config_latent_dim = config_data["LATENT_DIM_FIXED"]
    feature_num = get_feature_num(run_id)

    np.random.seed(4598)
    background_tensor, test_tensor = get_random_data_split_tensors(
        input_data, background_n=500, test_n=50
    )

    # print(f"Background tensor shape: {background_tensor.shape}")
    # print(f"Test tensor shape: {test_tensor.shape}")
    # print(f"Input dimension: {input_dim}")

    if model_type == "varix":
        model = Varix(input_dim, config_latent_dim)
        model_wrapper = VAEWrapper(model)
        model_encoder_dim = VAEEncoderSingleDim(model, dimension)

    elif model_type == "vanillix":
        model = Vanillix(input_dim, config_latent_dim)
        model_wrapper = AEWrapper(model)
        model_encoder_dim = AEEncoderSingleDim(model, dimension)

    else:
        print(
            "{} is not a valid model type, please choose Varix or Vanillix instead.".format(
                model_type
            )
        )
        sys.exit()

    model.load_state_dict(state_dict)

    if latent_space_explain:
        model = model_encoder_dim
    else:
        model = model_wrapper

    model.eval()

    if xai_method == "deepshap":
        dl = DeepLiftShap(model)
        attributions, delta = dl.attribute(
            inputs=test_tensor,
            baselines=background_tensor,
            return_convergence_delta=True,
            target=dimension if not latent_space_explain else None
        )

    elif xai_method == "gradientshap":
        gradient_shap = GradientShap(model)
        attributions, delta = gradient_shap.attribute(
            inputs=test_tensor,
            baselines=background_tensor,
            return_convergence_delta=True,
            target=dimension if not latent_space_explain else None
        )

    elif xai_method == "integrated_gradients":
        integrated_gradients = IntegratedGradients(model)

        #### TODO
        random_indices = torch.randperm(background_tensor.shape[0])[:background_tensor.shape[0] // 10] # same amount as input tensor
        downsampled_tensor = background_tensor[random_indices]
        ####

        attributions, delta = integrated_gradients.attribute(
            inputs=test_tensor,
            baselines=downsampled_tensor,
            return_convergence_delta=True,
            target=dimension if not latent_space_explain else None
        )

    elif xai_method == "lime":
        lime = Lime(model)
        attributions = lime.attribute(
            inputs=test_tensor[0].unsqueeze(0), #### TODO
            target=dimension if not latent_space_explain else None
        )

    elif xai_method == "lrp":
        lrp = LRP(model)
        attributions, delta = lrp.attribute(
            inputs=test_tensor,
            return_convergence_delta=True,
            target=dimension if not latent_space_explain else None
        )

    else:
        print(
            "{} is not a valid method type, please choose deepshap, integrated_gradients, gradientshap, or lrp instead.".format(
                model_type
            )
        )
        sys.exit()

    if visualize:
        if xai_method != 'lime':
            shap.summary_plot(
                attributions.detach().numpy(),
                test_tensor,
                feature_names=feature_names_final,
                max_display=10,
            )

    if return_delta:
        print('Attribution values: ', attributions)
        print('Delta values: ', delta)
        print('Max absolute delta: {:.4f}'.format(tf.reduce_max(tf.abs(delta)).numpy()))
        return attributions, delta
    else:
        print('Attribution values: ', attributions)
        print('Max absolute attirbution: {:.4f}'.format(tf.reduce_max(attributions).numpy()))
        return attributions


def shap_importance_values(
    run_id,
    model_type="varix",
    dimension=0,
    latent_space_explain=False,
    xai_method="deepshap",
    visualize=True,
    return_delta=True,
):
    """
    Function returns attribution values according to parameters specified.
    :param run_id: id to identify model
    :param model_type: 'AE', 'VAE' supported
    :param dimension: if latent_space_explain is True, refers to latent dimension; else it refers to feature of model output
    :param latent_space_explain: if True, attribution scores are calculated to a dimension in latent space; else for model output
    :param xai_method: 'deepshap', 'gradientshap', 'kernelshap' supported
    :param visualize: plots shap.summary_plot of 10 top features
    :param return_delta: if True, returns attributions and delta values, else only attributions; supported for DeepShap and KernelShap
                         TODO return delta values instead of assertion error
    """

    # get model and data information related to run_id
    model_type = model_type.lower()
    state_dict = get_state_dict(run_id, model_type=model_type)

    config_data = get_config(run_id)
    input_data = get_processed_data(run_id)
    feature_names = input_data.columns.tolist()

    #####
    input_dim = get_input_dim(state_dict)
    config_latent_dim = config_data["LATENT_DIM_FIXED"]

    background_tensor, test_tensor = get_random_data_split_tensors(
        input_data, background_n=500, test_n=50
    )
    # for KernelExplainer:
    background_df, test_df = get_random_data_split_arrays(
        input_data, background_n=10, test_n=1
    )

    if model_type == "varix":
        model = Varix(input_dim, config_latent_dim)
        model_wrapper = VAEWrapper(model)
        model_encoder_dim = VAEEncoderSingleDim(model, dimension)

    elif model_type == "vanillix":
        model = Vanillix(input_dim, config_latent_dim)
        model_wrapper = AEWrapper(model)
        model_encoder_dim = AEEncoderSingleDim(model, dimension)

    else:
        print(
            "{} is not a valid model type, please choose Varix or Vanillix instead.".format(
                model_type
            )
        )
        sys.exit()

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        if latent_space_explain:
            model = model_encoder_dim
        else:
            model = model_wrapper

    if xai_method == "deepshap":
        e = shap.DeepExplainer(model, background_tensor)
        attributions = e.shap_values(test_tensor, check_additivity=return_delta)

    elif xai_method == "gradientshap":
        e = shap.GradientExplainer(model, background_tensor)
        attributions = e.shap_values(test_tensor)

    elif xai_method == "kernelshap":

        def pred(input_values):
            with torch.no_grad():
                # df -> torch tensor
                input_values = torch.tensor(input_values, dtype=torch.float32)
                output = model(input_values)
                return output.numpy()

        e = shap.KernelExplainer(pred, background_df)
        attributions = e.shap_values(test_df, check_additivity=return_delta)
        # test_data_numpy = test_df.to_numpy()
        # shap.summary_plot(shap_values, test_data_numpy, max_display=10)

    else:
        print(
            "{} is not a valid method type, please choose deepshap, gradientshap, or kernelshap instead.".format(
                model_type
            )
        )
        sys.exit()

    if visualize:
        if xai_method == "kernelshap":
            shap.summary_plot(
                attributions,
                test_df.to_numpy(),
                feature_names=feature_names,
                max_display=10,
            )
        else:
            shap.summary_plot(
                attributions, test_tensor, feature_names=feature_names, max_display=10
            )

    return attributions
