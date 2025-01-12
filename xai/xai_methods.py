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
    data_set="cf",
    dimension=0,
    latent_space_explain=False,
    xai_method="deepliftshap",
    visualize=True,
    return_delta=True,
    random_seed=None
):
    """
    Function returns attribution values according to parameters specified.
    :param run_id: id to identify model
    :param model_type: 'AE', 'VAE' supported
    :param data_types: 'RNA' etc. TBD
    :param dimension: if latent_space_explain is True, refers to latent dimension; else it refers to feature of model output
    :param latent_space_explain: if True, attribution scores are calculated for a dimension in latent space; else for model output
    :param xai_method: 'deepliftshap', 'gradientshap', 'lrp' supported
    :param visualize: plots shap.summary_plot of 10 top features
    :param return_delta: if True, returns attributions and delta values, else only attributions

    Args:
        random_seed:
    """

    # get model and data information related to run_id
    model_type = model_type.lower()
    state_dict = get_state_dict(run_id, data_types=data_types, model_type=model_type)
    interim_data = get_interim_data(run_id)
    config_data = get_config(run_id)

    input_data = get_interim_data(run_id, model_type)
    clin_data_cf = get_cf_clin_data()
    feature_names = input_data.columns.tolist()
    feature_names = [name.replace("RNA_", "") for name in feature_names]  # remove RNA_prefix

    # get gene names -- works for cf
    gene_metadata = get_cf_metadata(feature_names)
    gene_names = [gene_metadata[id]['feature_name'] for id in feature_names if id in gene_metadata]

    input_dim = get_input_dim(state_dict)
    config_latent_dim = config_data["LATENT_DIM_FIXED"]

    train_data = get_training_data(run_id, input_data)

    if data_set == "synthetic":
        test_tensor, background_tensor = get_sex_specific_split(
            input_data=train_data, clin_data=clin_data_cf, test_n=150,  ref_n=300, seed=random_seed
            )
    elif data_set == "cf":
        test_tensor, background_tensor = get_cf_specific_split(
            input_data=train_data, clin_data=clin_data_cf, test_n=150, ref_n=300, seed=random_seed
        )
    elif data_set == "tcga":    # TODO
        test_tensor, background_tensor = get_cf_specific_split(
            input_data=train_data, clin_data=clin_data_cf, test_n=150, ref_n=300, seed=random_seed
        )
    else:
        print("No valid data set was chosen, test and reference data will be random.")
        test_tensor, background_tensor = get_random_data_split_tensors(
            train_data, background_n=300, test_n=150, seed=random_seed
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

    if latent_space_explain:
        model = model_encoder_dim
    else:
        model = model_wrapper

    model.eval()

    if xai_method == "deepliftshap":
        dls = DeepLiftShap(model, multiply_by_inputs=True)

        attributions, delta = dls.attribute(
            inputs=test_tensor,
            baselines=background_tensor,
            return_convergence_delta=True,
            target=dimension if not latent_space_explain else None
        )

    elif xai_method == "integrated_gradients":
        integrated_gradients = IntegratedGradients(model, multiply_by_inputs=True)
        mean_baseline = background_tensor.mean(dim=0, keepdim=True)

        attributions, delta = integrated_gradients.attribute(
            inputs=test_tensor,
            baselines=mean_baseline,
            return_convergence_delta=True,
            target=dimension if not latent_space_explain else None
        )

    elif xai_method == "lime":
        lime = Lime(model)
        average_baseline = background_tensor.mean(dim=0)
        summed_attributions = []

        for sample in test_tensor:
            sample_attribution = lime.attribute(
                inputs=sample.unsqueeze(0),
                baselines=average_baseline.unsqueeze(0),
                target=dimension if not latent_space_explain else None
            )
            summed_attributions.append(sample_attribution.squeeze(0))

        stacked_attributions = torch.stack(summed_attributions, dim=0)
        absolute_attributions = stacked_attributions.abs()  # according to van der Linden et al. (2019)
        attributions = absolute_attributions.mean(dim=0)

    elif xai_method == "lrp":
        lrp = LRP(model)
        attributions, delta = lrp.attribute(
            inputs=test_tensor,
            return_convergence_delta=True,
            target=dimension if not latent_space_explain else None
        )

    else:
        print(
            "{} is not a valid method type, please choose deepshap, integrated_gradients, or lime instead.".format(
                xai_method
            )
        )
        sys.exit()

    if visualize:
        if xai_method == 'lime': #or xai_method == 'integrated_gradients'
            shap.summary_plot(
                stacked_attributions.numpy(),
                test_tensor,
                feature_names=gene_names,
                max_display=15
            )
            bar_plot_top_features(stacked_attributions.numpy(), interim_data, top_n=15, xai_method='lime')

        else:
            shap.summary_plot(
                attributions.detach().numpy(),
                test_tensor,
                feature_names=gene_names,
                max_display=15,
            )
            bar_plot_top_features(attributions, interim_data, top_n=15)

            ## for using other shap plots
            # shap_explanation = shap.Explanation(
            #     values=attributions.detach().numpy(),
            #     feature_names=feature_names,
            #     data=test_tensor.numpy()
            # )
            # shap.plots.beeswarm(shap_explanation, max_display=10)
            # shap.plots.heatmap(shap_explanation, max_display=10)
            # shap.plots.bar(shap_explanation, max_display=10)

    # attribution dict with all scores across samples
    if xai_method == 'lime':
        attr_dict_all = attr_dict_all_samples(stacked_attributions, gene_names)
    else:
        attr_dict_all = attr_dict_all_samples(attributions, gene_names)

    if return_delta:
        return attributions, delta
    else:
        return attributions
        # return attributions, attr_dict_all
