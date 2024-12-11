import glob
import re
import json
from helper_functions import *
from collections import Counter
from xai_methods import captum_importance_values


def disease_results_using_model_aggr(xai_method='deepliftshap', beta=0.01, top_n=15):

    beta001_ids = ['base1', 'base2', 'base3', 'base4', 'base5', 'base6', 'base7', 'base8', 'base9', 'base10']
    beta1_ids = ['b1']
    beta001_ids = ['base2']

    if beta == 0.01:
        model_ids = beta001_ids
    elif beta == 1:
        model_ids = beta1_ids
    else:
        print("{} is not a valid beta value, please choose 0.01 or 1 instead.".format(beta))
        sys.exit()

    aggregated_attributions = dict()

    for run_id in model_ids:
        attribution_values = captum_importance_values(
            run_id=run_id,
            data_types='rna',
            model_type='varix',
            dimension=get_best_dimension_cf(run_id),
            latent_space_explain=True,
            xai_method=xai_method,
            visualize=False,
            return_delta=False
        )

        attribution_dict = attribution_per_feature(
            attribution_values,
            get_interim_data(run_id, 'varix')
        )

        for feature, importance in attribution_dict.items():
            if feature not in aggregated_attributions:
                aggregated_attributions[feature] = importance
            else:
                aggregated_attributions[feature] += importance

    for feature in aggregated_attributions:
        aggregated_attributions[feature] /= len(model_ids)

    # Convert aggregated_attributions back to a tensor-like format
    attributions_tensor = torch.tensor(list(aggregated_attributions.values()))
    feature_names = list(aggregated_attributions.keys())

    bar_plot_top_features(attributions_tensor, feature_names, top_n=top_n, xai_method=xai_method)

    sorted_features = sorted(aggregated_attributions.items(), key=lambda x: x[1], reverse=True)
    top_features = [feature[0] for feature in sorted_features[:top_n]]

    return top_features


def feature_overlap(dls_list, lime_list, ig_list):

    overlap_1_2 = set(dls_list) & set(lime_list)
    overlap_1_3 = set(dls_list) & set(ig_list)
    overlap_2_3 = set(lime_list) & set(ig_list)
    overlap_all = set(dls_list) & set(lime_list) & set(ig_list)

    overlapping_features = {
        "dls_lime": list(overlap_1_2),
        "dls_ig": list(overlap_1_3),
        "lime_ig": list(overlap_2_3),
        "all_explainers": list(overlap_all),
    }

    return overlapping_features


deeplift_top15 = disease_results_using_model_aggr(xai_method='deepliftshap', top_n=15)
lime_top15 = disease_results_using_model_aggr(xai_method='lime', top_n=15)
ig_top15 = disease_results_using_model_aggr(xai_method='integrated_gradients', top_n=15)

overlaps = feature_overlap(deeplift_top15, lime_top15, ig_top15)
print(overlaps)

gene_metadata = get_cf_metadata(deeplift_top15)
print(gene_metadata)
