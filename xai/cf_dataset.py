import glob
import re
import json
import sys

from helper_functions import *
from collections import Counter
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
from xai_methods import captum_importance_values


def disease_results_using_model_aggr(xai_method='deepliftshap', barplot=True, beeswarmplot=False, beta=0.01, top_n=15):

    beta001_ids = ['cf_001_1', 'cf_001_2', 'cf_001_3', 'cf_001_4', 'cf_001_5', 'cf_001_6', 'cf_001_7', 'cf_001_8', 'cf_001_9', 'cf_001_10']
    beta1_ids = ['cf_1_1', 'cf_1_2', 'cf_1_3', 'cf_1_4', 'cf_1_5', 'cf_1_6', 'cf_1_7', 'cf_1_8', 'cf_1_9', 'cf_1_10']
    #beta001_ids = ['cf_001_1', 'cf_001_2']

    if beta == 0.01:
        model_ids = beta001_ids
        beta_val = '001'
    elif beta == 1:
        model_ids = beta1_ids
        beta_val = '1'
    else:
        print("{} is not a valid beta value, please choose 0.01 or 1 instead.".format(beta))
        sys.exit()

    aggregated_attributions = dict()

    for run_id in model_ids:
        attribution_values = captum_importance_values(
            run_id=run_id,
            data_types='rna',
            model_type='varix',
            data_set='cf',
            dimension=get_best_dimension_cf(run_id),
            latent_space_explain=True,
            xai_method=xai_method,
            visualize=beeswarmplot,
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

    gene_metadata = get_cf_metadata(feature_names)
    gene_names = [gene_metadata[id]['feature_name'] for id in feature_names if id in gene_metadata]

    if barplot:
        bar_plot_top_features(attributions_tensor, gene_names, dataset='cf', top_n=top_n, xai_method=xai_method,
                              beta=beta_val, n=top_n)

    sorted_features = sorted(aggregated_attributions.items(), key=lambda x: x[1], reverse=True)
    all_features = [feature[0] for feature in sorted_features]
    all_attribution_values = [feature[1] for feature in sorted_features]

    top_features = [feature[0] for feature in sorted_features[:top_n]]

    output_dir = "cf_reports"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"all_features_{beta_val}_{xai_method}.txt")
    with open(output_file, "w") as f:
        for feature in all_features:
            f.write(f"{feature}\n")

    return top_features, all_features, all_attribution_values


def feature_overlap(dls_list, lime_list, ig_list):
    gene_metadata_dls = get_cf_metadata(dls_list)
    gene_names_dls = [gene_metadata_dls[id]['feature_name'] for id in dls_list if id in gene_metadata_dls]

    gene_metadata_lime = get_cf_metadata(lime_list)
    gene_names_lime = [gene_metadata_lime[id]['feature_name'] for id in lime_list if id in gene_metadata_lime]

    gene_metadata_ig = get_cf_metadata(ig_list)
    gene_names_ig = [gene_metadata_ig[id]['feature_name'] for id in ig_list if id in gene_metadata_ig]

    overlap_1_2 = set(gene_names_dls) & set(gene_names_lime)
    overlap_1_3 = set(gene_names_dls) & set(gene_names_ig)
    overlap_2_3 = set(gene_names_lime) & set(gene_names_ig)
    overlap_all = set(gene_names_dls) & set(gene_names_lime) & set(gene_names_ig)

    # Calculate unique features for each explainer
    unique_dls = set(gene_names_dls) - (set(gene_names_lime) | set(gene_names_ig))
    unique_lime = set(gene_names_lime) - (set(gene_names_dls) | set(gene_names_ig))
    unique_ig = set(gene_names_ig) - (set(gene_names_dls) | set(gene_names_lime))


    # Print the overlaps and unique features
    print("Overlapping Features:")
    print(f"DLS & LIME: {list(overlap_1_2)}")
    print(f"DLS & IG: {list(overlap_1_3)}")
    print(f"LIME & IG: {list(overlap_2_3)}")
    print(f"All Explainers: {list(overlap_all)}")

    print("\nUnique Features:")
    print(f"Unique to DLS: {list(unique_dls)}")
    print(f"Unique to LIME: {list(unique_lime)}")
    print(f"Unique to IG: {list(unique_ig)}")


def get_cftr_rank(feature_list):
    cftr_id = 'ENSG00000001626'

    if cftr_id in feature_list:
        position = feature_list.index(cftr_id)
        print(f"\nCFTR gene is at position {position+1} in the list.")
    else:
        position = None
        print(f"\nCFTR gene is not in the list.")

    return position


def get_gene_rank(ensembl_id, feature_list):
    if ensembl_id in feature_list:
        position = feature_list.index(ensembl_id)
        print(f"\n{ensembl_id} is at position {position+1} in the list.")
    else:
        position = None
        print(f"\n{ensembl_id} is not in the list.")

    return position


def load_sorted_features(xai_method, directory="cf_reports"):
    """
    Load sorted feature list for a specific XAI method.

    Args:
        xai_method (str): XAI method name (e.g., 'deepliftshap', 'lime', 'integrated_gradients').
        directory (str): Directory where the files are stored.

    Returns:
        list: Sorted feature list for the specified XAI method.
    """
    file_path = os.path.join(directory, f"sorted_features_{xai_method}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    else:
        print(f"File for {xai_method} not found in {directory}.")
        return []


def print_overlap_only():
    deeplift_top15 = disease_results_using_model_aggr(xai_method='deepliftshap', top_n=15)
    lime_top15 = disease_results_using_model_aggr(xai_method='lime', top_n=15)
    ig_top15 = disease_results_using_model_aggr(xai_method='integrated_gradients', top_n=15)

    print(feature_overlap(deeplift_top15, lime_top15, ig_top15))


def plot_venn_diagram(dls_set, lime_set, ig_set, beta, n):
    """
    Plots a Venn diagram showing overlaps between three sets of features.
    Only colors regions with actual overlaps to maintain visual simplicity.
    """
    # Convert lists to sets
    dls_set = set(dls_set)
    lime_set = set(lime_set)
    ig_set = set(ig_set)

    # Set up the figure
    plt.figure(figsize=(6, 6))

    # Create the Venn diagram
    venn = venn3(
        [dls_set, lime_set, ig_set],
        ('DeepLiftShap', 'LIME', 'Integrated Gradients')
    )

    # Define colors for each region
    patch_colors = {
        '100': "#4354b5",  # DLS only (Blue)
        '010': "#43a2b5",  # LIME only (Teal)
        '001': "#43b582",  # IG only (Green)
        '110': "#4a74d6",  # DLS & LIME (Blue + Teal blend)
        '101': "#4a7c93",  # DLS & IG (Blue + Green blend)
        '011': "#43b5a8",  # LIME & IG (Teal + Green blend)
        '111': "#7a87cc"   # All three (Purple)
    }

    # Apply colors to the patches
    for patch_id, color in patch_colors.items():
        patch = venn.get_patch_by_id(patch_id)
        if patch is not None:
            patch.set_color(color)
            patch.set_alpha(0.6)  # Set transparency for the color

    plt.title(f"Top {n} Features Overlap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs("cf_reports", exist_ok=True)
    plt.savefig(f"cf_reports/venn_diagram_top{n}_{beta}.png")
    plt.show()


def plot_attribution_histogram(attribution_values, beta, xai_method='deepliftshap'):
    plt.figure(figsize=(10, 6))
    plt.hist(attribution_values, bins=50, color="#43a2b5", alpha=0.8)

    if xai_method == 'deepliftshap':
        xai_label = 'DeepLiftShap'
    elif xai_method == 'lime':
        xai_label = 'LIME'
    elif xai_method == 'integrated_gradients':
        xai_label = 'Integrated Gradients'
    else:
        print("{} is not a valid method, please choose deepliftshap, lime or integrated_gradient instead.".format(xai_method))
        sys.exit()

    plt.xlabel("Attribution Scores", fontsize=12)
    plt.ylabel("Number of Features", fontsize=12)
    plt.title(f" {xai_label} - Distribution of Attribution Scores", fontsize=14, fontweight="bold")

    if beta == 0.01:
        beta_val = '001'
    elif beta == 1:
        beta_val = '1'
    else:
        print('beta invalid.')
        sys.exit()

    os.makedirs("cf_reports", exist_ok=True)
    plt.savefig(f"cf_reports/attribution_histogram_{xai_method}_{beta_val}.png")
    plt.tight_layout()
    plt.show()


def run_cf_analysis(bar_plot=True, histogram=True, venn_diagram=True, beeswarm_plot=False, beta=0.01):
    dls_top15, dls_all, dls_attr = disease_results_using_model_aggr(xai_method='deepliftshap', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta)
    lime_top15, lime_all, lime_attr = disease_results_using_model_aggr(xai_method='lime', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta)
    ig_top15, ig_all, ig_attr = disease_results_using_model_aggr(xai_method='integrated_gradients', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta)

    feature_overlap(dls_top15, lime_top15, ig_top15)

    # plot venn diagram
    if venn_diagram:
        if beta == 0.01:
            beta_val = '001'
        else:
            beta_val = '1'

        plot_venn_diagram(dls_top15, lime_top15, ig_top15, beta_val, 15)

        dls_top100 = dls_all[:100]
        lime_top100 = lime_all[:100]
        ig_top100 = ig_all[:100]

        print(f"DLS Top 100: {len(dls_top100)}")
        print(f"LIME Top 100: {len(lime_top100)}")
        print(f"IG Top 100: {len(ig_top100)}")

        plot_venn_diagram(dls_top100, lime_top100, ig_top100, beta_val, 100)

    if histogram:
        plot_attribution_histogram(dls_attr, beta, xai_method='deepliftshap')
        plot_attribution_histogram(lime_attr, beta, xai_method='lime')
        plot_attribution_histogram(ig_attr, beta, xai_method='integrated_gradients')

    # optional
    # gene_metadata = get_cf_metadata(ig_top15)
    # print(gene_metadata)

    get_cftr_rank(dls_all)
    get_cftr_rank(lime_all)
    get_cftr_rank(ig_all)


#disease_results_using_model_aggr()
run_cf_analysis(beta=1)
# topf, allf, allattr = disease_results_using_model_aggr(xai_method='deepliftshap', barplot=True, top_n=15, beta=0.01)
# plot_attribution_histogram(allattr, xai_method='deepliftshap')

# deepliftshap_features = load_sorted_features("deepliftshap")
# lime_features = load_sorted_features("lime")
# integrated_gradients_features = load_sorted_features("integrated_gradients")
#
# gene_id = 'ENSG00000170421'
# get_gene_rank(gene_id, deepliftshap_features)
# get_gene_rank(gene_id, lime_features)
# get_gene_rank(gene_id, integrated_gradients_features)

