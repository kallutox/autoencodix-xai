from helper_functions import *
from xai_methods import captum_importance_values
warnings.filterwarnings("ignore")


def disease_results_using_model_aggr(xai_method='deepliftshap', barplot=True, beeswarmplot=False, beta=0.01, top_n=15):
    beta001_ids = [f'cf_001_{i}' for i in range(1, 11)]
    beta1_ids = [f'cf_1_{i}' for i in range(1, 11)]

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

    gene_metadata = get_cf_metadata(all_features)
    gene_names = [gene_metadata[id] for id in all_features if id in gene_metadata]

    with open(output_file, "w") as f:
        f.write("Gene Name,Score\n")
        for gene_name, score in zip(gene_names, all_attribution_values):
            f.write(f"{gene_name},{score:.4f}\n")

    return top_features, all_features, all_attribution_values


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


def run_cf_analysis(bar_plot=True, histogram=True, venn_diagram=True, beeswarm_plot=False, beta=0.01):
    print(f"Calculating DeepLiftShap attributions now.")
    dls_top15, dls_all, dls_attr = disease_results_using_model_aggr(xai_method='deepliftshap', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta)
    print(f"\nCalculating LIME attributions now.")
    lime_top15, lime_all, lime_attr = disease_results_using_model_aggr(xai_method='lime', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta)
    print(f"\nCalculating Integrated Gradients attributions now.")
    ig_top15, ig_all, ig_attr = disease_results_using_model_aggr(xai_method='integrated_gradients', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta)

    feature_overlap(dls_top15, lime_top15, ig_top15, dataset='cf')

    # plot venn diagram
    if venn_diagram:
        plot_venn_diagram(dls_top15, lime_top15, ig_top15, beta, 15, show=False, dataset='cf')

        dls_top100 = dls_all[:100]
        lime_top100 = lime_all[:100]
        ig_top100 = ig_all[:100]

        print(f"DLS Top 100: {len(dls_top100)}")
        print(f"LIME Top 100: {len(lime_top100)}")
        print(f"IG Top 100: {len(ig_top100)}")

        plot_venn_diagram(dls_top100, lime_top100, ig_top100, beta, 100, show=False, dataset='cf')

    if histogram:
        plot_attribution_histogram(dls_attr, beta, xai_method='deepliftshap', show=True, dataset='cf')
        plot_attribution_histogram(lime_attr, beta, xai_method='lime', show=True, dataset='cf')
        plot_attribution_histogram(ig_attr, beta, xai_method='integrated_gradients', show=True, dataset='cf')

    # optional
    # gene_metadata = get_cf_metadata(ig_top15)
    # print(gene_metadata)

    get_cftr_rank(dls_all)
    get_cftr_rank(lime_all)
    get_cftr_rank(ig_all)
