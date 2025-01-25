from helper_functions import *
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
from xai_methods import captum_importance_values
import warnings

# Suppress all warnings (due to gene name retrieval or LIME)
warnings.filterwarnings("ignore")


def disease_results_using_model_aggr(cancer_type, xai_method='deepliftshap', barplot=True, beeswarmplot=False, beta=0.01, top_n=15):

    #beta001_ids = ['tcga_001_1', 'tcga_001_2', 'tcga_001_3', 'tcga_001_4', 'tcga_001_5', 'tcga_001_6', 'tcga_001_7',
    #              'tcga_001_8', 'tcga_001_9', 'tcga_001_10']
    beta1_ids = ['tcga_1_1', 'tcga_1_2', 'tcga_1_3', 'tcga_1_4', 'tcga_1_5', 'tcga_1_6', 'tcga_1_7', 'tcga_1_8',
                 'tcga_1_9', 'tcga_1_10']
    beta001_ids = ['tcga_001_1', 'tcga_001_2']

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
            data_types='METH_MUT_RNA',
            model_type='varix',
            data_set='tcga',
            cancer_type=cancer_type,
            dimension=get_best_dimension_by_cancer_lda(run_id, cancer_type),
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

    #print("attribution tensor: ", attributions_tensor)
    #print("attribution tensor length: ", len(attributions_tensor))

    gene_metadata = get_tcga_metadata(feature_names)
    gene_names = [gene_metadata[id] for id in feature_names if id in gene_metadata]

    if barplot:
        bar_plot_top_features(attributions_tensor, gene_names, dataset='tcga', top_n=top_n, xai_method=xai_method,
                              beta=beta_val, n=top_n, cancer_type=cancer_type, )

    sorted_features = sorted(aggregated_attributions.items(), key=lambda x: x[1], reverse=True)
    all_features = [feature[0] for feature in sorted_features]
    all_attribution_values = [feature[1] for feature in sorted_features]

    top_features = [feature[0] for feature in sorted_features[:top_n]]

    output_dir = "tcga_reports"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"all_features_{cancer_type}_{beta_val}_{xai_method}.txt")
    with open(output_file, "w") as f:
        for feature in all_features:
            f.write(f"{feature}\n")

    return top_features, all_features, all_attribution_values


def run_tcga_analysis(cancer_type, bar_plot=True, histogram=True, venn_diagram=True, beeswarm_plot=False, beta=0.01, show=True):
    print(f"Calculating DeepLiftShap attributions now.")
    dls_top15, dls_all, dls_attr = disease_results_using_model_aggr(xai_method='deepliftshap', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta, cancer_type=cancer_type)
    print("\nTop 15 features (DeepLiftShap):")
    for feature, score in zip(dls_top15, dls_attr[:15]):
        print(f"Feature: {feature}, Score: {score:.2f}")

    print(f"\nCalculating LIME attributions now.")
    lime_top15, lime_all, lime_attr = disease_results_using_model_aggr(xai_method='lime', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta, cancer_type=cancer_type)
    print("\nTop 15 features (LIME):")
    for feature, score in zip(lime_top15, lime_attr[:15]):
        print(f"Feature: {feature}, Score: {score:.2f}")

    print(f"\nCalculating Integrated Gradients attributions now.")
    ig_top15, ig_all, ig_attr = disease_results_using_model_aggr(xai_method='integrated_gradients', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta, cancer_type=cancer_type)
    print("\nTop 15 features (Integrated Gradients):")
    for feature, score in zip(ig_top15, ig_attr[:15]):
        print(f"Feature: {feature}, Score: {score:.2f}")

    feature_overlap(dls_top15, lime_top15, ig_top15)

    # plot venn diagram
    if venn_diagram:

        plot_venn_diagram(dls_top15, lime_top15, ig_top15, beta, 15, show=show, dataset='tcga', cancer_type=cancer_type)

        dls_top100 = dls_all[:100]
        lime_top100 = lime_all[:100]
        ig_top100 = ig_all[:100]

        plot_venn_diagram(dls_top100, lime_top100, ig_top100, 0.01, 100, show=show, dataset='tcga', cancer_type=cancer_type)

    # plot histogram
    if histogram:
        plot_attribution_histogram(dls_attr, beta, xai_method='deepliftshap', show=show, dataset='tcga',
                                   cancer_type=cancer_type)
        plot_attribution_histogram(lime_attr, beta, xai_method='lime', show=show, dataset='tcga', cancer_type=cancer_type)
        plot_attribution_histogram(ig_attr, beta, xai_method='integrated_gradients', show=show, dataset='tcga', cancer_type=cancer_type)


#disease_results_using_model_aggr(xai_method='deepliftshap', barplot=True, beeswarmplot=False, top_n=15, beta=0.01, cancer_type='KIRC')
#run_tcga_analysis(cancer_type='LUAD', beta=0.01, bar_plot=True, venn_diagram=True, show=False)

# cancer_list = ['CHOL', 'OV', 'CESC', 'UCS', 'PRAD', 'DLBC', 'PCPG', 'KIRC', 'UVM', 'BLCA', 'STAD', 'UCEC', 'LGG',
#                'KIRP', 'READ', 'COAD', 'LIHC', 'LUAD', 'GBM', 'THCA', 'PAAD', 'SARC', 'MESO', 'ACC', 'HNSC', 'ESCA',
#                'LUSC', 'SKCM', 'KICH', 'BRCA', 'THYM', 'LAML', 'TGCT']

# get results for single cancer
cancer = 'LAML'
run_tcga_analysis(cancer_type=cancer, beta=0.01, bar_plot=True, venn_diagram=True, show=False)

# get results for all 5 cancers
cancer_list = ['LAML', 'PRAD', 'THCA', 'OV', 'LGG']

# for cancer in cancer_list:
#     run_tcga_analysis(cancer_type=cancer, beta=0.01, bar_plot=True, venn_diagram=True, show=False)
#     print(f"{cancer} done! \n\n")
