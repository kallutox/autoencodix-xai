from helper_functions import *
import matplotlib.pyplot as plt
from xai_methods import captum_importance_values
from scipy.stats import kruskal
import scikit_posthocs as sp

import warnings

# Suppress all warnings (due to gene name retrieval or LIME)
warnings.filterwarnings("ignore")


def disease_results_using_model_aggr(cancer_type, xai_method='deepliftshap', barplot=True, beeswarmplot=False, beta=0.01, top_n=15):

    beta001_ids = ['tcga_001_1', 'tcga_001_2', 'tcga_001_3', 'tcga_001_4', 'tcga_001_5', 'tcga_001_6', 'tcga_001_7',
                   'tcga_001_8', 'tcga_001_9', 'tcga_001_10']
    beta1_ids = ['tcga_1_1', 'tcga_1_2', 'tcga_1_3', 'tcga_1_4', 'tcga_1_5', 'tcga_1_6', 'tcga_1_7', 'tcga_1_8',
                 'tcga_1_9', 'tcga_1_10']

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

    gene_metadata = get_tcga_metadata(all_features)
    gene_names = [gene_metadata[id] for id in all_features if id in gene_metadata]

    with open(output_file, "w") as f:
        f.write("Gene Name,Score\n")
        for gene_name, score in zip(gene_names, all_attribution_values):
            f.write(f"{gene_name},{score:.4f}\n")

    return top_features, all_features, all_attribution_values


def run_tcga_analysis(cancer_type, bar_plot=True, histogram=True, venn_diagram=True, beeswarm_plot=False, beta=0.01, show=True):
    print(f"Calculating DeepLiftShap attributions now.")
    dls_top15, dls_all, dls_attr = disease_results_using_model_aggr(xai_method='deepliftshap', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta, cancer_type=cancer_type)

    dls_gene_metadata = get_tcga_metadata(dls_top15)
    dls_gene_names = [dls_gene_metadata[id] for id in dls_top15 if id in dls_gene_metadata]
    print("\nTop 15 features (DeepLiftShap):")
    for feature, score in zip(dls_gene_names, dls_attr[:15]):
        print(f"Feature: {feature}, Score: {score:.4f}")

    print(f"\nCalculating LIME attributions now.")
    lime_top15, lime_all, lime_attr = disease_results_using_model_aggr(xai_method='lime', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta, cancer_type=cancer_type)

    lime_gene_metadata = get_tcga_metadata(lime_top15)
    lime_gene_names = [lime_gene_metadata[id] for id in lime_top15 if id in lime_gene_metadata]
    print("\nTop 15 features (LIME):")
    for feature, score in zip(lime_gene_names, lime_attr[:15]):
        print(f"Feature: {feature}, Score: {score:.4f}")

    print(f"\nCalculating Integrated Gradients attributions now.")
    ig_top15, ig_all, ig_attr = disease_results_using_model_aggr(xai_method='integrated_gradients', barplot=bar_plot, beeswarmplot=beeswarm_plot, top_n=15, beta=beta, cancer_type=cancer_type)

    ig_gene_metadata = get_tcga_metadata(ig_top15)
    ig_gene_names = [ig_gene_metadata[id] for id in ig_top15 if id in ig_gene_metadata]
    print("\nTop 15 features (Integrated Gradients):")
    for feature, score in zip(ig_gene_names, ig_attr[:15]):
        print(f"Feature: {feature}, Score: {score:.4f}")

    feature_overlap(dls_top15, lime_top15, ig_top15, dataset='tcga')

    # plot venn diagram
    if venn_diagram:

        plot_venn_diagram(dls_top15, lime_top15, ig_top15, beta, 15, show=show, dataset='tcga', cancer_type=cancer_type)

        dls_top100 = dls_all[:100]
        lime_top100 = lime_all[:100]
        ig_top100 = ig_all[:100]

        plot_venn_diagram(dls_top100, lime_top100, ig_top100, beta, 100, show=show, dataset='tcga', cancer_type=cancer_type)

    # plot histogram
    if histogram:
        plot_attribution_histogram(dls_attr, beta, xai_method='deepliftshap', show=show, dataset='tcga',
                                   cancer_type=cancer_type)
        plot_attribution_histogram(lime_attr, beta, xai_method='lime', show=show, dataset='tcga', cancer_type=cancer_type)
        plot_attribution_histogram(ig_attr, beta, xai_method='integrated_gradients', show=show, dataset='tcga', cancer_type=cancer_type)


def get_results_final_cancers(beta):
    #cancer_list = ['LAML', 'PRAD', 'THCA', 'OV', 'LGG']
    cancer_list = ['PRAD', 'THCA']

    for cancer in cancer_list:
        run_tcga_analysis(cancer_type=cancer, beta=beta, bar_plot=True, venn_diagram=True, show=False)
        print(f"{cancer} done! \n\n")


def parse_ranking_file(cancer_type, xai_method, beta):
    if beta == 0.01:
        beta = '001'
    file_path = f"tcga_reports/all_features_{cancer_type}_{beta}_{xai_method}.txt"
    df = pd.read_csv(file_path, skiprows=1, header=None, names=['Feature', 'Score'])

    df['Modality'] = df['Feature'].apply(
        lambda x: 'meth.' if '(meth.)' in x else 'mut.' if '(mut.)' in x else 'RNA'
    )
    df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1
    df_sorted['Method'] = xai_method

    return df_sorted


def parse_all_methods(cancer_type, beta=0.01):
    methods = ['deepliftshap', 'lime', 'integrated_gradients']
    all_data = []

    for method in methods:
        df = parse_ranking_file(cancer_type, method, beta)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def rank_distribution_stats(df_sorted):
    stats = df_sorted.groupby('Modality')['Rank'].agg(['mean', 'median', 'std'])
    stats = stats.round(2)
    return stats


def modality_distribution_stats(df_modality):
    """
    Given a DataFrame filtered to one modality,
    compute summary stats (mean, median, std) of 'Rank' by 'Method'.
    """
    stats = df_modality.groupby('Method')['Rank'].agg(['mean', 'median', 'std'])
    stats = stats.round(2)
    return stats


def top_k_analysis(df_sorted, k_values=[15, 100]):
    results = {}
    for k in k_values:
        top_k_df = df_sorted[df_sorted['Rank'] <= k]
        counts = top_k_df['Modality'].value_counts()
        results[k] = counts
    return results


def rank_based_analysis(cancer_type, xai_method, beta):
    df_sorted = parse_ranking_file(cancer_type, xai_method, beta)

    stats = rank_distribution_stats(df_sorted)
    print(f"\nRank distribution statistics for {xai_method}, beta={beta}")
    print(stats)

    top_k_counts = top_k_analysis(df_sorted)
    for k, counts in top_k_counts.items():
        print(f"\nTop-{k} coverage by modality for {xai_method}, beta={beta}:")
        print(counts)

    return df_sorted, stats, top_k_counts


def plot_rank_distribution(df_sorted, xai_method='LIME'):
    plt.figure(figsize=(8, 6))

    plt.style.use("seaborn-whitegrid")
    boxprops = dict(facecolor="#43a2b5", alpha=0.7, linewidth=1.0)

    ax = sns.boxplot(
        x='Modality',
        y='Rank',
        data=df_sorted,
        boxprops=boxprops,
        linewidth=0.7
    )
    ax.invert_yaxis()

    plt.xlabel("Modality", fontsize=12)
    plt.ylabel("Rank", fontsize=12)
    plt.title(f"Rank Distribution by Modality",
              fontsize=14, fontweight="bold")

    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_rank_distribution_multi(df_dls, df_lime, df_ig, cancer_type, beta):
    df_dls["Method"] = "DeepLiftShap"
    df_lime["Method"] = "LIME"
    df_ig["Method"] = "Integrated Gradients"

    df_all = pd.concat([df_dls, df_lime, df_ig], ignore_index=True)
    df_all["Modality"].replace({"meth.": "Methylation", "mut.": "Mutation"}, inplace=True)
    df_all["Modality"] = pd.Categorical(
        df_all["Modality"],
        categories=["RNA", "Methylation", "Mutation"],
        ordered=True
    )

    plt.figure(figsize=(8, 6))
    plt.style.use("seaborn-whitegrid")

    boxprops = dict(alpha=0.7, linewidth=1.0)
    flierprops = dict(marker='o', markerfacecolor='lightgrey', alpha=0.7, markersize=5, markeredgecolor='none')

    ax = sns.boxplot(
        x="Modality",
        y="Rank",
        hue="Method",
        data=df_all,
        palette=["#4354b5", "#43a2b5", "#43b582"],
        boxprops=boxprops,
        flierprops=flierprops,
        linewidth=0.7
    )

    ax.invert_yaxis()
    legend = ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.7)

    plt.xlabel("Modality", fontsize=12)
    plt.ylabel("Rank", fontsize=12)
    plt.title("Rank Distribution by Modality", fontsize=14, fontweight="bold")

    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if beta == 0.01:
        beta = '001'
    plt.savefig(f"tcga_reports/boxplot_modalities_{cancer_type}_{beta}.png")

    plt.show()


def modality_results_by_cancer(cancer_type, beta):
    df_dls = parse_ranking_file(cancer_type, 'deepliftshap', beta)
    stats = rank_distribution_stats(df_dls)
    print(f"\nRank distribution statistics for DeepLiftShap:")
    print(stats)

    top_k_counts = top_k_analysis(df_dls)
    for k, counts in top_k_counts.items():
        print(f"\nTop-{k} coverage by modality for DeepLiftShap:")
        print(counts)

    df_lime = parse_ranking_file(cancer_type, 'lime', beta)
    stats = rank_distribution_stats(df_lime)
    print(f"\nRank distribution statistics for LIME:")
    print(stats)

    top_k_counts = top_k_analysis(df_lime)
    for k, counts in top_k_counts.items():
        print(f"\nTop-{k} coverage by modality for LIME:")
        print(counts)

    df_ig = parse_ranking_file(cancer_type, 'integrated_gradients', beta)
    stats = rank_distribution_stats(df_ig)
    print(f"\nRank distribution statistics for Integrated Gradients:")
    print(stats)

    top_k_counts = top_k_analysis(df_ig)
    for k, counts in top_k_counts.items():
        print(f"\nTop-{k} coverage by modality for Integrated Gradients:")
        print(counts)

    plot_rank_distribution_multi(df_dls, df_lime, df_ig, cancer_type, beta)


def test_differences_in_ranks(df_sorted):
    """
    Perform a Kruskal-Wallis test within this method's DataFrame
    to see if rank distributions differ across modalities.
    If significant, do post-hoc comparisons (Dunn test).
    """
    # Extract rank distributions for each modality
    modalities = df_sorted['Modality'].unique()
    groups = []
    for m in modalities:
        ranks = df_sorted.loc[df_sorted['Modality'] == m, 'Rank']
        groups.append(ranks)

    # Kruskal-Wallis
    H_stat, pval = kruskal(*groups)
    print(f"Kruskal–Wallis H={H_stat:.3f}, p={pval:.3e}")

    if pval < 0.05:
        # Post-hoc (Dunn) test
        print("Post-hoc Dunn test results (Bonferroni corrected):")
        # We need a "long form" df with columns [Group, Value].
        # Group is the modality, Value is the rank.
        # Then pass to sp.posthoc_dunn
        data_long = df_sorted[['Modality', 'Rank']]
        dunn_res = sp.posthoc_dunn(data_long, val_col='Rank',
                                   group_col='Modality',
                                   p_adjust='holm')
        print(dunn_res)
    else:
        print("No significant difference among modalities.")


def analyze_method_ranks(df_all, method_name):
    """
    Extract ranks for a single method, summarize rank distribution by modality,
    and run Kruskal–Wallis + post-hoc if desired.
    """
    df_method = df_all[df_all['Method'] == method_name].copy()

    # Summaries
    stats = rank_distribution_stats(df_method)
    print(f"\nRank Distribution Stats - {method_name}")
    print(stats)

    # Statistical test
    test_differences_in_ranks(df_method)


def test_differences_in_methods(df_modality):
    """
    Perform a Kruskal–Wallis test comparing rank distributions
    across *methods* for a single modality.
    If significant, run a post-hoc Dunn test.
    """
    methods = df_modality['Method'].unique()
    # Collect the rank distributions, one group per method
    groups = []
    for m in methods:
        ranks = df_modality.loc[df_modality['Method'] == m, 'Rank']
        groups.append(ranks)

    # Kruskal–Wallis
    H_stat, pval = kruskal(*groups)
    print(f"\nKruskal–Wallis H={H_stat:.3f}, p={pval:.3e}")

    # If significant, do post-hoc Dunn test
    if pval < 0.05:
        print("Post-hoc Dunn test results (FDR corrected):")
        # We need a “long form” DataFrame with columns [group_col, val_col]
        data_long = df_modality[['Method', 'Rank']]
        dunn_res = sp.posthoc_dunn(data_long, val_col='Rank',
                                   group_col='Method',
                                   p_adjust='fdr_bh')
        print(dunn_res)
    else:
        print("No significant difference among methods for this modality.")


def analyze_modality_across_methods(df_all, modality):
    """
    1) Filter df_all to just the chosen modality.
    2) Compute stats of rank by Method.
    3) Perform Kruskal–Wallis + post-hoc if needed.
    """
    df_mod = df_all[df_all['Modality'] == modality].copy()

    if df_mod.empty:
        print(f"No data found for modality: {modality}")
        return

    # Summaries
    stats = modality_distribution_stats(df_mod)
    print(f"\nRank Distribution Stats for modality = {modality}")
    print(stats)

    # Statistical test
    test_differences_in_methods(df_mod)


def within_group_stat_test(cancer_type):
    # within method statistical testing of modality ranking
    df_all = parse_all_methods(cancer_type, beta=0.01)
    methods = ['deepliftshap', 'lime', 'integrated_gradients']
    for method in methods:
        print(f"\n\nMethod: {method}")
        analyze_method_ranks(df_all, method)


def between_groups_stat_test(cancer_type, beta):
    df_all = parse_all_methods(cancer_type, beta)
    modalities = df_all['Modality'].unique()
    for mod in modalities:
        print(f"\n\nModality: {mod}")
        analyze_modality_across_methods(df_all, mod)


# # results for all cancers
# # cancer_list = ['LAML', 'PRAD', 'THCA', 'OV', 'LGG']
# cancer_list = ['LAML', 'PRAD', 'THCA']
#
# for cancer in cancer_list:
#     print(cancer)
#     modality_results_by_cancer(cancer, beta=0.01)


between_groups_stat_test('LAML', beta=1)
