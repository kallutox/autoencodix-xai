from helper_functions import *
import matplotlib.pyplot as plt


def get_sorted_dimensions_by_cancer_lda(run_id, cancer_type_acronym):
    """
    Identifies and sorts latent dimensions that best separate a specific cancer type from others
    using Fisher's score.

    Args:
        run_id (str): Identifier for the specific run.
        cancer_type_acronym (str): The acronym of the cancer type to analyze.

    Returns:
        list: A sorted list of tuples (dimension, score) in descending order of Fisher's score.
    """
    cancer_list = ['OV', 'CESC', 'PRAD', 'PCPG', 'KIRC', 'BLCA', 'STAD', 'UCEC', 'LGG',
                   'KIRP', 'READ', 'COAD', 'LIHC', 'LUAD', 'GBM', 'THCA', 'PAAD', 'SARC', 'HNSC', 'ESCA',
                   'LUSC', 'SKCM', 'BRCA', 'LAML']

    if cancer_type_acronym not in cancer_list:
        raise ValueError("Invalid cancer type.")

    latent_df = get_latent_space(run_id)
    clin_df = get_tcga_clin_data()

    # Merge latent space data with clinical data
    merged_df = latent_df.merge(clin_df[['CANCER_TYPE_ACRONYM']], left_index=True, right_index=True)

    fisher_scores = {}

    for column in merged_df.columns:
        if "L_COMBINED-METH_MUT_RNA__varix_INPUT_" in column:
            latent_number = int(column.split("_")[-1])

            # Separate cancer type and other values
            cancer_values = merged_df[merged_df['CANCER_TYPE_ACRONYM'] == cancer_type_acronym][column].dropna()
            others_values = merged_df[merged_df['CANCER_TYPE_ACRONYM'] != cancer_type_acronym][column].dropna()

            # Ensure there are enough samples in both groups
            if len(cancer_values) > 1 and len(others_values) > 1:
                mu_c = cancer_values.mean()
                mu_o = others_values.mean()
                var_c = cancer_values.var(ddof=1)
                var_o = others_values.var(ddof=1)

                # Calculate Fisher's ratio
                fisher_ratio = (mu_c - mu_o) ** 2 / (var_c + var_o)
                fisher_scores[latent_number] = fisher_ratio

    if not fisher_scores:
        raise ValueError("No valid dimensions found for LDA.")

    # Sort dimensions by Fisher's score in descending order
    sorted_dimensions = sorted(fisher_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_dimensions


def get_top_dim(top_n=5):
    #run_ids = ['tcga_001_1', 'tcga_001_2']
    run_ids = ['tcga_001_1', 'tcga_001_2', 'tcga_001_3', 'tcga_001_4', 'tcga_001_5', 'tcga_001_6', 'tcga_001_7',
               'tcga_001_8', 'tcga_001_9', 'tcga_001_10']

    cancer_list = ['OV', 'CESC', 'PRAD', 'PCPG', 'KIRC', 'BLCA', 'STAD', 'UCEC', 'LGG',
                   'KIRP', 'COAD', 'LIHC', 'LUAD', 'THCA', 'PAAD', 'SARC', 'HNSC', 'ESCA',
                   'LUSC', 'SKCM', 'BRCA', 'LAML']

    for run_id in run_ids:
        print(f"{run_id}:")
        for cancer in cancer_list:
            sorted_dimensions = get_sorted_dimensions_by_cancer_lda(run_id, cancer)
            dimensions = ", ".join([f"{dim}" for dim, score in sorted_dimensions[:top_n]])
            print(f"{cancer}: {dimensions}")
        print("\n")


def get_results_per_model(run_ids, top_n=1):
    """
    Calculate the top N dimensions and corresponding scores for each cancer across multiple models.

    Args:
        top_n (int): Number of top dimensions to retrieve for each cancer.
    Returns:
        dict: A dictionary where each key is a cancer type, and the value is a list of dictionaries
              containing the top dimensions and scores for each model.
    """
    results_per_model = {}

    cancer_list = ['OV', 'CESC', 'PRAD', 'PCPG', 'KIRC', 'BLCA', 'STAD', 'UCEC', 'LGG',
                   'KIRP', 'COAD', 'LIHC', 'LUAD', 'THCA', 'PAAD', 'SARC', 'HNSC', 'ESCA',
                   'LUSC', 'SKCM', 'BRCA', 'LAML']

    for cancer in cancer_list:
        results_per_model[cancer] = []

        for run_id in run_ids:
            sorted_dimensions = get_sorted_dimensions_by_cancer_lda(run_id, cancer)

            # Append the top N dimensions for the current model
            top_dimensions = [{'dimension': dim, 'score': score} for dim, score in sorted_dimensions[:top_n]]
            results_per_model[cancer].extend(top_dimensions)

    return results_per_model


def calculate_exclusivity(results_per_model):
    """
    Calculate the dimensional exclusivity for each cancer.

    Args:
        results_per_model (dict): A dictionary where keys are cancer types, and values are lists of top dimensions
                                  across all models.

    Returns:
        dict: A dictionary with cancers as keys and exclusivity scores as values.
    """
    exclusivity_scores = {}

    for cancer, dimensions in results_per_model.items():
        # Collect all dimensions for this cancer across models
        dims = [dim['dimension'] for dim in dimensions]

        # Calculate overlaps with other cancers
        other_dims = [
            dim['dimension']
            for other_cancer, other_dimensions in results_per_model.items()
            if other_cancer != cancer
            for dim in other_dimensions
        ]

        # Fraction of overlap
        overlap = len(set(dims) & set(other_dims)) / len(set(dims)) if dims else 0
        exclusivity_scores[cancer] = 1 - overlap  # Higher is more exclusive

    return exclusivity_scores


def aggregate_results(results_per_model):
    aggregated = {}
    exclusivity_scores = calculate_exclusivity(results_per_model)

    for cancer, dimensions in results_per_model.items():
        # Collect all scores
        scores = [dim['score'] for dim in dimensions]

        # Mean score and exclusivity
        mean_score = np.mean(scores) if scores else 0
        exclusivity_score = exclusivity_scores.get(cancer, 0)

        aggregated[cancer] = {
            'mean_score': mean_score,
            'exclusivity_score': exclusivity_score,
        }
    return aggregated


def rank_cancers(aggregated_results, w1=1, w2=0.5):
    ranked_cancers = []
    for cancer, stats in aggregated_results.items():
        rank_metric = w1 * stats['mean_score'] + w2 * stats['exclusivity_score']
        ranked_cancers.append((cancer, rank_metric, stats['mean_score'], stats['exclusivity_score']))

    print("\n\nResults ranked by metric:")
    print(f"{'Cancer':<10} | {'Rank Metric':<12} | {'Mean Score':<11} | {'Exclusivity':<12}")
    print("-" * 54)
    for cancer, rank_metric, mean_score, exclusivity in ranked_cancers:
        print(f"{cancer:<10} | {rank_metric:<12.2} |{mean_score:<12.2} | {exclusivity:<12}")

    return sorted(ranked_cancers, key=lambda x: x[1], reverse=True)


def visualize_exclusivity_vs_score(aggregated_results):
    cancers = list(aggregated_results.keys())
    mean_scores = [aggregated_results[cancer]['mean_score'] for cancer in cancers]
    exclusivity_scores = [aggregated_results[cancer]['exclusivity_score'] for cancer in cancers]

    plt.figure(figsize=(8, 6))
    plt.scatter(exclusivity_scores, mean_scores, color="#43b582", s=80, alpha=0.8, linewidth=0.8)

    for i, cancer in enumerate(cancers):
        plt.text(exclusivity_scores[i], mean_scores[i] + 0.05, cancer, fontsize=10, ha='center', va='bottom')

    plt.xlabel("Exclusivity Score", fontsize=12)
    plt.ylabel("Mean Score", fontsize=12)
    plt.title("Exclusivity vs. Mean Score for Cancer Types", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()


def run_dimension_analysis(top_n=1, show=False):
    run_ids_001 = ['tcga_001_1', 'tcga_001_2', 'tcga_001_3', 'tcga_001_4', 'tcga_001_5', 'tcga_001_6', 'tcga_001_7',
                   'tcga_001_8', 'tcga_001_9', 'tcga_001_10']
    results_per_model_001 = get_results_per_model(run_ids_001, top_n=top_n)
    aggregated_results_001 = aggregate_results(results_per_model_001)

    sorted_results_001 = sorted(aggregated_results_001.items(), key=lambda x: x[1]['mean_score'], reverse=True)

    print("Beta 0.01:")
    print("Results ranked by Mean Score only:")
    print(f"{'Cancer':<10} | {'Mean Score':<12}")
    print("-" * 29)

    for cancer, data in sorted_results_001:
        mean_score = f"{data['mean_score']:.2f}"
        print(f"{cancer:<10} | {mean_score:<12}")

    run_ids_1 = ['tcga_1_1', 'tcga_1_2', 'tcga_1_3', 'tcga_1_4', 'tcga_1_5', 'tcga_1_6',
                 'tcga_1_7', 'tcga_1_8', 'tcga_1_9', 'tcga_1_10']
    results_per_model_1 = get_results_per_model(run_ids_1, top_n=top_n)
    aggregated_results_1 = aggregate_results(results_per_model_1)

    sorted_results_1 = sorted(aggregated_results_1.items(), key=lambda x: x[1]['mean_score'], reverse=True)

    print("Beta 1:")
    print("Results ranked by Mean Score only:")
    print(f"{'Cancer':<10} | {'Mean Score':<12}")
    print("-" * 29)

    for cancer, data in sorted_results_1:
        mean_score = f"{data['mean_score']:.2f}"
        print(f"{cancer:<10} | {mean_score:<12}")

    # Combine and rank by aggregated results
    combined_results = {}
    for cancer in aggregated_results_001.keys():
        mean_score_001 = aggregated_results_001.get(cancer, {}).get('mean_score', 0)
        mean_score_1 = aggregated_results_1.get(cancer, {}).get('mean_score', 0)
        combined_mean_score = mean_score_001 + mean_score_1

        exclusivity_001 = aggregated_results_001.get(cancer, {}).get('exclusivity_score', 0)
        exclusivity_1 = aggregated_results_1.get(cancer, {}).get('exclusivity_score', 0)
        combined_exclusivity = (exclusivity_001 + exclusivity_1) / 2

        combined_results[cancer] = {
            'combined_mean_score': combined_mean_score,
            'combined_exclusivity': combined_exclusivity
        }

    sorted_combined_results = sorted(combined_results.items(), key=lambda x: x[1]['combined_mean_score'], reverse=True)

    print("\nCombined Results (Beta 0.01 + Beta 1):")
    print(f"{'Cancer':<10} | {'Combined Mean Score':<20}")
    print("-" * 40)

    for cancer, data in sorted_combined_results:
        combined_mean_score = f"{data['combined_mean_score']:.2f}"
        print(f"{cancer:<10} | {combined_mean_score:<20}")

    if show:
        visualize_exclusivity_vs_score(aggregated_results_001)
        visualize_exclusivity_vs_score(aggregated_results_1)


run_dimension_analysis()
#get_top_dim()
