import glob
import re
import json
from scipy.stats import spearmanr
from scipy.stats import rankdata
from collections import defaultdict
from helper_functions import *
from collections import Counter
from xai_methods import captum_importance_values


# synthetic signal function
def signal_strength_plot_across_model(xai_method='deepliftshap'):
    base_folders = ['base1', 'base2', 'base3']
    aggregated_top_10_positions = []
    num_configs_per_signal = None

    for base_folder in base_folders:
        config_folder = f"../synth_configs/{base_folder}"
        top_10_positions = []

        # sort config files by signal strength
        config_files = glob.glob(os.path.join(config_folder, "*_config.yaml"))
        config_files = sorted(config_files, key=lambda x: int(re.search(r'(\d+)signal', x).group(1)))

        if num_configs_per_signal is None:
            num_configs_per_signal = len(config_files)
        elif num_configs_per_signal != len(config_files):
            raise ValueError("Inconsistent number of config files across base folders.")

        for file_path in config_files:
            with open(file_path, "r") as f:
                config_data = yaml.safe_load(f)

            run_id = config_data.get("RUN_ID")

            # check for a missing RUN_ID and skip if not present
            if run_id is None:
                print(f"Skipping file {file_path} because it does not contain a RUN_ID.")
                continue

            model_type = 'varix'
            data_type = 'rna'
            latent_dim = get_best_dimension_by_sex_means(run_id)

            attribution_values = captum_importance_values(
                run_id=run_id,
                data_types=data_type,
                model_type=model_type,
                dimension=latent_dim,
                latent_space_explain=True,
                xai_method=xai_method,
                visualize=False,
                return_delta=False
            )

            attribution_dict = attribution_per_feature(
                attribution_values,
                get_interim_data(run_id, model_type)
            )

            top_features = get_top_features(attribution_dict)

            feature_list_path = os.path.join("..", config_data.get("FEATURE_SIGNAL"))
            with open(feature_list_path, "r") as f:
                feature_list = f.read().splitlines()

            positions = [i for i, feature in enumerate(top_features) if feature in feature_list]
            top_10_positions.append(positions)

            print(f"Feature Importance for {run_id} finished computing.")

        aggregated_top_10_positions.append(top_10_positions)

    # calculate average counts and standard deviation if multiple bases are provided
    if len(base_folders) > 1:
        average_counts = []
        std_dev_counts = []
        all_counts = [[] for _ in range(num_configs_per_signal)]
        for i in range(num_configs_per_signal):
            counts_per_base = [len(positions[i]) for positions in aggregated_top_10_positions]
            average_counts.append(np.mean(counts_per_base))
            std_dev_counts.append(np.std(counts_per_base))
            all_counts[i] = counts_per_base
    else:
        average_counts = [len(positions) for positions in aggregated_top_10_positions[0]]
        std_dev_counts = [0] * len(average_counts)
        all_counts = [[len(positions)] for positions in aggregated_top_10_positions[0]]

    # define signal strengths based on number of configs
    signal_strength = [0.1 * i for i in range(1, num_configs_per_signal + 1)]

    if xai_method == 'deepliftshap':
        xai_name = 'DeepLiftShap'
    elif xai_method == 'lime':
        xai_name = 'LIME'
    elif xai_method == 'integrated_gradients':
        xai_name = 'Integrated Gradients'
    else:
        xai_name = None

    cmap = LinearSegmentedColormap.from_list(
        "custom_gradient",
        ["#4354b5", "#43a2b5", "#43b582"]
    )

    # box plot
    plt.figure(figsize=(10, 6))

    positions = np.arange(0.1, 1.0, 0.1)  # Match your signal strengths
    plt.boxplot(all_counts, positions=positions, widths=0.05)

    plt.xlabel("Signal Strength")
    plt.ylabel("Number of Synthetic Features in Top 10")
    plt.title(f"Synthetic Features in Top 10 for {xai_name}", fontsize=15, fontweight="bold")

    plt.xticks(positions, [f"{s:.1f}" for s in signal_strength])

    base_names = "_".join(base_folders)
    os.makedirs("synth_reports/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/synth_data_{base_names}_{xai_method}_male_target_abs.png")
    plt.show()

    # save the averaged positions data
    data_to_save = {
        "signal_strength": signal_strength,
        "average_counts": average_counts,
        "std_dev_counts": std_dev_counts,
        "aggregated_top_10_positions": aggregated_top_10_positions
    }
    os.makedirs("synth_data", exist_ok=True)
    with open(f"synth_reports/synth_data_{base_names}_{xai_method}_male_target_abs.json", "w") as f:
        json.dump(data_to_save, f)


def signal_strength_plot_across_model_scatter(xai_method='deepliftshap'):

    base_folders = ['base1', 'base2', 'base3']
    aggregated_top_10_positions = []
    num_configs_per_signal = None

    for base_folder in base_folders:
        config_folder = f"../synth_configs/{base_folder}"
        top_10_positions = []

        # sort config files by signal strength
        config_files = glob.glob(os.path.join(config_folder, "*_config.yaml"))
        config_files = sorted(config_files, key=lambda x: int(re.search(r'(\d+)signal', x).group(1)))

        if num_configs_per_signal is None:
            num_configs_per_signal = len(config_files)
        elif num_configs_per_signal != len(config_files):
            raise ValueError("Inconsistent number of config files across base folders.")

        for file_path in config_files:
            with open(file_path, "r") as f:
                config_data = yaml.safe_load(f)

            run_id = config_data.get("RUN_ID")

            # Check for a missing RUN_ID and skip if not present
            if run_id is None:
                print(f"Skipping file {file_path} because it does not contain a RUN_ID.")
                continue

            model_type = 'varix'
            data_type = 'rna'
            latent_dim = get_best_dimension_by_sex_means(run_id)

            attribution_values = captum_importance_values(
                run_id=run_id,
                data_types=data_type,
                model_type=model_type,
                dimension=latent_dim,
                latent_space_explain=True,
                xai_method=xai_method,
                visualize=False,
                return_delta=False
            )

            attribution_dict = attribution_per_feature(
                attribution_values,
                get_interim_data(run_id, model_type)
            )

            top_features = get_top_features(attribution_dict)

            feature_list_path = os.path.join("..", config_data.get("FEATURE_SIGNAL"))
            with open(feature_list_path, "r") as f:
                feature_list = f.read().splitlines()

            positions = [i for i, feature in enumerate(top_features) if feature in feature_list]
            top_10_positions.append(positions)

            print(f"Feature Importance for {run_id} finished computing.")

        aggregated_top_10_positions.append(top_10_positions)

    average_counts = []
    std_dev_counts = []
    scatter_points = []

    for i in range(num_configs_per_signal):
        counts_per_base = [len(positions[i]) for positions in aggregated_top_10_positions]
        average_counts.append(np.mean(counts_per_base))
        std_dev_counts.append(np.std(counts_per_base))
        scatter_points.append(counts_per_base)

    signal_strength = [0.1 * i for i in range(1, num_configs_per_signal + 1)]

    plt.figure(figsize=(10, 6))
    plt.bar(
        signal_strength,
        average_counts,
        color="#627ad1",
        width=0.05,
        align='center',
        yerr=std_dev_counts,
        capsize=5,
    )

    # Scatter plot for individual data points with jitter
    for i, signal in enumerate(signal_strength):
        jitter = np.random.uniform(-0.02, 0.02, len(scatter_points[i]))  # Add jitter to x positions
        plt.scatter(
            signal + jitter,  # Apply jitter to x-axis
            scatter_points[i],
            color="lightgrey",
            alpha=0.7,
            s=50,
        )

    plt.xlabel('Signal Strength', fontsize=12)
    plt.ylabel('Average Number of Synthetic Features in Top 10', fontsize=12)
    plt.title(f'Synthetic Features in Top 10 for {xai_method}', fontsize=15, fontweight="bold")
    plt.xticks(np.arange(0.1, 1.0, 0.1))
    plt.tight_layout()

    os.makedirs("synth_data/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/synth_data_{xai_method}_scatter_plot.png")
    plt.show()


def signal_strength_plot_single_model(xai_method='deepliftshap', num_repeats=3):
    config_folder = f"../synth_configs/base1"
    top_10_positions = []

    # sort config files by signal strength
    config_files = glob.glob(os.path.join(config_folder, "*_config.yaml"))
    config_files = sorted(config_files, key=lambda x: int(re.search(r'(\d+)signal', x).group(1)))

    num_configs_per_signal = len(config_files)

    for file_path in config_files:
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)

        run_id = config_data.get("RUN_ID")

        # check for a missing RUN_ID and skip if not present
        if run_id is None:
            print(f"Skipping file {file_path} because it does not contain a RUN_ID.")
            continue

        signal_positions = []

        for _ in range(num_repeats):
            model_type = 'varix'
            data_type = 'rna'
            data_set = 'cf'
            latent_dim = get_best_dimension_by_sex_means(run_id)

            attribution_values = captum_importance_values(
                run_id=run_id,
                data_types=data_type,
                model_type=model_type,
                dimension=latent_dim,
                latent_space_explain=True,
                xai_method=xai_method,
                visualize=False,
                return_delta=False
            )

            attribution_dict = attribution_per_feature(
                attribution_values,
                get_interim_data(run_id, model_type)
            )

            top_features = get_top_features(attribution_dict)

            feature_list_path = os.path.join("..", config_data.get("FEATURE_SIGNAL"))
            with open(feature_list_path, "r") as f:
                feature_list = f.read().splitlines()

            positions = [i for i, feature in enumerate(top_features) if feature in feature_list]
            signal_positions.append(positions)

            print(f"Feature Importance for {run_id} (repeat) finished computing.")

        top_10_positions.append(signal_positions)

    # calculate average counts and standard deviation for each signal strength
    average_counts = []
    std_dev_counts = []
    for signal_runs in top_10_positions:
        counts = [len(positions) for positions in signal_runs]
        average_counts.append(np.mean(counts))
        std_dev_counts.append(np.std(counts))

    # define signal strengths based on the number of configs
    signal_strength = [0.1 * i for i in range(1, num_configs_per_signal + 1)]

    plt.figure(figsize=(10, 6))
    plt.bar(signal_strength, average_counts, width=0.05, align='center', yerr=std_dev_counts, capsize=5)
    plt.xlabel('Signal Strength')
    plt.ylabel('Number of Synthetic Features in Top 10 (Average)')
    plt.title(f'Synthetic Features in Top 10 for {xai_method}', fontsize=15, fontweight="bold")
    plt.xticks(np.arange(0.1, 1.0, 0.1))

    # save the plot
    os.makedirs("synth_data/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/synth_data_base1_{xai_method}_male_target.png")
    plt.show()

    # save the averaged positions data
    data_to_save = {
        "signal_strength": signal_strength,
        "average_counts": average_counts,
        "std_dev_counts": std_dev_counts,
        "top_10_positions": top_10_positions
    }
    os.makedirs("synth_data", exist_ok=True)
    with open(f"synth_reports/synth_data_base1_{xai_method}_male_target.json", "w") as f:
        json.dump(data_to_save, f)


def feature_importance_and_visualize(run_id, data_set, xai_method='deepliftshap', scatterplot=True, synth_test=False):

    attribution_values = captum_importance_values(
        run_id=run_id,
        data_types='rna',
        model_type='varix',
        dimension=get_best_dimension_by_sex_means(run_id),
        latent_space_explain=True,
        xai_method=xai_method,
        visualize=scatterplot,
        return_delta=False
    )

    attribution_dict = attribution_per_feature(
        attribution_values,
        get_interim_data(run_id, 'varix')
    )

    top_features = get_top_features(attribution_dict, 10)

    if synth_test:
        config_data = get_config(run_id)
        feature_list_path = os.path.join("..", config_data.get("FEATURE_SIGNAL"))
        with open(feature_list_path, "r") as f:
            feature_list = f.read().splitlines()

        positions = [i for i, feature in enumerate(top_features) if feature in feature_list]

        print('Positions of top features that were modified:', positions)
        print('Number of top features that were modified:', len(positions))

    print(top_features)


# reproducibility functions
def intra_model_overlap(intra_run_id, top_n=100, num_repeats=3):

    # intra model: same model, calculate feature importance 5 times for each method (15x)
    xai_methods = ['deepliftshap', 'lime', 'integrated_gradients']
    overlap_results = {}

    for xai_method in xai_methods:
        intra_top_features = []

        # calculate feature importance n times for the current XAI method
        for _ in range(num_repeats):
            attribution_values = captum_importance_values(
                run_id=intra_run_id,
                data_types='rna',
                model_type='varix',
                dimension=get_best_dimension_by_sex_means(intra_run_id),
                latent_space_explain=True,
                xai_method=xai_method,
                visualize=False,
                return_delta=False
            )

            attribution_dict = attribution_per_feature(
                attribution_values,
                get_interim_data(intra_run_id, 'varix')
            )

            top_features = get_top_features(attribution_dict, top_n=top_n)
            intra_top_features.append(top_features)

        all_features = [feature for sublist in intra_top_features for feature in sublist]
        feature_counts = Counter(all_features)
        overlap_in_all = sum(1 for count in feature_counts.values() if count == len(intra_top_features))
        overlap_results[xai_method] = overlap_in_all

    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})
    palette = sns.color_palette("pastel")

    plt.figure(figsize=(8, 5))
    plt.bar(overlap_results.keys(), overlap_results.values(), color=palette[:len(overlap_results)], alpha=0.9)
    plt.ylabel("Number of Features", fontsize=12)
    plt.title(f"Overlap of Top {top_n} Features Across XAI Methods", fontsize=15, fontweight="bold")
    plt.tight_layout()

    os.makedirs("synth_data/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/overlap_intramodel_{xai_method}_{num_repeats}repeats.png")
    plt.show()


def intra_inter_overlap(top_n=100, num_repeats=3):
    xai_methods = ['deepliftshap', 'lime', 'integrated_gradients']
    overlap_results_intra = {}
    overlap_results_inter = {}

    base_run_ids = ['base1', 'base2', 'base3', 'base4', 'base5']
    max_bases = len(base_run_ids)

    if num_repeats > max_bases:
        raise ValueError(f"num_repeats cannot exceed {max_bases}, the number of available bases.")

    base_run_ids = base_run_ids[:num_repeats]
    intra_run_id = 'base1'

    for xai_method in xai_methods:
        # intra-model reproducibility
        intra_top_features = []

        for _ in range(num_repeats):
            attribution_values = captum_importance_values(
                run_id=intra_run_id,
                data_types='rna',
                model_type='varix',
                dimension=get_best_dimension_by_sex_means(intra_run_id),
                latent_space_explain=True,
                xai_method=xai_method,
                visualize=False,
                return_delta=False
            )

            if xai_method == 'lime':
                attribution_dict = attribution_per_feature(
                    attribution_values,
                    get_interim_data(intra_run_id, 'varix')
                )
            else:
                attribution_dict = attribution_per_feature(
                    attribution_values,
                    get_interim_data(intra_run_id, 'varix')
                )

            top_features = get_top_features(attribution_dict, top_n=top_n)
            intra_top_features.append(top_features)

        all_features_intra = [feature for sublist in intra_top_features for feature in sublist]
        feature_counts_intra = Counter(all_features_intra)

        overlap_in_all_intra = sum(1 for count in feature_counts_intra.values() if count == len(intra_top_features))
        overlap_results_intra[xai_method] = overlap_in_all_intra

        # inter-model reproducibility
        inter_top_features = []

        for run_id in base_run_ids:
            attribution_values = captum_importance_values(
                run_id=run_id,
                data_types='rna',
                model_type='varix',
                dimension=get_best_dimension_by_sex_means(run_id),
                latent_space_explain=True,
                xai_method=xai_method,
                visualize=False,
                return_delta=False
            )

            attribution_dict = attribution_per_feature(
                attribution_values,
                get_interim_data(run_id, 'varix')
            )

            top_features = get_top_features(attribution_dict, top_n=top_n)
            inter_top_features.append(top_features)

        all_features_inter = [feature for sublist in inter_top_features for feature in sublist]
        feature_counts_inter = Counter(all_features_inter)

        overlap_in_all_inter = sum(1 for count in feature_counts_inter.values() if count == len(base_run_ids))
        overlap_results_inter[xai_method] = overlap_in_all_inter

    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    x = np.arange(len(xai_methods))
    width = 0.35

    plt.figure(figsize=(10, 6))

    if top_n == 100:
        intra_values = [overlap_results_intra[method] for method in xai_methods]
        inter_values = [overlap_results_inter[method] for method in xai_methods]
        y_label = "Number of Features"
    else:
        intra_values = [overlap_results_intra[method] / top_n * 100 for method in xai_methods]
        inter_values = [overlap_results_inter[method] / top_n * 100 for method in xai_methods]
        y_label = "Percentage of Features (%)"
        plt.gca().set_ylim(0, 100)  # Optionally set y-axis limits for percentages

    intra_values = [overlap_results_intra[method] / top_n * 100 for method in xai_methods]
    inter_values = [overlap_results_inter[method] / top_n * 100 for method in xai_methods]
    y_label = "Percentage of Features (%)"
    plt.gca().set_ylim(0, 100)

    # plot bars
    plt.bar(x - width / 2, intra_values, width, label='Intra-Model', color=["#627ad1"])
    plt.bar(x + width / 2, inter_values, width, label='Inter-Model', color=["#7ac293"])

    xai_names = ['DeepLiftShap', 'LIME', 'Integrated Gradients']
    plt.xticks(x, xai_names, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f"Overlap of Top {top_n} Features: Intra vs Inter Model", fontsize=15, fontweight="bold")
    plt.legend(fontsize=12)

    os.makedirs("synth_data/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/overlap_intra_inter_{top_n}top_features_{num_repeats}runs.png")
    plt.tight_layout()
    plt.show()


def intra_inter_correlation(top_n=100, num_repeats=3):
    xai_methods = ['deepliftshap', 'lime', 'integrated_gradients']
    base_run_ids = ['base1', 'base2', 'base3', 'base4', 'base5']
    max_bases = len(base_run_ids)

    if num_repeats > max_bases:
        raise ValueError(f"num_repeats cannot exceed {max_bases}, the number of available bases.")

    base_run_ids = base_run_ids[:num_repeats]
    intra_run_id = 'base1'

    correlation_results_intra = {}
    correlation_results_inter = {}

    for xai_method in xai_methods:
        # intra-model correlation
        intra_correlations = []

        for _ in range(num_repeats):
            attribution_values_1 = captum_importance_values(
                run_id=intra_run_id,
                data_types='rna',
                model_type='varix',
                dimension=get_best_dimension_by_sex_means(intra_run_id),
                latent_space_explain=True,
                xai_method=xai_method,
                visualize=False,
                return_delta=False
            )

            attribution_values_2 = captum_importance_values(
                run_id=intra_run_id,
                data_types='rna',
                model_type='varix',
                dimension=get_best_dimension_by_sex_means(intra_run_id),
                latent_space_explain=True,
                xai_method=xai_method,
                visualize=False,
                return_delta=False
            )

            attribution_dict_1 = attribution_per_feature(
                attribution_values_1,
                get_interim_data(intra_run_id, 'varix')
            )

            attribution_dict_2 = attribution_per_feature(
                attribution_values_2,
                get_interim_data(intra_run_id, 'varix')
            )

            ranked_features_1 = sorted(attribution_dict_1.items(), key=lambda x: x[1], reverse=True)[:top_n]
            ranked_features_2 = sorted(attribution_dict_2.items(), key=lambda x: x[1], reverse=True)[:top_n]

            features_1 = [feature[0] for feature in ranked_features_1]
            features_2 = [feature[0] for feature in ranked_features_2]

            # compute Spearman's correlation
            ranks_1 = np.arange(1, len(features_1) + 1)
            ranks_2 = [ranks_1[features_1.index(f)] if f in features_1 else len(ranks_1) + 1 for f in features_2]
            spearman_corr, _ = spearmanr(ranks_1, ranks_2)
            intra_correlations.append(spearman_corr)

        correlation_results_intra[xai_method] = np.mean(intra_correlations)

        # nter-model correlation
        inter_correlations = []

        for i, run_id_1 in enumerate(base_run_ids):
            for run_id_2 in base_run_ids[i + 1:]:
                attribution_values_1 = captum_importance_values(
                    run_id=run_id_1,
                    data_types='rna',
                    model_type='varix',
                    dimension=get_best_dimension_by_sex_means(run_id_1),
                    latent_space_explain=True,
                    xai_method=xai_method,
                    visualize=False,
                    return_delta=False
                )

                attribution_values_2 = captum_importance_values(
                    run_id=run_id_2,
                    data_types='rna',
                    model_type='varix',
                    dimension=get_best_dimension_by_sex_means(run_id_2),
                    latent_space_explain=True,
                    xai_method=xai_method,
                    visualize=False,
                    return_delta=False
                )

                attribution_dict_1 = attribution_per_feature(
                    attribution_values_1,
                    get_interim_data(run_id_1, 'varix')
                )

                attribution_dict_2 = attribution_per_feature(
                    attribution_values_2,
                    get_interim_data(run_id_2, 'varix')
                )

                # Get ranks for top_n features
                ranked_features_1 = sorted(attribution_dict_1.items(), key=lambda x: x[1], reverse=True)[:top_n]
                ranked_features_2 = sorted(attribution_dict_2.items(), key=lambda x: x[1], reverse=True)[:top_n]

                features_1 = [feature[0] for feature in ranked_features_1]
                features_2 = [feature[0] for feature in ranked_features_2]

                # Compute Spearman's correlation
                ranks_1 = np.arange(1, len(features_1) + 1)
                ranks_2 = [ranks_1[features_1.index(f)] if f in features_1 else len(ranks_1) + 1 for f in features_2]
                spearman_corr, _ = spearmanr(ranks_1, ranks_2)
                inter_correlations.append(spearman_corr)

        correlation_results_inter[xai_method] = np.mean(inter_correlations)

    # Plot grouped barplot
    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    x = np.arange(len(xai_methods))  # Bar positions
    width = 0.35  # Width of each bar

    plt.figure(figsize=(10, 6))

    intra_values = [correlation_results_intra[method] for method in xai_methods]
    inter_values = [correlation_results_inter[method] for method in xai_methods]

    # Plot bars
    plt.bar(x - width / 2, intra_values, width, label='Intra-Model', color=["#627ad1"])
    plt.bar(x + width / 2, inter_values, width, label='Inter-Model', color=["#7ac293"])

    xai_names = ['DeepLiftShap', 'LIME', 'Integrated Gradients']
    plt.xticks(x, xai_names, fontsize=12)
    plt.ylabel("Spearman's Correlation", fontsize=12)
    plt.ylim(0, 1)
    plt.title(f"Spearman's Correlation: Intra vs Inter Model", fontsize=15, fontweight="bold")
    plt.legend(fontsize=12)

    # Save and show the plot
    os.makedirs("synth_data/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/correlation_intra_inter_{top_n}top_features_{num_repeats}runs.png")
    plt.tight_layout()
    plt.show()

    print('inter-model correlation ', correlation_results_inter)
    print('intra-model correlation ', correlation_results_intra)


def inter_model_aggregation_overlap(num_repeats=3, top_n=100):
    """
    Calculates and plots the overlap of top features across aggregated models for multiple XAI methods.

    Args:
        num_repeats (int): Number of repetitions of the aggregation process.
        top_n (int): Number of top features to consider.
    """
    xai_methods = ['deepliftshap', 'lime', 'integrated_gradients']
    base_run_ids = ['base1', 'base2', 'base3', 'base4', 'base5', 'base6', 'base7', 'base8', 'base9', 'base10']
    overlap_results = {}

    for xai_method in xai_methods:
        aggregated_top_features_all_runs = []

        for repeat in range(num_repeats):
            # Aggregate attributions across the 3 base models
            aggregated_attributions = Counter()

            for run_id in base_run_ids:
                # Compute attributions for the current run
                attribution_values = captum_importance_values(
                    run_id=run_id,
                    data_types='rna',
                    model_type='varix',
                    dimension=get_best_dimension_by_sex_means(run_id),
                    latent_space_explain=True,
                    xai_method=xai_method,
                    visualize=False,
                    return_delta=False
                )

                # Convert attributions to a dictionary of feature importance
                attribution_dict = attribution_per_feature(
                    attribution_values,
                    get_interim_data(run_id, 'varix')
                )

                # Aggregate attributions across the 3 models
                for feature, importance in attribution_dict.items():
                    aggregated_attributions[feature] += importance

            # Average attributions across the 3 models
            num_models = len(base_run_ids)
            for feature in aggregated_attributions:
                aggregated_attributions[feature] /= num_models

            # Sort features by aggregated importance and get the top features
            sorted_features = sorted(aggregated_attributions.items(), key=lambda x: x[1], reverse=True)
            top_features = [feature[0] for feature in sorted_features[:top_n]]

            aggregated_top_features_all_runs.append(top_features)

        # Calculate overlaps across the 3 repetitions
        all_features = [feature for sublist in aggregated_top_features_all_runs for feature in sublist]
        feature_counts = Counter(all_features)
        overlap_in_all = sum(1 for count in feature_counts.values() if count == num_repeats)

        # Store the result for this XAI method
        overlap_results[xai_method] = overlap_in_all

    # Plot results for all methods
    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    xai_names = ['DeepLiftShap', 'LIME', 'Integrated Gradients']
    plt.figure(figsize=(10, 6)) # ["#627ad1"], ["#7ac293"]
    plt.bar(xai_names, overlap_results.values(), color=["#627ad1", "#5fb6c7", "#7ac293"], alpha=0.9, width=0.5)
    plt.ylabel("Number of Features", fontsize=12)
    plt.title(f"Overlap of Top {top_n} Features for Inter-Model Aggregation", fontsize=15, fontweight="bold")
    plt.tight_layout()

    # Save and show the plot
    os.makedirs("synth_data/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/inter_model_aggr_{num_repeats}repeats.png")
    plt.show()

    return overlap_results


def intra_model_aggregation_overlap(num_repeats=10, top_n=100):
    """
    Calculates and plots the overlap of top features for multiple XAI methods
    by averaging repeated calculations per model and comparing across models.

    Args:
        num_repeats (int): Number of repeated calculations per model.
        top_n (int): Number of top features to consider.
    """
    xai_methods = ['deepliftshap', 'lime', 'integrated_gradients']
    base_run_ids = ['base1', 'base2', 'base3']
    overlap_results = {}

    for xai_method in xai_methods:
        averaged_top_features_all_models = []

        for run_id in base_run_ids:
            # Aggregate attributions across num_repeats for this model
            aggregated_attributions = Counter()

            for _ in range(num_repeats):
                attribution_values = captum_importance_values(
                    run_id=run_id,
                    data_types='rna',
                    model_type='varix',
                    dimension=get_best_dimension_by_sex_means(run_id),
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
                    aggregated_attributions[feature] += importance

            for feature in aggregated_attributions:
                aggregated_attributions[feature] /= num_repeats

            sorted_features = sorted(aggregated_attributions.items(), key=lambda x: x[1], reverse=True)
            top_features = [feature[0] for feature in sorted_features[:top_n]]

            averaged_top_features_all_models.append(top_features)

        # Calculate overlaps across all models
        all_features = [feature for sublist in averaged_top_features_all_models for feature in sublist]
        feature_counts = Counter(all_features)
        overlap_in_all = sum(1 for count in feature_counts.values() if count == len(base_run_ids))

        # Store the result for this XAI method
        overlap_results[xai_method] = overlap_in_all

    # Plot results for all methods
    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    xai_names = ['DeepLiftShap', 'LIME', 'Integrated Gradients']
    plt.figure(figsize=(10, 6))
    plt.bar(xai_names, overlap_results.values(), color=["#627ad1", "#5fb6c7", "#7ac293"], alpha=0.9, width=0.5)
    plt.ylabel("Number of Features", fontsize=12)
    plt.title(f"Overlap of Top {top_n} Features for Intra-Model Aggregation", fontsize=15, fontweight="bold")
    plt.tight_layout()

    # Save and show the plot
    os.makedirs("synth_data/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/intra_model_aggr_{num_repeats}repeats.png")
    plt.show()

    return overlap_results


def intra_model_aggregation_corr(num_repeats=3, top_n=100):
    """
    Calculates and plots the Spearman's correlation of feature attribution rankings
    across multiple models for different XAI methods.

    Args:
        num_repeats (int): Number of repeated calculations per model.
        top_n (int): Number of top features to consider for ranking.
    """
    xai_methods = ['deepliftshap', 'lime', 'integrated_gradients']
    base_run_ids = ['base1', 'base2', 'base3']
    correlation_results = {}

    for xai_method in xai_methods:
        model_rankings = []

        for run_id in base_run_ids:
            # Aggregate attributions across num_repeats for this model
            aggregated_attributions = Counter()

            for _ in range(num_repeats):
                attribution_values = captum_importance_values(
                    run_id=run_id,
                    data_types='rna',
                    model_type='varix',
                    dimension=get_best_dimension_by_sex_means(run_id),
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
                    aggregated_attributions[feature] += importance

            # Average attributions across repeats
            for feature in aggregated_attributions:
                aggregated_attributions[feature] /= num_repeats

            # Sort features by aggregated importance and retain their rankings
            sorted_features = sorted(aggregated_attributions.items(), key=lambda x: x[1], reverse=True)
            ranked_features = [feature[0] for feature in sorted_features[:top_n]]
            model_rankings.append(ranked_features)

        # Calculate pairwise Spearman's correlation across all models
        correlation_matrix = []
        for i in range(len(model_rankings)):
            for j in range(i + 1, len(model_rankings)):
                rank_i = np.arange(1, len(model_rankings[i]) + 1)
                rank_j = np.array([rank_i[model_rankings[i].index(f)] if f in model_rankings[i] else len(rank_i) + 1 for f in model_rankings[j]])
                spearman_corr, _ = spearmanr(rank_i, rank_j)
                correlation_matrix.append(spearman_corr)

        # Store the average Spearman's correlation for this XAI method
        avg_correlation = np.mean(correlation_matrix)
        correlation_results[xai_method] = avg_correlation

    # Plot results for all methods
    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    xai_names = ['DeepLiftShap', 'LIME', 'Integrated Gradients']
    plt.figure(figsize=(10, 6))
    plt.bar(xai_names, correlation_results.values(), color=["#627ad1", "#5fb6c7", "#7ac293"], alpha=0.9, width=0.5)
    plt.ylabel("Spearman's Correlation", fontsize=12)
    plt.title(f"Spearman's Correlation of Feature Rankings Across Models", fontsize=15, fontweight="bold")
    plt.tight_layout()

    # Save and show the plot
    os.makedirs("synth_data/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/intra_model_correlation_{num_repeats}repeats.png")
    plt.show()

    return correlation_results


def intra_inter_aggregation_overlap(num_repeats_intra=10, num_repeats_inter=3, top_n=100):
    """
    Calculates and plots the overlap of top features for intra- and inter-model aggregation across XAI methods.

    Args:
        num_repeats_intra (int): Number of repeated calculations per model for intra-model aggregation.
        num_repeats_inter (int): Number of repetitions of the aggregation process for inter-model aggregation.
        top_n (int): Number of top features to consider.
    """
    xai_methods = ['deepliftshap', 'lime', 'integrated_gradients']
    intra_base_run_ids = ['base1', 'base2', 'base3']
    inter_base_run_ids = ['base1', 'base2', 'base3', 'base4', 'base5', 'base6', 'base7', 'base8', 'base9', 'base10']

    overlap_results_intra = {}
    overlap_results_inter = {}

    # Intra-Model Aggregation
    for xai_method in xai_methods:
        averaged_top_features_all_models = []

        for run_id in intra_base_run_ids:
            aggregated_attributions = Counter()

            for _ in range(num_repeats_intra):
                attribution_values = captum_importance_values(
                    run_id=run_id,
                    data_types='rna',
                    model_type='varix',
                    dimension=get_best_dimension_by_sex_means(run_id),
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
                    aggregated_attributions[feature] += importance

            for feature in aggregated_attributions:
                aggregated_attributions[feature] /= num_repeats_intra

            sorted_features = sorted(aggregated_attributions.items(), key=lambda x: x[1], reverse=True)
            top_features = [feature[0] for feature in sorted_features[:top_n]]

            averaged_top_features_all_models.append(top_features)

        all_features = [feature for sublist in averaged_top_features_all_models for feature in sublist]
        feature_counts = Counter(all_features)
        overlap_in_all = sum(1 for count in feature_counts.values() if count == len(intra_base_run_ids))

        overlap_results_intra[xai_method] = overlap_in_all

    # Inter-Model Aggregation
    for xai_method in xai_methods:
        aggregated_top_features_all_runs = []

        for repeat in range(num_repeats_inter):
            aggregated_attributions = Counter()

            for run_id in inter_base_run_ids:
                attribution_values = captum_importance_values(
                    run_id=run_id,
                    data_types='rna',
                    model_type='varix',
                    dimension=get_best_dimension_by_sex_means(run_id),
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
                    aggregated_attributions[feature] += importance

            for feature in aggregated_attributions:
                aggregated_attributions[feature] /= len(inter_base_run_ids)

            sorted_features = sorted(aggregated_attributions.items(), key=lambda x: x[1], reverse=True)
            top_features = [feature[0] for feature in sorted_features[:top_n]]

            aggregated_top_features_all_runs.append(top_features)

        all_features = [feature for sublist in aggregated_top_features_all_runs for feature in sublist]
        feature_counts = Counter(all_features)
        overlap_in_all = sum(1 for count in feature_counts.values() if count == num_repeats_inter)

        overlap_results_inter[xai_method] = overlap_in_all

    # Plot grouped barplot
    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    x = np.arange(len(xai_methods))  # Bar positions
    width = 0.35  # Width of each bar

    plt.figure(figsize=(12, 6))

    intra_values = [overlap_results_intra[method] for method in xai_methods]
    inter_values = [overlap_results_inter[method] for method in xai_methods]

    # Plot bars
    plt.bar(x - width / 2, intra_values, width, label='Intra-Model Aggregation', color=["#627ad1"])
    plt.bar(x + width / 2, inter_values, width, label='Inter-Model Aggregation', color=["#7ac293"])

    xai_names = ['DeepLiftShap', 'LIME', 'Integrated Gradients']
    plt.xticks(x, xai_names, fontsize=12)
    plt.ylabel("Number of Features", fontsize=12)
    plt.title(f"Overlap of Top {top_n} Features: Intra vs Inter Model Aggregation", fontsize=15, fontweight="bold")
    plt.legend(fontsize=12)

    # Save and show the plot
    os.makedirs("synth_data/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/intra_inter_aggr_overlap_{top_n}features.png")
    plt.tight_layout()
    plt.show()

    return overlap_results_intra, overlap_results_inter


def intra_inter_aggr_spearman_correlation_topn(num_repeats_intra=10, num_repeats_inter=3, top_n=100):
    """
    Calculates and plots the Spearman's correlation of feature attribution rankings
    for intra- and inter-model aggregation across XAI methods.

    Args:
        num_repeats_intra (int): Number of repeated calculations per model for intra-model aggregation.
        num_repeats_inter (int): Number of repetitions of the aggregation process for inter-model aggregation.
        top_n (int): Number of top features to consider.
    """
    xai_methods = ['deepliftshap', 'lime', 'integrated_gradients']
    intra_base_run_ids = ['base1', 'base2', 'base3']
    inter_base_run_ids = ['base1', 'base2', 'base3', 'base4', 'base5', 'base6', 'base7', 'base8', 'base9', 'base10']

    spearman_results_intra = {}
    spearman_results_inter = {}

    # Intra-Model Aggregation
    for xai_method in xai_methods:
        intra_rankings = []

        for run_id in intra_base_run_ids:
            attribution_repeats = []

            for _ in range(num_repeats_intra):
                attribution_values = captum_importance_values(
                    run_id=run_id,
                    data_types='rna',
                    model_type='varix',
                    dimension=get_best_dimension_by_sex_means(run_id),
                    latent_space_explain=True,
                    xai_method=xai_method,
                    visualize=False,
                    return_delta=False
                )

                attribution_dict = attribution_per_feature(
                    attribution_values,
                    get_interim_data(run_id, 'varix')
                )

                # Sort and keep top_n features
                sorted_features = sorted(attribution_dict.items(), key=lambda x: x[1], reverse=True)
                ranked_features = [feature[0] for feature in sorted_features[:top_n]]
                attribution_repeats.append(ranked_features)

            # Average rankings for this model
            intra_rankings.append(attribution_repeats)

        # Calculate Spearman's correlation for intra-model
        correlation_matrix = []
        for model_repeats in intra_rankings:
            for i in range(len(model_repeats)):
                for j in range(i + 1, len(model_repeats)):
                    rank_i = np.arange(1, len(model_repeats[i]) + 1)
                    rank_j = np.array([rank_i[model_repeats[i].index(f)] if f in model_repeats[i] else len(rank_i) + 1 for f in model_repeats[j]])
                    spearman_corr, _ = spearmanr(rank_i, rank_j)
                    correlation_matrix.append(spearman_corr)

        spearman_results_intra[xai_method] = np.mean(correlation_matrix)

    # Inter-Model Aggregation
    for xai_method in xai_methods:
        inter_rankings = []

        for repeat in range(num_repeats_inter):
            aggregated_attributions = Counter()

            for run_id in inter_base_run_ids:
                attribution_values = captum_importance_values(
                    run_id=run_id,
                    data_types='rna',
                    model_type='varix',
                    dimension=get_best_dimension_by_sex_means(run_id),
                    latent_space_explain=True,
                    xai_method=xai_method,
                    visualize=False,
                    return_delta=False
                )

                attribution_dict = attribution_per_feature(
                    attribution_values,
                    get_interim_data(run_id, 'varix')
                )

                # Aggregate attribution scores
                for feature, importance in attribution_dict.items():
                    aggregated_attributions[feature] += importance

            # Average attributions and store top_n rankings
            for feature in aggregated_attributions:
                aggregated_attributions[feature] /= num_repeats_inter

            sorted_features = sorted(aggregated_attributions.items(), key=lambda x: x[1], reverse=True)
            ranked_features = [feature[0] for feature in sorted_features[:top_n]]
            inter_rankings.append(ranked_features)

        # Calculate Spearman's correlation for inter-model
        correlation_matrix = []
        for i in range(len(inter_rankings)):
            for j in range(i + 1, len(inter_rankings)):
                rank_i = np.arange(1, len(inter_rankings[i]) + 1)
                rank_j = np.array([rank_i[inter_rankings[i].index(f)] if f in inter_rankings[i] else len(rank_i) + 1 for f in inter_rankings[j]])
                spearman_corr, _ = spearmanr(rank_i, rank_j)
                correlation_matrix.append(spearman_corr)

        spearman_results_inter[xai_method] = np.mean(correlation_matrix)

    # Plot grouped barplot
    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    x = np.arange(len(xai_methods))  # Bar positions
    width = 0.35  # Width of each bar

    plt.figure(figsize=(12, 6))

    intra_values = [spearman_results_intra[method] for method in xai_methods]
    inter_values = [spearman_results_inter[method] for method in xai_methods]

    # Plot bars
    plt.bar(x - width / 2, intra_values, width, label='Intra-Model Aggregation', color=["#627ad1"])
    plt.bar(x + width / 2, inter_values, width, label='Inter-Model Aggregation', color=["#7ac293"])

    xai_names = ['DeepLiftShap', 'LIME', 'Integrated Gradients']
    plt.xticks(x, xai_names, fontsize=12)
    plt.ylabel("Spearman's Correlation", fontsize=12)
    plt.title(f"Spearman's Correlation: Intra vs Inter Model Aggregation", fontsize=15, fontweight="bold")
    plt.legend(fontsize=12)

    # Save and show the plot
    os.makedirs("synth_data/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/spearman_intra_inter_aggr_correlation_{top_n}features.png")
    plt.tight_layout()
    plt.show()

    return spearman_results_intra, spearman_results_inter


intra_inter_correlation()
