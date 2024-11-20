import glob
import re
import json
from helper_functions import *
from xai_methods import captum_importance_values


def feature_importance_across_models(base_folders, xai_method='deepliftshap'):
    """
    Calculates and plots the number of synthetic features in the top 10 for one or multiple base folders,
    including the variance of values by signal strength.

    Args:
    base_folders (list): List of base folders (e.g., ['base1', 'base2', 'base3']) containing configuration files for each model.

    """
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
            data_set = 'cf'
            latent_dim = get_best_dimension_means(run_id)

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
                get_interim_data(run_id, model_type),
                dataset=data_set
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
        for i in range(num_configs_per_signal):
            counts_per_base = [len(positions[i]) for positions in aggregated_top_10_positions]
            average_counts.append(np.mean(counts_per_base))
            std_dev_counts.append(np.std(counts_per_base))
    else:
        average_counts = [len(positions) for positions in aggregated_top_10_positions[0]]
        std_dev_counts = [0] * len(average_counts)  # No variance with a single base

    # define signal strengths based on number of configs
    signal_strength = [0.1 * i for i in range(1, num_configs_per_signal + 1)]

    plt.figure(figsize=(10, 6))
    plt.bar(signal_strength, average_counts, width=0.05, align='center', yerr=std_dev_counts, capsize=5)
    plt.xlabel('Signal Strength')
    plt.ylabel('Number of Synthetic Features in Top 10' + (' (Average)' if len(base_folders) > 1 else ''))
    plt.title(f'Synthetic Features in Top 10 for {xai_method} {" (Averaged)" if len(base_folders) > 1 else ""}')
    plt.xticks(np.arange(0.1, 1.0, 0.1))

    # save the plot
    base_names = "_".join(base_folders)
    os.makedirs("synth_data/figures", exist_ok=True)
    plt.savefig(f"synth_reports/figures/synth_data_{base_names}_{xai_method}_male_target.png")
    plt.show()

    # save the averaged positions data
    data_to_save = {
        "signal_strength": signal_strength,
        "average_counts": average_counts,
        "std_dev_counts": std_dev_counts,
        "aggregated_top_10_positions": aggregated_top_10_positions
    }
    os.makedirs("synth_data", exist_ok=True)
    with open(f"synth_reports/synth_data_{base_names}_{xai_method}_male_target.json", "w") as f:
        json.dump(data_to_save, f)


def feature_importance_and_visualize(run_id, data_set, xai_method='deepliftshap', scatterplot=True, synth_test=False):

    attribution_values = captum_importance_values(
        run_id=run_id,
        data_types='rna',
        model_type='varix',
        dimension=get_best_dimension_means(run_id),
        latent_space_explain=True,
        xai_method=xai_method,
        visualize=scatterplot,
        return_delta=False
    )

    attribution_dict = attribution_per_feature(
        attribution_values,
        get_interim_data(run_id, 'varix'),
        dataset=data_set
    )

    #top_n_attributions_with_plot(attribution_dict, top_n=10)

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


def overlap_plot(intra_run_id, top_n=100):

    # intra model: same model, calculate feature importance 5 times for each method (15x)
    


