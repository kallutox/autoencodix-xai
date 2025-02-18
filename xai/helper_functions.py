import torch
#import shap
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os
import torch.nn as nn
from scipy.stats import ttest_ind
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_venn import venn3
from collections import Counter
import torch.nn.functional as F
from scipy.stats import f_oneway
import mygene
import warnings

# Suppress all warnings (due to gene name retrieval or LIME)
warnings.filterwarnings("ignore")

# project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
current_directory = os.path.dirname(os.path.abspath(__file__))


class VAEEncoderSingleDim(nn.Module):
    """A class to access a specific latent dimension of the VAE's encoder part directly"""

    def __init__(self, vae_model, dim):
        super(VAEEncoderSingleDim, self).__init__()
        # Accessing the required components from the original VAE model
        self.encoder = vae_model.encoder
        self.mu = vae_model.mu
        self.logvar = vae_model.logvar
        self.reparameterize = vae_model.reparameterize
        self.input_dim = vae_model.input_dim
        self.dim = dim  # latent dim

    def forward(self, x):
        #print(f"Input shape before view: {x.shape}")

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input with {self.input_dim} features, but got {x.shape[1]} features. "
                f"This may indicate missing data modalities or incorrect data preparation."
            )

        total_elements = x.numel()
        #print(f"Total elements: {total_elements}, Input_dim: {self.input_dim}")

        # Ensure the total number of elements is a multiple of input_dim
        assert total_elements % self.input_dim == 0, (
            f"Total elements {total_elements} is not a multiple of input_dim {self.input_dim}"
        )

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        latent = self.encoder(x)
        mu = self.mu(latent)
        logvar = self.logvar(latent)
        z = self.reparameterize(mu, logvar)
        output = z[:, self.dim]
        output = output.unsqueeze(1)  # Equivalent to output.reshape(output.shape[0], 1)
        return output


class VAEWrapper(nn.Module):
    """A class to access only the model's reconstructed output from a VAE model."""

    def __init__(self, model):
        super(VAEWrapper, self).__init__()
        self.vae_model = model

    def forward(self, x):
        # Ensure input tensor is correctly reshaped if necessary
        if x.dim() != 2 or x.shape[1] != self.vae_model.input_dim:
            x = x.view(-1, self.vae_model.input_dim)

        reconstructed, _, _ = self.vae_model(x)
        return reconstructed


class AEEncoderSingleDim(nn.Module):
    """A class to access a specific latent dimension of the AE's encoder part directly."""

    def __init__(self, ae_model, dim):
        super(AEEncoderSingleDim, self).__init__()
        self.encoder = ae_model.encoder  # Access the encoder directly
        self.dim = dim  # Specify which dimension of the latent space to access

    def forward(self, x):
        # Ensure x is properly shaped
        if x.dim() != 2 or x.shape[1] != self.encoder[0].in_features:
            x = x.view(-1, self.encoder[0].in_features)

        latent = self.encoder(x)  # Get the latent representation
        output = latent[:, self.dim]  # Access the specific latent dimension
        return output.unsqueeze(1)  # Keep output as [batch_size, 1]


class AEWrapper(nn.Module):
    """A class to access only the model's reconstructed output from a Vanilla Autoencoder."""

    def __init__(self, model):
        super(AEWrapper, self).__init__()
        self.ae_model = model

    def forward(self, x):
        # Assuming x might need reshaping to match input_dim if not already processed
        if x.dim() != 2 or x.shape[1] != self.ae_model.input_dim:
            x = x.view(-1, self.ae_model.input_dim)

        reconstructed, _, _ = self.ae_model(x)
        return reconstructed


def get_state_dict(run_id, data_types="RNA", model_type="varix"):
    parent_directory = os.path.abspath(os.path.join(current_directory, ".."))
    dict_path = os.path.join(
        parent_directory,
        "models",
        str(run_id),
        f"{data_types}_{model_type}{run_id}.pt",
    )

    state_dict = torch.load(dict_path, map_location=torch.device('cpu'))
    return state_dict


def get_config(run_id):
    config_path = os.path.join(os.path.abspath(os.path.join(current_directory, "..")), f"{run_id}_config.yaml")
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    return config_data


def get_feature_num(run_id):
    config_path = os.path.join(os.path.abspath(os.path.join(current_directory, "..")), f"{run_id}_config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        k_filter = config["K_FILTER"]

    return k_filter


# updated
def get_input_dim(state_dict):
    first_layer_weights = state_dict['encoder.0.weight']
    weight_input_dim = first_layer_weights.shape[1]
    return weight_input_dim


def get_processed_data(run_id, data_type="RNA"):
    parent_directory = os.path.abspath(os.path.join(current_directory, ".."))
    data_path = os.path.join(
        parent_directory,
        "data",
        "processed",
        str(run_id),
        f"{data_type}_data.parquet"
    )
    data = pd.read_parquet(data_path, engine="pyarrow")
    return data


def get_interim_data(run_id, model_type="varix"):
    parent_directory = os.path.abspath(os.path.join(current_directory, ".."))
    data_path = os.path.join(
        parent_directory,
        "data",
        "interim",
        str(run_id),
        f"combined_{model_type}_input.parquet"
    )
    data = pd.read_parquet(data_path, engine="pyarrow")
    return data


def get_raw_data(file_name, file_type="parquet"):
    parent_directory = os.path.abspath(os.path.join(current_directory, ".."))
    data_path = os.path.join(
        parent_directory,
        "data",
        "raw",
        f"{file_name}"
    )

    if file_type == "parquet":
        data = pd.read_parquet(data_path, engine="pyarrow")
    elif file_type == "txt" or file_type == "tsv" or file_type == "csv":
        data = pd.read_csv(data_path, sep="\t", index_col=[0], header=None)

    else:
        print("File type not available.")
        return
    return data


def get_random_data_split_arrays(data, background_n, test_n, seed=4598):
    np.random.seed(seed)
    random_state_bg = np.random.randint(0, 10000)
    df_part1 = data.sample(frac=0.8, random_state=random_state_bg)
    df_part2 = data.drop(df_part1.index)

    background_data = df_part1.sample(n=background_n, replace=False, random_state=seed)
    test_data = df_part2.sample(n=test_n, replace=False, random_state=seed)

    return background_data, test_data


def get_random_data_split_tensors(data, background_n, test_n, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    random_state_bg = np.random.randint(0, 10000)
    df_part1 = data.sample(frac=0.8, random_state=random_state_bg)
    df_part2 = data.drop(df_part1.index)

    background_data = df_part1.sample(n=background_n, replace=False, random_state=seed)
    test_data = df_part2.sample(n=test_n, replace=False, random_state=seed)

    test_tensor = torch.tensor(test_data.to_numpy()).float()

    background_tensor = torch.tensor(background_data.to_numpy()).float()

    return test_tensor, background_tensor


def get_latent_space(run_id):
    parent_directory = os.path.abspath(os.path.join(current_directory, ".."))
    latent_path = os.path.join(
        parent_directory,
        "reports",
        str(run_id),
        "predicted_latent_space.parquet",
    )

    latent_space = pd.read_parquet(latent_path, engine="pyarrow")
    return latent_space


def get_joined_data(run_id):
    parent_directory = os.path.abspath(os.path.join(current_directory, ".."))
    data_path = os.path.join(
        parent_directory,
        "data",
        "processed",
        str(run_id),
        "/",
    )

    all_data = None

    for filename in os.listdir(data_path):
        if filename.endswith(".parquet"):
            data_type = filename.replace("_data.parquet", "")

            file_path = os.path.join(data_path, filename)
            df = pd.read_parquet(file_path).add_prefix(f"{data_type}_")

            if all_data is None:
                all_data = df
            else:
                all_data = pd.concat([all_data, df], axis=1)

    return all_data


def attribution_per_feature(attributions, data):
    if attributions.ndim == 1:
        average_attributions = np.abs(attributions.detach().numpy())
    else:
        average_attributions = np.mean(np.abs(attributions.detach().numpy()), axis=0)

    #print("data: ", data)
    if isinstance(data, list):
        feature_names = data
    else:
        feature_names = list(data.columns)
    #print("feature names: ", feature_names)

    processed_feature_names = []
    if not isinstance(attributions, dict):
        for name in feature_names:
            if 'ENSG' in name:  # remove RNA_ prefix if it exists
                processed_name = name.replace('RNA_', '')
                processed_feature_names.append(processed_name)
            else:
                processed_feature_names.append(name)

    attribution_dict = dict(zip(processed_feature_names, average_attributions))

    return attribution_dict


def get_top_features(attribution_dict, top_n=10):
    sorted_features = sorted(attribution_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = [feature[0] for feature in sorted_features[:top_n]]

    return top_features


def top_n_attributions_with_plot(attribution_dict, top_n=10):
    """
    Get the top N features with their attributions, calculate the average for the rest with min/max range,
    and plot the result as a barplot.

    Parameters:
        attribution_dict (dict): A dictionary of feature names and their attributions.
        top_n (int): Number of top features to display.

    Returns:
        pd.DataFrame: A DataFrame containing the top N features and the average of the rest.
        The function also creates a bar plot of the attributions.
    """
    # Sort attributions in descending order
    sorted_attributions = sorted(attribution_dict.items(), key=lambda x: x[1], reverse=True)

    # Extract top N features
    top_features = sorted_attributions[:top_n]

    # Extract the rest of the features
    rest_features = sorted_attributions[top_n:]
    rest_attributions = [value for _, value in rest_features]

    # Calculate mean, min, and max for the rest
    rest_mean = np.mean(rest_attributions)
    rest_min = np.min(rest_attributions)
    rest_max = np.max(rest_attributions)

    # Create a DataFrame for the top N features
    data = pd.DataFrame(top_features, columns=["Feature", "Attribution"])
    data["Error"] = 0  # Set error to 0 for top features as individual errors are not calculated

    # Add the average of the rest as the last row using pd.concat()
    rest_row = pd.DataFrame({
        "Feature": ["Average of remaining features"],
        "Attribution": [rest_mean],
        "Error": [0],  # No error for bar height itself
        "Min": [rest_min],
        "Max": [rest_max]
    })
    data = pd.concat([data, rest_row], ignore_index=True)

    # Reverse the order for plotting (highest at the top, average at the bottom)
    data = data.iloc[::-1].reset_index(drop=True)

    # Plot the results
    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("pastel")
    bar_colors = [palette[i % len(palette)] for i in range(len(data))]

    # Plot the bars
    plt.barh(
        data["Feature"],
        data["Attribution"],
        color=bar_colors,
        alpha=0.9,
    )

    # Add min/max range as a horizontal line for the "Average of remaining features"
    rest_index = data[data["Feature"] == "Average of remaining features"].index[0]
    plt.hlines(
        y=rest_index + 0.5,  # Center the line horizontally for the last bar
        xmin=data.loc[rest_index, "Min"],
        xmax=data.loc[rest_index, "Max"],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Range (min-max)" if rest_index == len(data) - 1 else None,  # Add label only once
    )

    # Draw a vertical line at x=0
    plt.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Add labels and title
    plt.xlabel("Attribution Value", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Top {top_n} Features and Average of Remaining Features", fontsize=14)

    # Configure y-axis ticks
    plt.yticks(
        range(len(data)),
        data["Feature"],
        fontsize=12
    )

    # Add legend for the range line
    plt.legend(loc="lower right")

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return data


def get_cf_metadata(ensembl_ids=None):
    """
    Fetches metadata for the provided Ensembl IDs. If no IDs are provided, returns the entire metadata.

    Args:
        ensembl_ids (list or pd.Index, optional): List of Ensembl IDs for which metadata is to be fetched.

    Returns:
        dict: Dictionary with Ensembl IDs as keys and their metadata as values.
    """
    var_path = os.path.join(
        os.path.abspath(os.path.join(current_directory, "..")),
        "data",
        "raw",
        "cf_gene_metadata_formatted.parquet",
    )
    var_df = pd.read_parquet(var_path)

    if ensembl_ids is None:
        return var_df.to_dict(orient='index')

    if not isinstance(ensembl_ids, pd.Index):
        ensembl_ids = pd.Index(ensembl_ids)

    matched_var_info = var_df.reindex(ensembl_ids)
    var_info_dict = matched_var_info.to_dict(orient='index')

    return var_info_dict


def get_cf_clin_data():
    clin_path = os.path.join(
        os.path.abspath(os.path.join(current_directory, "..")),
        "data",
        "raw",
        "cf_clinical_data_formatted.parquet",
    )
    clin_df = pd.read_parquet(clin_path)
    return clin_df


def get_tcga_metadata(entrez_ids):
    """
    Maps feature names (Entrez Gene IDs) to Hugo symbols using the TCGA mutation dataset.

    Args:
        feature_names (pd.Index or list): Feature names (Entrez Gene IDs) from expr_tcga.

    Returns:
        dict: A dictionary mapping Entrez Gene IDs to their corresponding Hugo symbols.
    """
    mg = mygene.MyGeneInfo()

    stripped_ids = [gene_id.split('_')[1] for gene_id in entrez_ids]

    results = mg.querymany(stripped_ids, scopes="entrezgene", fields="symbol", species="human")

    id_to_name = {}
    for res, prefixed_id in zip(results, entrez_ids):
        if "symbol" in res:
            if prefixed_id.startswith("MUT_"):
                id_to_name[prefixed_id] = f"{res['symbol']} (mut.)"
            elif prefixed_id.startswith("METH_"):
                id_to_name[prefixed_id] = f"{res['symbol']} (meth.)"
            else:
                id_to_name[prefixed_id] = res["symbol"]
        else:
            id_to_name[prefixed_id] = 'not found'

    return id_to_name


def get_tcga_clin_data():
    clin_path = os.path.join(
        os.path.abspath(os.path.join(current_directory, "..")),
        "data",
        "raw",
        "data_clinical_formatted.parquet",
    )
    clin_df = pd.read_parquet(clin_path)
    return clin_df


def get_training_data(run_id, input_data):
    """
    Filters the input data to only include training samples for the given run_id.

    Parameters:
        run_id (str): Identifier for the current run.
        input_data (pd.DataFrame): The interim data containing all samples.
        sample_split_path (str): Path to the sample_split.parquet file.

    Returns:
        pd.DataFrame: Filtered input data containing only training samples.
    """

    sample_split_path = f"../data/processed/{run_id}/sample_split.parquet"
    sample_split = pd.read_parquet(sample_split_path)

    training_samples = sample_split[sample_split["SPLIT"] == "train"]["SAMPLE_ID"]

    # Filter the input_data based on the training samples
    filtered_input_data = input_data[
        input_data.index.isin(training_samples)
    ]

    return filtered_input_data


def get_best_dimension_by_sex_ttest(run_id):
    latent_df = get_latent_space(run_id)
    clin_df = get_cf_clin_data()
    merged_df = latent_df.merge(clin_df[['sex']], left_index=True, right_index=True)

    p_values = {}

    for column in merged_df.columns:
        if "L_COMBINED-RNA__varix_INPUT_" in column:
            latent_number = int(column.split("_")[-1])  # Convert latent dimension to integer
            male_values = merged_df[merged_df['sex'] == 'male'][column]
            female_values = merged_df[merged_df['sex'] == 'female'][column]

            t_stat, p_val = ttest_ind(male_values, female_values, equal_var=False)
            p_values[latent_number] = p_val  # Use the integer latent number as the key

            #print(f"Dimension {latent_number} - Males: {len(male_values)}, Females: {len(female_values)}")
            #print(f"p-value: {p_val}\n")

    best_dimension = min(p_values, key=p_values.get)
    return best_dimension


def get_best_dimension_by_sex_means(run_id):
    latent_df = get_latent_space(run_id)
    clin_df = get_cf_clin_data()
    merged_df = latent_df.merge(clin_df[['sex']], left_index=True, right_index=True)

    mean_differences = {}

    for column in merged_df.columns:
        if "L_COMBINED-RNA__varix_INPUT_" in column:
            latent_number = column.split("_")[-1]

            male_values = merged_df[merged_df['sex'] == 'male'][column]
            female_values = merged_df[merged_df['sex'] == 'female'][column]

            male_mean = male_values.mean()
            female_mean = female_values.mean()

            mean_diff = abs(male_mean - female_mean)

            mean_differences[int(latent_number)] = mean_diff

            #print(f"Dimension {latent_number} - Male Mean: {male_mean}, Female Mean: {female_mean}")
            #print(f"Absolute Difference in Means: {mean_diff}\n")

    best_dimension = max(mean_differences, key=mean_differences.get)
    #print(f"The best dimension based on mean difference is {best_dimension} with a difference of {mean_differences[best_dimension]}")
    return int(best_dimension)


def get_sex_specific_split(input_data, clin_data, test_n, ref_n, which_data='input', seed=None):
    """
    Select n random samples where sex == "male" as the test tensor and
    m random samples of all sexes as the reference tensor, ensuring no overlap.
    If ref_n == 1, the reference tensor contains the average values across all remaining indices.

    Parameters:
    - input_data (pd.DataFrame): Dataframe containing input data.
    - clin_data (pd.DataFrame): Dataframe containing clinical data with a 'sex' column.
    - test_n (int): Number of samples where sex == "male" for the test tensor.
    - ref_n (int): Number of samples for the reference tensor (all sexes) or 1 for the average reference tensor.
    - which_data (str): Whether to specify test samples based on 'input' or 'reference'.
    - seed (int): Random seed for reproducibility (default is None).

    Returns:
    - test_tensor (torch.Tensor): Tensor of n random samples where sex == "male".
    - reference_tensor (torch.Tensor): Tensor of m random samples or the average values tensor if ref_n == 1.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Ensure input and clin_data have matching indices
    clin_data = clin_data.loc[input_data.index]

    # Filter male samples
    male_samples = clin_data[clin_data["sex"] == "male"].index
    female_unknown_samples = clin_data[clin_data["sex"].isin(["female", "unknown"])].index

    if which_data == 'input':
        test_indices = male_samples.to_series().sample(n=test_n, random_state=seed).index

        # Reference indices
        remaining_indices = input_data.index.difference(test_indices)
        if ref_n == 1:
            reference_indices = remaining_indices
        else:
            reference_indices = remaining_indices.to_series().sample(n=ref_n, random_state=seed).index

    elif which_data == 'reference':
        reference_indices = male_samples.to_series().sample(n=ref_n, random_state=seed).index
        remaining_indices = input_data.index.difference(reference_indices)
        test_indices = remaining_indices.to_series().sample(n=test_n, random_state=seed).index

    else:
        raise ValueError(
            "'input' or 'reference' expected for which_data"
        )

    # Extract corresponding data for test and reference
    test_data = input_data.loc[test_indices]

    if ref_n == 1:
        # Compute the average values for reference data
        reference_data = input_data.loc[remaining_indices].mean(axis=0)
        reference_tensor = torch.tensor(reference_data.values.reshape(1, -1), dtype=torch.float32)
    else:
        reference_data = input_data.loc[reference_indices]
        reference_tensor = torch.tensor(reference_data.values, dtype=torch.float32)

    # Convert test data to tensor
    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)

    return test_tensor, reference_tensor


def get_cf_specific_split(input_data, clin_data, test_n, ref_n, seed=None):
    """
    Select n random samples where sex == "male" as the test tensor and
    m random samples of all sexes as the reference tensor, ensuring no overlap.
    If ref_n == 1, the reference tensor contains the average values across all remaining indices.

    Parameters:
    - input_data (pd.DataFrame): Dataframe containing input data.
    - clin_data (pd.DataFrame): Dataframe containing clinical data with a 'sex' column.
    - test_n (int): Number of samples where sex == "male" for the test tensor.
    - ref_n (int): Number of samples for the reference tensor (all sexes) or 1 for the average reference tensor.
    - which_data (str): Whether to specify test samples based on 'input' or 'reference'.
    - seed (int): Random seed for reproducibility (default is None).

    Returns:
    - test_tensor (torch.Tensor): Tensor of n random samples where sex == "male".
    - reference_tensor (torch.Tensor): Tensor of m random samples or the average values tensor if ref_n == 1.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # ensure input and clin_data have matching indices
    clin_data = clin_data.loc[input_data.index]

    cf_samples = clin_data[clin_data['disease'] == 'cystic fibrosis'].index
    normal_samples = clin_data[clin_data['disease'] == 'normal'].index

    test_indices = cf_samples.to_series().sample(n=test_n, random_state=seed).index

    remaining_indices = input_data.index.difference(test_indices)
    if ref_n == 1:
        reference_indices = remaining_indices
    else:
        reference_indices = remaining_indices.to_series().sample(n=ref_n, random_state=seed).index

    test_data = input_data.loc[test_indices]

    if ref_n == 1:
        # compute the average values for reference data
        reference_data = input_data.loc[remaining_indices].mean(axis=0)
        reference_tensor = torch.tensor(reference_data.values.reshape(1, -1), dtype=torch.float32)
    else:
        reference_data = input_data.loc[reference_indices]
        reference_tensor = torch.tensor(reference_data.values, dtype=torch.float32)

    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)

    return test_tensor, reference_tensor


def get_cancer_specific_split(input_data, clin_data, cancer_type, test_n, ref_n, seed=None):
    """
    Select n random samples where CANCER_TYPE_ACRONYM == cancer_type as the test tensor
    and m random samples from all cancer types as the reference tensor, ensuring no overlap.
    If ref_n == 1, the reference tensor contains the average values across all remaining indices.

    Parameters:
    - input_data (pd.DataFrame): Dataframe containing input data.
    - clin_data (pd.DataFrame): Dataframe containing clinical data with a 'CANCER_TYPE_ACRONYM' column.
    - cancer_type (str): Cancer type acronym to specify the test group.
    - test_n (int): Number of samples where CANCER_TYPE_ACRONYM == cancer_type for the test tensor.
    - ref_n (int): Number of samples for the reference tensor (all cancer types) or 1 for the average reference tensor.
    - seed (int): Random seed for reproducibility (default is None).

    Returns:
    - test_tensor (torch.Tensor): Tensor of n random samples where CANCER_TYPE_ACRONYM == cancer_type.
    - reference_tensor (torch.Tensor): Tensor of m random samples or the average values tensor if ref_n == 1.
    """
    if seed is not None:
        torch.manual_seed(seed)

    clin_data = clin_data.loc[input_data.index]
    cancer_samples = clin_data[clin_data['CANCER_TYPE_ACRONYM'] == cancer_type].index
    #print(f"Available samples for {cancer_type}: {len(cancer_samples)}")

    test_indices = cancer_samples.to_series().sample(n=test_n, random_state=seed).index

    remaining_indices = input_data.index.difference(test_indices)
    if ref_n == 1:
        reference_indices = remaining_indices
    else:
        reference_indices = remaining_indices.to_series().sample(n=ref_n, random_state=seed).index

    test_data = input_data.loc[test_indices]

    if ref_n == 1:
        reference_data = input_data.loc[remaining_indices].mean(axis=0)
        reference_tensor = torch.tensor(reference_data.values.reshape(1, -1), dtype=torch.float32)
    else:
        reference_data = input_data.loc[reference_indices]
        reference_tensor = torch.tensor(reference_data.values, dtype=torch.float32)

    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)

    return test_tensor, reference_tensor


def bar_plot_top_features(attributions, data, top_n=15, dataset='synth', xai_method='deepshaplift', cancer_type='LUAD', beta=0.01, n=15):
    attribution_dict = attribution_per_feature(attributions, data)
    top_features = get_top_features(attribution_dict, top_n=top_n)

    # Prepare data for plotting
    top_features_names = [feature for feature in top_features]
    top_features_scores = [attribution_dict[feature] for feature in top_features]

    # Reverse for horizontal barplot (highest score at the top)
    top_features_names.reverse()
    top_features_scores.reverse()

    # Set seaborn style and context
    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    norm_scores = (np.array(top_features_scores) - min(top_features_scores)) / (max(top_features_scores) - min(top_features_scores))
    cmap = LinearSegmentedColormap.from_list(
        "custom_gradient",
        ["#4354b5", "#43a2b5", "#43b582"]
    )
    colors = cmap(norm_scores)
    if xai_method == 'deepliftshap':
        xai_name = 'DeepLiftShap'
    elif xai_method == 'lime':
        xai_name = 'LIME'
    elif xai_method == 'integrated_gradients':
        xai_name = 'Integrated Gradients'
    else:
        xai_name = None

    # Create the barplot with a gradient
    plt.figure(figsize=(8, 6))
    for i, (name, score, color) in enumerate(zip(top_features_names, top_features_scores, colors)):
        plt.barh(name, score, color=color, alpha=0.9)

    plt.xlabel("Attribution Value", fontsize=12)
    if dataset == 'tcga':
        plt.title(f"{xai_name} - Top {n} Features by Attribution Value for {cancer_type}", fontsize=14, fontweight="bold")
    else:
        plt.title(f"{xai_name} - Top {n} Features by Attribution Value", fontsize=14, fontweight="bold")
    plt.tight_layout()
    # Save and show the plot
    if dataset == 'synth':
        os.makedirs("synth_reports/figures", exist_ok=True)
        plt.savefig(f"synth_reports/figures/top_features_stacked_barplot.png")
    elif dataset == 'cf':
        os.makedirs("cf_reports", exist_ok=True)
        plt.savefig(f"cf_reports/barplot_top{n}_{xai_method}_{beta}.png")
    elif dataset == 'tcga':
        os.makedirs("tcga_reports", exist_ok=True)
        plt.savefig(f"tcga_reports/barplot_top{n}_{xai_method}_{cancer_type}_{beta}.png")
    #plt.show()


# TODO - does not work
def stacked_bar_plot_top_features(attributions, data, clin_data, feature_names, dataset='synth', cancer_type='LUAD', top_n=10):
    try:
        # Attribution dictionary
        attribution_dict = attribution_per_feature(attributions, data)
        top_features = get_top_features(attribution_dict, top_n=top_n)

        # Map feature IDs to gene names
        gene_metadata = get_cf_metadata(feature_names)
        feature_id_to_gene_name = {id: gene_metadata[id]['feature_name'] for id in feature_names if id in gene_metadata}

        # Prepare data for plotting
        top_feature_ids = [feature for feature in top_features]  # Feature IDs
        top_features_scores = [attribution_dict[feature] for feature in top_features]
        top_feature_names = [feature_id_to_gene_name.get(feature, feature) for feature in top_feature_ids]

        # Reverse for horizontal barplot (highest score at the top)
        top_feature_ids.reverse()
        top_features_scores.reverse()
        top_feature_names.reverse()

        # Calculate contributions by lung conditions
        lung_conditions = clin_data['lung_condition'].unique()
        condition_contributions = {condition: [] for condition in lung_conditions}

        for feature_id in top_feature_ids:
            for condition in lung_conditions:
                # Get clinical data for this condition
                condition_data = clin_data[clin_data['lung_condition'] == condition]

                # Calculate the average contribution of this feature in this condition
                if feature_id in condition_data.columns:
                    avg_contribution = condition_data[feature_id].mean()
                else:
                    avg_contribution = 0
                condition_contributions[condition].append(avg_contribution)

        # Normalize contributions to ensure stacked percentages
        for i in range(len(top_feature_ids)):
            total = sum(condition_contributions[cond][i] for cond in lung_conditions)
            if total > 0:
                for cond in lung_conditions:
                    condition_contributions[cond][i] /= total

        # Stacked bar plot
        colors = ["#4354b5", "#43a2b5", "#43b582", "#FF6347", "#FFD700"]  # Add more colors if needed
        condition_colors = {cond: color for cond, color in zip(lung_conditions, colors)}

        plt.figure(figsize=(10, 8))
        bottom = np.zeros(len(top_feature_names))
        for condition, color in condition_colors.items():
            plt.barh(top_feature_names, condition_contributions[condition], color=color, label=condition, left=bottom)
            bottom += np.array(condition_contributions[condition])

        # Add labels and title
        plt.xlabel("Attribution Value (normalized)", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.title(f"Top {top_n} Features by Attribution Value with Lung Condition Contributions", fontsize=14, fontweight="bold")
        plt.legend(title="Lung Conditions", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save and show plot

        if dataset == 'synth':
            os.makedirs("synth_reports/figures", exist_ok=True)
            plt.savefig(f"synth_reports/figures/top_features_stacked_barplot.png")
        elif dataset == 'cf':
            os.makedirs("cf_reports", exist_ok=True)
            plt.savefig(f"cf_reports/venn_diagram_all_explainers.png")
        elif dataset == 'tcga':
            os.makedirs("tcga_reports", exist_ok=True)
            plt.savefig(f"tcga_reports/{cancer_type}_venn_diagram_all_explainers.png")
        plt.show()

    except Exception as e:
        print("An error occurred during plotting:", e)


def zero_counts_per_feature(attributions, test_tensor):
    """
    Calculate the percentage of samples with zero attribution scores for each feature.

    Args:
        attributions (torch.Tensor): Attribution scores with shape [num_samples, num_features].
        test_tensor (torch.Tensor): The input tensor for which the attributions were calculated.

    Returns:
        zero_percentage (list): Percentage of zero attributions per feature.
    """
    # Ensure the attributions are a tensor if not already
    if not isinstance(attributions, torch.Tensor):
        attributions = torch.tensor(attributions, dtype=torch.float32)

    # Boolean mask where attributions are zero
    zero_mask = (attributions == 0)

    # Count the number of zero scores per feature
    zero_counts = zero_mask.sum(dim=0).numpy()  # Summing across samples (rows)

    # Calculate percentage of zeros
    num_samples = test_tensor.shape[0]
    zero_percentage = (zero_counts / num_samples) * 100

    return zero_percentage


def attr_dict_all_samples(attribution_values, feature_names):
    attribution_scores_dict = {}

    for feature_idx, feature_name in enumerate(feature_names):
        attribution_scores_dict[feature_name] = attribution_values[:, feature_idx].detach().numpy().tolist()

    return attribution_scores_dict


def calculate_zero_score_percentage(attr_dict):
    zero_percentage_dict = {}

    for feature, scores in attr_dict.items():
        if isinstance(scores, (list, np.ndarray)):  # Handle lists or arrays of scores
            total_scores = len(scores)
            zero_count = np.sum(np.array(scores) == 0)
        else:  # Handle single score
            total_scores = 1
            zero_count = 1 if scores == 0 else 0

        zero_percentage = (zero_count / total_scores) * 100
        zero_percentage_dict[feature] = zero_percentage

    #features_with_zero_scores_dict = {feature: percentage for feature, percentage in zero_percentage_dict.items() if percentage > 0}
    return zero_percentage_dict


def separate_zero_score_percentage(zero_percentage_dict, feature_list):
    feature_set = set(feature_list)

    specified_features = {feature: zero_percentage_dict[feature] for feature in zero_percentage_dict if feature in feature_set}
    remaining_features = {feature: zero_percentage_dict[feature] for feature in zero_percentage_dict if feature not in feature_set}

    return specified_features, remaining_features


def get_best_dimension_cf(run_id):
    latent_df = get_latent_space(run_id)
    clin_df = get_cf_clin_data()
    merged_df = latent_df.merge(clin_df[['disease']], left_index=True, right_index=True)

    mean_differences = {}

    for column in merged_df.columns:
        if "L_COMBINED-RNA__varix_INPUT_" in column:
            latent_number = column.split("_")[-1]

            disease_values = merged_df[merged_df['disease'] == 'cystic fibrosis'][column]
            control_values = merged_df[merged_df['disease'] == 'normal'][column]

            cf_mean = disease_values.mean()
            normal_mean = control_values.mean()
            mean_diff = abs(cf_mean - normal_mean)
            mean_differences[int(latent_number)] = mean_diff

    best_dimension = max(mean_differences, key=mean_differences.get)
    return int(best_dimension)


def get_best_dimension_tcga_by_cancer_means(run_id, cancer_type_acronym):
    cancer_list = ['CHOL', 'OV', 'CESC', 'UCS', 'PRAD', 'DLBC', 'PCPG', 'KIRC', 'UVM', 'BLCA', 'STAD', 'UCEC', 'LGG',
                   'KIRP', 'READ', 'COAD', 'LIHC', 'LUAD', 'GBM', 'THCA', 'PAAD', 'SARC', 'MESO', 'ACC', 'HNSC', 'ESCA',
                   'LUSC', 'SKCM', 'KICH', 'BRCA', 'THYM', 'LAML', 'TGCT']

    if cancer_type_acronym not in cancer_list:
        print(f"Cancer type acronym {cancer_type_acronym} invalid.")
        sys.exit()

    latent_df = get_latent_space(run_id)
    clin_df = get_tcga_clin_data()

    merged_df = latent_df.merge(clin_df[['CANCER_TYPE_ACRONYM']], left_index=True, right_index=True)

    mean_differences = {}

    for column in merged_df.columns:
        if "L_COMBINED-RNA__varix_INPUT_" in column:
            # extract the latent dimension number from the column name
            latent_number = int(column.split("_")[-1])

            cancer_values = merged_df[merged_df['CANCER_TYPE_ACRONYM'] == cancer_type_acronym][column]
            rest_values = merged_df[merged_df['CANCER_TYPE_ACRONYM'] != cancer_type_acronym][column]

            cancer_mean = cancer_values.mean()
            rest_mean = rest_values.mean()
            mean_diff = abs(cancer_mean - rest_mean)
            mean_differences[latent_number] = mean_diff

    best_dimension = max(mean_differences, key=mean_differences.get)
    return best_dimension


def get_best_dimension_by_cancer_anova(run_id, cancer_type_acronym):
    """
    Identifies the latent dimension that best separates a specific cancer type from all other cancer types
    based on one-way ANOVA tests.

    Args:
        run_id (str): Identifier for the specific run.
        cancer_type_acronym (str): The acronym of the cancer type to analyze.

    Returns:
        int: The latent dimension with the highest F-statistic.
    """
    # List of valid cancer types
    cancer_list = ['CHOL', 'OV', 'CESC', 'UCS', 'PRAD', 'DLBC', 'PCPG', 'KIRC', 'UVM', 'BLCA', 'STAD', 'UCEC', 'LGG',
                   'KIRP', 'READ', 'COAD', 'LIHC', 'LUAD', 'GBM', 'THCA', 'PAAD', 'SARC', 'MESO', 'ACC', 'HNSC', 'ESCA',
                   'LUSC', 'SKCM', 'KICH', 'BRCA', 'THYM', 'LAML', 'TGCT']

    if cancer_type_acronym not in cancer_list:
        raise ValueError(f"Invalid cancer type: {cancer_type_acronym}.")

    latent_df = get_latent_space(run_id)
    clin_df = get_tcga_clin_data()

    merged_df = latent_df.merge(clin_df[['CANCER_TYPE_ACRONYM']], left_index=True, right_index=True)

    mean_differences = {}

    for column in merged_df.columns:
        if "L_COMBINED-RNA_METH_MUT__varix_INPUT_" in column:
            latent_number = int(column.split("_")[-1])

            cancer_values = merged_df[merged_df['CANCER_TYPE_ACRONYM'] == cancer_type_acronym][column].dropna()
            others_values = merged_df[merged_df['CANCER_TYPE_ACRONYM'] != cancer_type_acronym][column].dropna()

            if len(cancer_values) > 1 and len(others_values) > 1:
                # perform one-way ANOVA
                f_stat, p_value = f_oneway(cancer_values, others_values)
                mean_differences[latent_number] = f_stat

    if not mean_differences:
        raise ValueError("No valid dimensions found for ANOVA.")

    # find the dimension with the highest F-statistic
    best_dimension = max(mean_differences, key=mean_differences.get)
    return best_dimension


def get_best_dimension_by_cancer_lda(run_id, cancer_type_acronym):
    cancer_list = ['CHOL','OV','CESC','UCS','PRAD','DLBC','PCPG','KIRC','UVM','BLCA','STAD','UCEC','LGG','KIRP','READ',
                   'COAD','LIHC','LUAD','GBM','THCA','PAAD','SARC','MESO','ACC','HNSC','ESCA','LUSC','SKCM','KICH','BRCA',
                   'THYM','LAML','TGCT']
    if cancer_type_acronym not in cancer_list:
        raise ValueError("Invalid cancer type.")
    latent_df = get_latent_space(run_id)
    clin_df = get_tcga_clin_data()
    merged_df = latent_df.merge(clin_df[['CANCER_TYPE_ACRONYM']], left_index=True, right_index=True)
    fisher_scores = {}
    for column in merged_df.columns:
        if "L_COMBINED-METH_MUT_RNA__varix_INPUT_" in column:
            latent_number = int(column.split("_")[-1])
            cancer_values = merged_df[merged_df['CANCER_TYPE_ACRONYM'] == cancer_type_acronym][column].dropna()
            others_values = merged_df[merged_df['CANCER_TYPE_ACRONYM'] != cancer_type_acronym][column].dropna()
            if len(cancer_values) > 1 and len(others_values) > 1:
                mu_c = cancer_values.mean()
                mu_o = others_values.mean()
                var_c = cancer_values.var(ddof=1)
                var_o = others_values.var(ddof=1)
                fisher_ratio = (mu_c - mu_o) ** 2 / (var_c + var_o)
                fisher_scores[latent_number] = fisher_ratio
    if not fisher_scores:
        raise ValueError("No valid dimensions found for LDA.")
    best_dimension = max(fisher_scores, key=fisher_scores.get)
    return best_dimension


def feature_overlap(dls_list, lime_list, ig_list, dataset='cf'):
    if dataset == 'cf':
        gene_metadata_dls = get_cf_metadata(dls_list)
        gene_metadata_lime = get_cf_metadata(lime_list)
        gene_metadata_ig = get_cf_metadata(ig_list)

        gene_names_dls = [gene_metadata_dls[id]['feature_name'] for id in dls_list if id in gene_metadata_dls]
        gene_names_lime = [gene_metadata_lime[id]['feature_name'] for id in lime_list if id in gene_metadata_lime]
        gene_names_ig = [gene_metadata_ig[id]['feature_name'] for id in ig_list if id in gene_metadata_ig]

    elif dataset == 'tcga':
        gene_metadata_dls = get_tcga_metadata(dls_list)
        gene_metadata_lime = get_tcga_metadata(lime_list)
        gene_metadata_ig = get_tcga_metadata(ig_list)

        gene_names_dls = [gene_metadata_dls[id] for id in dls_list if id in gene_metadata_dls]
        gene_names_lime = [gene_metadata_lime[id] for id in lime_list if id in gene_metadata_lime]
        gene_names_ig = [gene_metadata_ig[id] for id in ig_list if id in gene_metadata_ig]
    else:
        print(f"{dataset} invalid, please choose cf or tcga instead.")
        sys.exit()

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


def plot_venn_diagram(dls_set, lime_set, ig_set, beta, n, show=True, dataset='cf', cancer_type='LUAD'):
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

    if beta == 0.01:
        beta_val = '001'
    elif beta == 1:
        beta_val = '1'
    else:
        print('beta invalid (venn diagram)')
        sys.exit()

    if dataset == 'cf':
        os.makedirs("cf_reports", exist_ok=True)
        plt.savefig(f"cf_reports/venn_diagram_top{n}_{beta_val}.png")
    elif dataset == 'tcga':
        os.makedirs("tcga_reports", exist_ok=True)
        plt.savefig(f"tcga_reports/venn_diagram_top{n}_{cancer_type}_{beta_val}.png")
    else:
        print(f"{dataset} not a valid dataset, choose cf or tcga instead.")
        sys.exit()

    if show:
        plt.show()


def plot_attribution_histogram(attribution_values, beta, xai_method='deepliftshap', show=True, dataset='cf', cancer_type='LUAD'):
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
    plt.tight_layout()

    if beta == 0.01:
        beta_val = '001'
    elif beta == 1:
        beta_val = '1'
    else:
        print('beta invalid.')
        sys.exit()

    if dataset == 'cf':
        os.makedirs("cf_reports", exist_ok=True)
        plt.savefig(f"cf_reports/attribution_histogram_{xai_method}_{beta_val}.png")
    elif dataset == 'tcga':
        os.makedirs("tcga_reports", exist_ok=True)
        plt.savefig(f"tcga_reports/attribution_histogram_{xai_method}_{cancer_type}_{beta_val}.png")
    else:
        print(f"{dataset} not a valid dataset, choose cf or tcga instead.")
        sys.exit()

    if show:
        plt.show()


def get_gene_rank(gene_name, gene_list):
    if gene_name in gene_list:
        position = gene_list.index(gene_name)
        print(f"{gene_name} position: {position+1}")
    else:
        position = None
        #print(f"{gene_name} is not in the list.")

    return position