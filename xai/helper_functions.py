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
import torch.nn.functional as F

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

    background_tensor = torch.tensor(background_data.to_numpy()).float()
    test_tensor = torch.tensor(test_data.to_numpy()).float()

    return background_tensor, test_tensor


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


def attribution_per_feature(attributions, data, dataset='tcga'):
    if attributions.ndim == 1:
        average_attributions = np.abs(attributions.detach().numpy())
    else:
        average_attributions = np.mean(np.abs(attributions.detach().numpy()), axis=0)

    # process feature names if dataset is 'cf'
    feature_names = list(data.columns)
    if dataset == 'cf':
        processed_feature_names = []
        for name in feature_names:
            if 'ENSG' in name:
                processed_name = name.replace('RNA_', '')
                processed_feature_names.append(processed_name)
            else:
                processed_feature_names.append(name)
    else:
        processed_feature_names = feature_names

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


def get_cf_metadata(ensembl_ids):
    var_path = os.path.join(
        os.path.abspath(os.path.join(current_directory, "..")),
        "data",
        "raw",
        "cf_gene_metadata_formatted.parquet",
    )
    var_df = pd.read_parquet(var_path)
    if not isinstance(ensembl_ids, pd.Index):
        ensembl_ids = pd.Index(ensembl_ids)

    #matched_var_info = var_df.reindex[ensembl_ids]
    matched_var_info = var_df.reindex(ensembl_ids)

    var_info_dict = matched_var_info.to_dict(orient='index')

    return var_info_dict


def get_cf_clin_data(run_id):
    clin_path = os.path.join(
        os.path.abspath(os.path.join(current_directory, "..")),
        "data",
        "raw",
        "cf_clinical_data_formatted.parquet",
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


def get_best_dimension_ttest(run_id):
    latent_df = get_latent_space(run_id)
    clin_df = get_cf_clin_data(run_id)
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


def get_best_dimension_means(run_id):
    latent_df = get_latent_space(run_id)
    clin_df = get_cf_clin_data(run_id)
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


def get_sex_matched_split(input_data, clin_data, background_n, test_n, random_state=None):
    """
    Splits input_data into background and test sets with sex distribution matching.

    Args:
        input_data (pd.DataFrame): Input data with features (rows: samples, columns: features).
        clin_data (pd.DataFrame): Clinical data containing a 'sex' column aligned with input_data.
        background_n (int): Number of samples for the background dataset.
        test_n (int): Number of samples for the test dataset.
        random_state (int): Seed for reproducibility.

    Returns:
        background_tensor (torch.Tensor): Background dataset matching sex distribution.
        test_tensor (torch.Tensor): Test dataset matching sex distribution.
    """
    np.random.seed(random_state)

    # Merge input_data with clin_data to ensure alignment
    merged_data = input_data.join(clin_data['sex'])

    # Calculate sex proportions
    sex_counts = merged_data['sex'].value_counts(normalize=True)
    male_fraction = sex_counts.get('male', 0)
    female_fraction = sex_counts.get('female', 0)

    # Separate data by sex
    male_data = merged_data[merged_data['sex'] == 'male'].drop('sex', axis=1)
    female_data = merged_data[merged_data['sex'] == 'female'].drop('sex', axis=1)

    # Sample background data
    male_background = male_data.sample(int(background_n * male_fraction), random_state=random_state)
    female_background = female_data.sample(int(background_n * female_fraction), random_state=random_state)
    background_data = pd.concat([male_background, female_background])

    # Sample test data
    male_test = male_data.sample(int(test_n * male_fraction), random_state=random_state)
    female_test = female_data.sample(int(test_n * female_fraction), random_state=random_state)
    test_data = pd.concat([male_test, female_test])

    # Convert to PyTorch tensors
    background_tensor = torch.tensor(background_data.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)

    return background_tensor, test_tensor


def get_sex_specific_split(input_data, clin_data, test_n, ref_n, which_data='input', seed=None):
    """
    Select n random samples where sex == "male" as the test tensor and
    m random samples of all sexes as the reference tensor, ensuring no overlap.

    Parameters:
    - input_data (pd.DataFrame): Dataframe containing input data.
    - clin_data (pd.DataFrame): Dataframe containing clinical data with a 'sex' column.
    - n (int): Number of samples where sex == "male" for the test tensor.
    - m (int): Number of samples for the reference tensor (all sexes).
    - seed (int): Random seed for reproducibility (default is None).

    Returns:
    - test_tensor (torch.Tensor): Tensor of n random samples where sex == "male".
    - reference_tensor (torch.Tensor): Tensor of m random samples of all sexes.
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

        # reference with no male samples
        #reference_indices = female_unknown_samples.to_series().sample(n=ref_n, random_state=seed).index

        # reference random
        remaining_indices = input_data.index.difference(test_indices)
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
    reference_data = input_data.loc[reference_indices]

    # Convert to torch tensors
    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)
    reference_tensor = torch.tensor(reference_data.values, dtype=torch.float32)

    return test_tensor, reference_tensor



