import torch
import shap
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os
import torch.nn as nn
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

    state_dict = torch.load(dict_path)
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


def get_random_data_split_tensors(data, background_n, test_n, seed=4598):
    np.random.seed(seed)
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


def baseline_add_check(vae_model, data, background_n_arr, n_per_baseline=10):
    avg_md_per_background = []
    not_none_n = 0
    for i in background_n_arr:
        avg_md = []

        for _ in range(n_per_baseline):
            background_tensor, test_tensor = get_random_data_split_tensors(
                data, background_n=i, test_n=20
            )
            e = shap.DeepExplainer(vae_model, background_tensor)
            max_diff = e.shap_values(test_tensor, check_additivity=True)
            if max_diff is not None:
                avg_md.append(max_diff.item())
                not_none_n += 1
        avg_md_per_background.append(sum(avg_md) / len(avg_md))
        # if not_none_n < n_per_baseline:
        #     print("Not all values raised assertion.")
        print(f"Max. difference for background_n {i} : {max(avg_md)}")
        print(f"Min. difference for background_n {i} : {min(avg_md)}")

    # print(f"Number of values < 0.01: ", not_none_n)
    print(f"Average value per background_n: ", avg_md_per_background)

    # plt.plot(background_n_arr, avg_md_per_background)
    # plt.xlabel('baseline_n')
    # plt.ylabel('average max. diff')
    # plt.title(f'test_n: 20, features: 400, latent dim: 16, beta: 0.01')
    # plt.show()


def get_top_features(attributions, data, dataset='tcga', top_n=10):
    #average_attributions = np.mean(np.abs(attributions.detach().numpy()), axis=0)
    if attributions.ndim == 1:
        average_attributions = np.abs(attributions.detach().numpy())
    else:
        average_attributions = np.mean(np.abs(attributions.detach().numpy()), axis=0)

    # match attribution values to genes
    result = pd.concat(
        [pd.DataFrame(data.columns), pd.DataFrame(average_attributions)], axis=1
    )
    result.columns = ["feature_name", "importance_value"]
    sorted_results = result.sort_values("importance_value", ascending=False)
    top_feature_names = sorted_results["feature_name"].head(top_n).tolist()

    if dataset == 'cf':
        top_feature_names_final = []
        # if data == cf data
        for name in top_feature_names:
            if 'ENSG' in name:
                processed_name = name.replace('RNA_', '')  # Remove the 'RNA_' prefix
                top_feature_names_final.append(processed_name)
            else:
                top_feature_names_final.append(name)

        return top_feature_names_final
    else:
        return top_feature_names


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

    # Select the matching rows in var_df
    matched_var_info = var_df.loc[ensembl_ids]

    # Convert the DataFrame to a dictionary of dictionaries
    var_info_dict = matched_var_info.to_dict(orient='index')

    return var_info_dict

