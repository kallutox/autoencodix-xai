import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helper_functions import *
from matplotlib.colors import LinearSegmentedColormap

data_cf = pd.read_parquet("../data/raw/cf_clinical_data_formatted.parquet")
data_tcga = pd.read_parquet("../data/raw/data_clinical_formatted.parquet")

sex_counts_tcga = data_tcga['SEX'].value_counts()
sex_counts_cf = data_cf['sex'].value_counts()
cf_perc = data_cf['sex'].value_counts(normalize=True) * 100
tcga_perc = data_tcga['SEX'].value_counts(normalize=True) * 100

sample_split = pd.read_parquet("../data/processed/synth_data_10features_09signal/sample_split.parquet")
#print(sample_split.head())


def example_plot():
    # Example data format (replace with your actual data)
    data = pd.DataFrame({
        "latent_dim": np.repeat([f"DIM_{i}" for i in range(1, 9)], 50),
        "latent_intensity": np.random.gamma(2, 2, 400),  # Non-zero variance
        "sex": np.tile(["Female", "Male", "Unknown"], 400 // 3 + 1)[:400]
    })

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"lines.linewidth": 3})

    # Define a modern color palette
    palette = ["#4354b5", "#43a2b5", "#43b582"]
    palette = {
        "Female": "#4354b5",  # Fully opaque
        "Male": "#43b582",  # Fully opaque
        "Unknown": "#43b582" # RGBA for transparency (50%)
    }

    # Create the ridge plot
    g = sns.FacetGrid(
        data,
        row="latent_dim",
        hue="sex",
        aspect=3,
        height=1,
        palette=palette,
        sharex=False,
        sharey=False,
        margin_titles=False
    )
    g.map(sns.kdeplot, "latent_intensity", alpha=0.7, fill=True, warn_singular=False)

    g.set(xlim=(-3, 12))
    for ax in g.axes.flat:
        ax.set_title("")
    g.set_titles("{row_name}", loc="left", size=11)
    g.set_axis_labels("Latent Intensity", "")
    g.set(yticks=[], ylabel="")
    g.despine(left=True)

    # Adjust spacing and position of the legend
    g.fig.subplots_adjust(hspace=0.5)
    g.add_legend(title="Latent Dimension - sex", loc="upper right", bbox_to_anchor=(1, 0.7))

    plt.tight_layout()
    plt.show()


run_id = 'synth_data_10features_09signal_base1'
example_plot()

def get_cf_metadata_old(ensembl_ids=None):
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
        "combined_data_var.parquet",
    )
    var_df = pd.read_parquet(var_path)

    if ensembl_ids is None:
        return var_df.to_dict(orient='index')

    if not isinstance(ensembl_ids, pd.Index):
        ensembl_ids = pd.Index(ensembl_ids)

    matched_var_info = var_df.reindex(ensembl_ids)
    var_info_dict = matched_var_info.to_dict(orient='index')

    return var_info_dict


def update_feature_names_only(meta_data_preprocessed, meta_data_original):
    """
    Updates the `feature_name` in `meta_data_preprocessed` with `original_gene_symbols`
    from `meta_data_original` for entries where `feature_name` starts with "ENSG".

    Args:
        meta_data_preprocessed (dict): The preprocessed metadata dictionary.
        meta_data_original (dict): The original metadata dictionary.

    Returns:
        dict: Updated `meta_data_preprocessed`.
    """
    for key, entry in meta_data_preprocessed.items():
        # Check if the feature_name starts with ENSG
        if entry.get("feature_name", "").startswith("ENSG"):
            # Fetch the corresponding entry from the original metadata
            original_entry = meta_data_original.get(key, {})
            original_gene_symbol = original_entry.get("original_gene_symbols")

            # Update the feature_name if original_gene_symbols exists
            if original_gene_symbol:
                entry["feature_name"] = original_gene_symbol

    return meta_data_preprocessed


def check_for_genes():
    clin_data = get_cf_clin_data()
    metadata_processed = get_cf_metadata()
    expr_data = get_raw_data('cf_formatted.parquet')

    ensembl_ids =['ENSG00000165471', 'ENSG00000134184', 'ENSG00000204490']

    for eid in ensembl_ids:
        if eid in expr_data.columns:
            print(f"{eid} exists in the DataFrame.")
        else:
            print(f"{eid} does not exist in the DataFrame.")


# sub_keys = set()  # Use a set to avoid duplicates
# for key, inner_dict in meta_data_old.items():
#     sub_keys.update(inner_dict.keys())
#
# print(meta_data_processed)


# # update feature_names manually:
# new_meta_data["ENSG00000185641"]["feature_name"] =
# new_meta_data["ENSG00000230979"]["feature_name"] =
# new_meta_data["ENSG00000233635"]["feature_name"] =



