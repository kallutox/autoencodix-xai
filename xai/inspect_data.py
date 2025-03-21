from helper_functions import *
import ast

data_cf = pd.read_parquet("../data/raw/cf_clinical_data_formatted.parquet")
data_tcga = pd.read_parquet("../data/raw/data_clinical_formatted.parquet")
expr_tcga = get_raw_data("data_mrna_seq_v2_rsem_formatted.parquet")
mut_tcga = pd.read_parquet("../data/raw/data_combi_MUT_CNA_formatted.parquet")
met_tcga = pd.read_parquet("../data/raw/data_methylation_per_gene_formatted.parquet")
var_path = os.path.join(
    os.path.abspath(os.path.join(current_directory, "..")),
    "data",
    "raw",
    "data_mutations.txt",
)
tcga_var = pd.read_csv(var_path, sep='\t')

sex_counts_tcga = data_tcga['SEX'].value_counts()
sex_counts_cf = data_cf['sex'].value_counts()
cf_perc = data_cf['sex'].value_counts(normalize=True) * 100
tcga_perc = data_tcga['SEX'].value_counts(normalize=True) * 100
tcga_interim = get_interim_data('tcga_001_9')

# Align column names if necessary
# mut_tcga.columns = expr_tcga.columns
# met_tcga.columns = expr_tcga.columns

combined_df = pd.concat([expr_tcga, mut_tcga, met_tcga], axis=1, join='inner')
merged_df = tcga_interim.merge(data_tcga['CANCER_TYPE_ACRONYM'], left_index=True, right_index=True)

print(tcga_interim.shape)

merged_df = combined_df.merge(data_tcga['CANCER_TYPE_ACRONYM'], left_index=True, right_index=True)
#merged_df = tcga_interim.merge(data_tcga['CANCER_TYPE_ACRONYM'], left_index=True, right_index=True)
#print(merged_df['CANCER_TYPE_ACRONYM'].value_counts())
#print(expr_tcga.shape)

# merged_df = expr_tcga.merge(data_tcga['CANCER_TYPE_ACRONYM'], left_index=True, right_index=True)
# print(len(data_tcga['CANCER_TYPE_ACRONYM'].unique()))


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
    palette = {
        "Unknown": "#43a2b5",
        "Female": "#4354b5",
        "Male": "#43b582"
    }

    # Create the ridge plot
    g = sns.FacetGrid(
        data,
        row="latent_dim",
        hue="sex",
        aspect=3,
        height=0.8,  # Reduce height for compression
        palette=palette,
        sharex=False,
        sharey=False
    )
    g.map(sns.kdeplot, "latent_intensity", alpha=0.8, fill=True, warn_singular=False)  # Increase opacity

    g.set(xlim=(-3, 12))
    for ax in g.axes.flat:
        ax.set_title("")
    g.set_titles("{row_name}", loc="left", size=10)
    g.set_axis_labels("Latent Intensity", "")
    g.set(yticks=[], ylabel="")
    g.despine(left=True)

    # Adjust spacing and position of the legend
    g.fig.subplots_adjust(hspace=0.4)  # Reduce spacing between rows
    g.add_legend(title="Sex", loc="upper right", bbox_to_anchor=(1, 0.7))

    plt.tight_layout()
    plt.show()


def get_cf_metadata_old(ensembl_ids=None):
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


def clean_data():
    file_path = "cf_reports/all_features_001_deepliftshap.txt"

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Process each line to extract the feature name and score
    data = []
    for line in lines:
        try:
            # Split metadata and score
            metadata_part, score_part = line.rsplit(",", 1)
            score = float(score_part.strip())  # Convert score to float

            # Convert metadata from string to dictionary safely
            metadata_dict = ast.literal_eval(metadata_part.strip())

            # Extract feature name
            feature_name = metadata_dict.get("feature_name", "Unknown")

            # Append to data list
            data.append([feature_name, score])
        except Exception as e:
            print(f"Skipping line due to error: {e}")

    # Create DataFrame
    df_cleaned = pd.DataFrame(data, columns=["Gene Name", "Score"])

    # Save to CSV
    output_csv = "cf_reports/cleaned_features_001_deepliftshap.txt"
    df_cleaned.to_csv(output_csv, index=False)

    print(df_cleaned.head())

