import scanpy as sc
import numpy as np
import pandas as pd
import os

cf_metadata_step4 = sc.read_h5ad("combined_data4_var.h5ad")
print("Shape beginning: ", cf_metadata_step4.shape)

ensembl_id_to_check = "ENSG00000001626"
if ensembl_id_to_check in cf_metadata_step4.var_names:
    print(f"{ensembl_id_to_check} is present in the combined data.")
else:
    print(f"{ensembl_id_to_check} is NOT present in the combined data.")

# Filter and make a copy to avoid ImplicitModificationWarning
features_to_keep = ~cf_metadata_step4.var["feature_name"].str.startswith("ENSG")
cf_metadata_cleaned = cf_metadata_step4[:, features_to_keep].copy()

print(cf_metadata_cleaned.shape)

obs_df = cf_metadata_cleaned.obs
var_df = cf_metadata_cleaned.var

categories_to_unify = ['NaN', 'nan', 'unknown']

# Loop through each column in the DataFrame
for column in obs_df.columns:
    if isinstance(obs_df[column].dtype, pd.CategoricalDtype):

        # Add 'Unknown' to categories if missing
        if 'Unknown' not in obs_df[column].cat.categories:
            obs_df[column] = obs_df[column].cat.add_categories('Unknown')

        # Replace NaN and other values with 'Unknown'
        obs_df[column] = obs_df[column].fillna('Unknown')
        obs_df[column] = obs_df[column].replace(categories_to_unify, 'Unknown')

        # Remove unused categories
        obs_df[column] = obs_df[column].cat.remove_unused_categories()

for column in var_df.columns:
    if isinstance(var_df[column].dtype, pd.CategoricalDtype):
        # Add 'Unknown' to categories if missing
        if 'Unknown' not in var_df[column].cat.categories:
            var_df[column] = var_df[column].cat.add_categories('Unknown')

        # Replace NaN and other values with 'Unknown'
        var_df[column] = var_df[column].fillna('Unknown')
        var_df[column] = var_df[column].replace(categories_to_unify, 'Unknown')

        # Remove unused categories
        var_df[column] = var_df[column].cat.remove_unused_categories()

    elif pd.api.types.is_numeric_dtype(var_df[column]):
        # Convert to numeric and replace NaN with -1
        var_df[column] = pd.to_numeric(var_df[column], errors='coerce')
        var_df[column] = var_df[column].fillna(-1)

print(var_df['feature_length'].head())
print(var_df['feature_length'].unique())

# Validate indices
assert obs_df.index.equals(cf_metadata_cleaned.obs_names), "Index mismatch in .obs"
assert var_df.index.equals(cf_metadata_cleaned.var_names), "Index mismatch in .var"

# Assign updated DataFrames back to AnnData object
cf_metadata_cleaned.obs = obs_df
cf_metadata_cleaned.var = var_df

# Save as parquet files
output_dir = "data/"
expression_df = pd.DataFrame(cf_metadata_cleaned.X.toarray(), index=cf_metadata_cleaned.obs_names, columns=cf_metadata_cleaned.var_names)

expression_df.to_parquet(os.path.join(output_dir, "combined_data_expression.parquet"))
cf_metadata_cleaned.obs.to_parquet(os.path.join(output_dir, "combined_data_obs.parquet"))
cf_metadata_cleaned.var.to_parquet(os.path.join(output_dir, "combined_data_var.parquet"))

print("Data saved as Parquet files in the directory:", output_dir)
