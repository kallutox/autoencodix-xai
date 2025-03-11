import scanpy as sc
import numpy as np
import pandas as pd
import os

cf_control_data = sc.read_h5ad("data/cf_control_data.h5ad")

# combine the CF and downsampled healthy control data
#combined_data = cf_data.concatenate(normal_data_subset)

# filter out genes present in less than 20% of genes
sc.pp.filter_genes(cf_control_data, min_cells=int(cf_control_data.shape[0] * 0.2), inplace=True)

# keep only k of most variable genes
k_filter = 4000
sc.pp.highly_variable_genes(cf_control_data ,n_top_genes=k_filter, subset=True, inplace=True)

cf_control_data.write("data/cf_control_data_filtered_4000.h5ad")

# define the path to save Parquet files
output_dir = "data/parquet_files_new"
os.makedirs(output_dir, exist_ok=True)

# dave the X matrix (expression data)
df_X = pd.DataFrame.sparse.from_spmatrix(
    cf_control_data.X,
    index=cf_control_data.obs.index,
    columns=cf_control_data.var.index
)

df_X.sparse.to_dense().to_parquet(os.path.join(output_dir, "cf_control_data_X.parquet"))

# save the obs DataFrame (observations metadata)
cf_control_data.obs.to_parquet(os.path.join(output_dir, "cf_control_data_obs.parquet"))

# save the var DataFrame (variables metadata)
cf_control_data.var.to_parquet(os.path.join(output_dir, "cf_control_data_var.parquet"))

categories_to_unify = ['NaN', 'nan', 'unknown']

# Loop through each column in the DataFrame
for column in obs_df.columns:
    if pd.api.types.is_categorical_dtype(obs_df[column]):
        
        # Check if 'Unknown' is already in the categories, if not, add it
        if 'Unknown' not in obs_df[column].cat.categories:
            obs_df[column] = obs_df[column].cat.add_categories('Unknown')
        
        # Replace actual NaN values with 'Unknown'
        obs_df[column] = obs_df[column].fillna('Unknown')

        if any(cat in obs_df[column].cat.categories for cat in categories_to_unify):
            print(f"Before replacement in column {column}: {obs_df[column].cat.categories}")
            
            # Replace each of the specified categories with 'Unknown'
            obs_df[column] = obs_df[column].replace(categories_to_unify, 'Unknown')
            
            # Remove unused categories
            obs_df[column] = obs_df[column].cat.remove_unused_categories()
            
            print(f"After replacement in column {column}: {obs_df[column].cat.categories}")
