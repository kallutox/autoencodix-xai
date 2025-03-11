import scanpy as sc
import numpy as np
import pandas as pd
import os

adata = sc.read_h5ad("data/normal_data_full.h5ad")

# 1 - extract only lung tissue
adata_lung = adata[adata.obs["tissue"] == "lung"].copy()
print("Shape after tissue filter: ", adata_lung.shape)
	#Shape after tissue filter:  (558404, 56239)
adata_lung.write("data/control_data_new/normal_data1_tissue.h5ad")

# 2 - match normal data size to cf size
np.random.seed(42)
cf_data_size = 17590 #checked in jupyter
sampled_indices = np.random.choice(adata_lung.obs_names, size=cf_data_size, replace=False)
adata_red = adata_lung[sampled_indices, :].copy()
print("Shape of reduced data: ", adata_red.shape)
	#Shape of reduced data: (17590, 56239)
adata_red.write("data/data_prep_new/normal_data2_reduced.h5ad")


# alt. 2 - downsample from cells expressing cftr gene only
# → cftr_prep.py, data: data/data_030924/with_cftr


# alt. 2a - downsample while matching cell_type distribution 
#to cf data → cell_type_distr.py, 
#data/data_090924/normal_data2a_distr.h5ad

# 3 - combine data
control_data = sc.read_h5ad("data/data_prep_new/normal_data2_reduced.h5ad")
cf_data = sc.read_h5ad("data/cf_data_full.h5ad")
combined_data = cf_data.concatenate(control_data)
print("Shape of combined data: ", combined_data.shape)
	#Shape of combined data:  (35180, 56239)
combined_data.write("data/data_prep_new/combined_data3.h5ad")

# 4 - filter out genes present in less than 3% of cells
sc.pp.filter_genes(combined_data, min_cells=int(combined_data.shape[0] * 0.03), inplace=True)
print("Shape after expression filter: ", combined_data.shape)
	#Shape after expression filter:  (35180, 11324)
combined_data.write("data/data_prep_new/combined_data4_expr.h5ad")


# 5 - keep only k of most variable genes
k_filter = 8000
sc.pp.highly_variable_genes(combined_data, n_top_genes=k_filter, subset=True, inplace=True)
print("Shape after var. filter: ", combined_data.shape)
	#Shape after var. filter:  (35180, 8000)
combined_data.write("data/data_prep_new/combined_data4_var.h5ad")

# check for CFTR ID
ensembl_id_to_check = "ENSG00000001626"
if ensembl_id_to_check in combined_data.var_names:
    print(f"{ensembl_id_to_check} is present in the combined data.")
else:
    print(f"{ensembl_id_to_check} is NOT present in the combined data.")
	#ENSG00000001626 is present in the combined data.

# 6 - as as parquet files
output_dir = "data/data_prep_new"
expression_df = pd.DataFrame(combined_data.X.toarray(), index=combined_data.obs_names, columns=combined_data.var_nam$

expression_df.to_parquet(os.path.join(output_dir, "combined_data_expression.parquet"))

