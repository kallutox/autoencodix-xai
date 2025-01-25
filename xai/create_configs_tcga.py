from itertools import product
import yaml
from pathlib import Path

# Define the ranges for each parameter
beta_values = [1]
latent_dim_values = [32]
k_filter_values = [2000]
lr_values = [0.001]
batch_size_values = [256]

# Create the config folder if it doesn't exist
config_save_root = "../tcga_configs/"
Path(config_save_root).mkdir(parents=False, exist_ok=True)

# Iterate through all combinations of the parameters using product
for beta, latent_dim, k_filter, lr, batch_size in product(beta_values, latent_dim_values, k_filter_values, lr_values, batch_size_values):
    cfg = dict()

    # Add your fixed values to the config
    cfg['EPOCHS'] = 300
    cfg['RECONSTR_LOSS'] = "MSE"
    cfg['VAE_LOSS'] = "KL"
    cfg['PREDICT_SPLIT'] = "all"
    cfg['MODEL_TYPE'] = "varix"
    cfg['SPLIT'] = [0.6, 0.2, 0.2]
    cfg['LOGLEVEL'] = 'INFO'
    cfg['START_FROM_LAST_CHECKPOINT'] = False
    cfg['USE_LABEL'] = 'ANNO'
    cfg['TRANSLATE'] = ''
    cfg['APPLY_SIGNAL'] = False
    cfg['PLOT_LATENTDIST'] = True
    cfg['PLOT_WEIGHTS'] = False
    cfg['DELIM'] = '\t'
    cfg['KEEP_NOT_ONT'] = True
    cfg['NON_ONT_LAYER'] = 1
    cfg['CV'] = 5

    # Additional configurations from the provided YAML file
    cfg['DATA_TYPE'] = {
        'ANNO': {'FILE_LABEL': 'clinical_data.parquet', 'FILE_RAW': 'data_clinical_formatted.parquet', 'TYPE': 'ANNOTATION'},
        'METH': {'FILE_RAW': 'data_methylation_per_gene_formatted.parquet', 'FILTERING': 'Var', 'SCALING': 'Standard', 'TYPE': 'NUMERIC'},
        'MUT': {'FILE_RAW': 'data_combi_MUT_CNA_formatted.parquet', 'FILTERING': 'Var', 'SCALING': 'Standard', 'TYPE': 'NUMERIC'},
        'RNA': {'FILE_RAW': 'data_mrna_seq_v2_rsem_formatted.parquet', 'FILTERING': 'Var', 'SCALING': 'Standard', 'TYPE': 'NUMERIC'}
    }
    cfg['CLINIC_PARAM'] = [
        'CANCER_TYPE_ACRONYM', 'TMB_NONSYNONYMOUS', 'AGE', 'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT',
        'OS_STATUS', 'OS_MONTHS', 'DFS_STATUS', 'PFS_STATUS', 'MSI_SCORE_MANTIS', 'ANEUPLOIDY_SCORE'
    ]
    cfg['DIM_RED_METH'] = 'UMAP'
    cfg['DROP_P'] = 0.1
    cfg['FILE_ONT_LVL1'] = 'full_ont_lvl1_reactome.txt'
    cfg['FILE_ONT_LVL2'] = 'full_ont_lvl2_reactome.txt'
    cfg['ML_TYPE'] = 'Auto-detect'
    cfg['ML_ALG'] = ['Linear', 'RF', 'SVM']
    cfg['ML_SPLIT'] = 'use-split'
    cfg['ML_TASKS'] = ['Latent', 'UMAP', 'PCA', 'RandomFeature']
    cfg['TRAIN_TYPE'] = 'train'
    cfg['FIX_RANDOMNESS'] = 'random'

    # Assign values from the loop
    cfg['BETA'] = beta
    cfg['LATENT_DIM_FIXED'] = latent_dim
    cfg['K_FILTER'] = k_filter
    cfg['LR_FIXED'] = lr
    cfg['BATCH_SIZE'] = batch_size

    formatted_beta_value = str(beta).replace('.', '')
    lr_val = "0001"

    for i in range(1,11):
        run_id = f"tcga_1_{i}_config.yaml"

        # Save the config to a YAML file
        with open(config_save_root + run_id, 'w') as file:
            yaml.dump(cfg, file)

print("Config files generated!")
