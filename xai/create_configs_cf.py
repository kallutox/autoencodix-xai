from itertools import product
import yaml
from pathlib import Path

# Define the ranges for each parameter
# beta_values = [0.01, 0.1, 1, 10]
# latent_dim_values = [8, 16, 32]
# k_filter_values = [1000, 2000, 3000]
# lr_values = [1e-4, 1e-3, 1e-2]
# batch_size_values = [128, 256, 512]

beta_values = [0.01, 0.1, 1, 3]
latent_dim_values = [8, 16]
k_filter_values = [4000, 6000]
lr_values = [1e-3]
batch_size_values = [256]

# Create the config folder if it doesn't exist
#config_save_root = "../cf_configs/"
config_save_root = "../"
Path(config_save_root).mkdir(parents=False, exist_ok=True)

# Iterate through all combinations of the parameters using product
for beta, latent_dim, k_filter, lr, batch_size in product(beta_values, latent_dim_values, k_filter_values, lr_values, batch_size_values):
    cfg = dict()

    # Add your fixed values to the config
    cfg['FIX_RANDOMNESS'] = 'all'
    cfg['GLOBAL_SEED'] = 42
    cfg['EPOCHS'] = 500
    cfg['RECONSTR_LOSS'] = "MSE"
    cfg['VAE_LOSS'] = "KL"
    cfg['PREDICT_SPLIT'] = "all"
    cfg['MODEL_TYPE'] = "varix"
    cfg['TRAIN_TYPE'] = "train"
    cfg['SPLIT'] = [0.6, 0.2, 0.2]

    # Additional configurations from the provided YAML file
    cfg['DATA_TYPE'] = {
        'RNA': {'FILE_RAW': 'cf_formatted.parquet', 'TYPE': 'NUMERIC', 'SCALING': 'Standard', 'FILTERING': 'Var'},
        'ANNO': {'FILE_RAW': 'cf_clinical_data_formatted.parquet', 'TYPE': 'ANNOTATION'}
    }
    cfg['DIM_RED_METH'] = 'UMAP'
    cfg['CLUSTER_ALG'] = 'HDBScan'
    cfg['CLUSTER_N'] = 32
    cfg['MIN_CLUSTER_N'] = 20
    cfg['CLINIC_PARAM'] = [
        'disease', 'lung_condition', 'age_range', 'cell_type', 'development_stage',
        'sex', 'smoking_status', 'assay', 'self_reported_ethnicity', 'development_stage'
    ]
    cfg['ML_TYPE'] = 'Auto-detect'
    cfg['ML_ALG'] = ['Linear', 'RF']
    cfg['ML_SPLIT'] = 'use-split'
    cfg['ML_TASKS'] = ['Latent', 'PCA']

    cfg['PLOT_CLUSTLATENT'] = True
    cfg['PLOT_WEIGHTS'] = True

    # Assign values from the loop
    cfg['BETA'] = beta
    cfg['LATENT_DIM_FIXED'] = latent_dim
    cfg['K_FILTER'] = k_filter
    cfg['LR_FIXED'] = lr
    cfg['BATCH_SIZE'] = batch_size

    # Convert float values to strings and remove dots if they exist
    beta_str = str(beta).replace('.', '') if '.' in str(beta) else str(beta)
    lr_str = str(lr).replace('.', '') if '.' in str(lr) else str(lr)

    # Create a unique filename based on the parameter values
    run_id = f"cf_beta{beta_str}_dim{latent_dim}_k{k_filter}_lr{lr_str}_bsize{batch_size}_config.yaml"

    # Save the config to a YAML file
    with open(config_save_root + run_id, 'w') as file:
        yaml.dump(cfg, file)

print("Config files generated!")
