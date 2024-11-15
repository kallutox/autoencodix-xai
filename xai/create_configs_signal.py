from itertools import product
import yaml
from pathlib import Path
import numpy as np

beta = 0.01
latent_dim = 8
k_filter = 2000
lr = 1e-3
batch_size = 256

# define range of signal strength
#signal_strength = np.linspace(0.1,1,10)
#feature_n = [1, 3, 5, 10, 50, 100]
signal_strength_values = [0.9]
feature_n = [10]

config_save_root = "../"
Path(config_save_root).mkdir(parents=False, exist_ok=True)


for signal_strength, n in product(signal_strength_values, feature_n):
    cfg = dict()

    # Add your fixed values to the config
    cfg['FIX_RANDOMNESS'] = 'all'
    cfg['GLOBAL_SEED'] = 42
    cfg['EPOCHS'] = 300
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

    cfg['BETA'] = beta
    cfg['LATENT_DIM_FIXED'] = latent_dim
    cfg['K_FILTER'] = k_filter
    cfg['LR_FIXED'] = lr
    cfg['BATCH_SIZE'] = batch_size

    # synthetic signal values
    cfg['APPLY_SIGNAL'] = True
    cfg['SIGNAL_STRENGTH'] = signal_strength
    cfg['SAMPLE_SIGNAL'] = "data/raw/sample_list_female.txt"
    cfg['FEATURE_SIGNAL'] = f"data/raw/feature_list_{n}.txt"

    # convert float values to strings and remove dots if they exist
    signal_strength_str = str(signal_strength).replace('.', '') if '.' in str(signal_strength) else str(signal_strength)

    run_id = f"synth_data_{n}features_{signal_strength_str}signal_config.yaml"

    with open(config_save_root + run_id, 'w') as file:
        yaml.dump(cfg, file)

print("Config files generated!")
