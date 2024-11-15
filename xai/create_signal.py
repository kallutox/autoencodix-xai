import numpy as np
from helper_functions import *

# rna_data = get_raw_data(file_name="cf_formatted.parquet")
rna_data = get_processed_data("synth_data_10features_09signal")
clin_data = get_raw_data(file_name="cf_clinical_data_formatted.parquet")

# sample list: list of female patients
sample_list_female = clin_data[clin_data['sex'] == 'female'].index
sample_list_female = sample_list_female.intersection(rna_data.index).tolist()
with open("../data/raw/sample_list_female.txt", "w") as f:
  f.write("\n".join(sample_list_female))


# feature list: random list of n features signal is applied to
feature_n = [1, 5, 10, 50, 100]
rna_data.columns = rna_data.columns.str.replace(r"^RNA_", "", regex=True)

for i in feature_n:
    random_features = np.random.choice(rna_data.columns, i, replace=False).tolist()
    with open(f"../data/raw/feature_list_{i}.txt", "w") as f:
       f.write("\n".join(random_features))


# # testing if it works
# feature_synth = pd.Index(random_features).intersection(rna_data.columns).to_list()
#
# with open("../data/raw/sample_list_female.txt", "r") as f:
#     sample_synth = f.read().splitlines()
#
# intersected_data = rna_data.loc[sample_synth, feature_synth]
# print(intersected_data)

