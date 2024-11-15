import pandas as pd

data_cf = pd.read_parquet("../data/raw/cf_clinical_data_formatted.parquet")
data_tcga = pd.read_parquet("../data/raw/data_clinical_formatted.parquet")

sex_counts_tcga = data_tcga['SEX'].value_counts()
sex_counts_cf = data_cf['sex'].value_counts()
cf_perc = data_cf['sex'].value_counts(normalize=True) * 100
tcga_perc = data_tcga['SEX'].value_counts(normalize=True) * 100

print(sex_counts_tcga)
print(tcga_perc)
print()
print(sex_counts_cf)
print(cf_perc)
