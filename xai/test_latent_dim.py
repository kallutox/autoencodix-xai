from helper_functions import *
from xai_methods import captum_importance_values
from scipy.stats import ttest_ind


def get_best_dimension_ttest(run_id):
    latent_df = get_latent_space(run_id)
    clin_df = get_cf_clin_data(run_id)
    merged_df = latent_df.merge(clin_df[['sex']], left_index=True, right_index=True)

    p_values = {}

    for column in merged_df.columns:
        if "L_COMBINED-RNA__varix_INPUT_" in column:
            latent_number = int(column.split("_")[-1])  # Convert latent dimension to integer
            male_values = merged_df[merged_df['sex'] == 'male'][column]
            female_values = merged_df[merged_df['sex'] == 'female'][column]

            t_stat, p_val = ttest_ind(male_values, female_values, equal_var=False)
            p_values[latent_number] = p_val  # Use the integer latent number as the key

            print(f"Dimension {latent_number} - Males: {len(male_values)}, Females: {len(female_values)}")
            print(f"p-value: {p_val}\n")

    best_dimension = min(p_values, key=p_values.get)
    return best_dimension


def get_best_dimension_means(run_id):
    latent_df = get_latent_space(run_id)
    clin_df = get_cf_clin_data(run_id)
    merged_df = latent_df.merge(clin_df[['sex']], left_index=True, right_index=True)

    mean_differences = {}

    for column in merged_df.columns:
        if "L_COMBINED-RNA__varix_INPUT_" in column:
            latent_number = column.split("_")[-1]

            male_values = merged_df[merged_df['sex'] == 'male'][column]
            female_values = merged_df[merged_df['sex'] == 'female'][column]

            male_mean = male_values.mean()
            female_mean = female_values.mean()

            mean_diff = abs(male_mean - female_mean)

            mean_differences[int(latent_number)] = mean_diff

            print(f"Dimension {latent_number} - Male Mean: {male_mean}, Female Mean: {female_mean}")
            print(f"Absolute Difference in Means: {mean_diff}\n")

    best_dimension = max(mean_differences, key=mean_differences.get)
    print(f"The best dimension based on mean difference is {best_dimension} with a difference of {mean_differences[best_dimension]}")
    return int(best_dimension)



run_id = 'synth_data_10features_09signal'
#print(get_best_dimension_ttest(run_id))
get_best_dimension_means(run_id)