import pandas as pd
import torch
import sys
import os
import pickle
from collections import Counter
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from captum.attr import GradientShap, DeepLiftShap, LRP
from helper_functions import *

from xai_methods import captum_importance_values
from src.models.models import Vanillix, Varix

# captum attribution methods
run_id = 'synth_data_10features_09signal_base1'
data_types = 'RNA'
dataset = 'cf'
model_type = 'varix'
xai_method = 'integrated_gradients'
latent_dim = get_best_dimension_cf(run_id)
# add parameter to change reference?

attribution_values = captum_importance_values(run_id=run_id, data_types=data_types, model_type=model_type,
    dimension=latent_dim, latent_space_explain=True, xai_method=xai_method, visualize=True, return_delta=False, random_seed=None)





# attribution_dict = attribution_per_feature(
#                 attribution_values,
#                 get_interim_data(run_id, model_type)
#             )
# with open(f"../data/raw/feature_list_10_base1.txt", "r") as f:
#     feature_list = f.read().splitlines()
#
# feature_list = set(feature_list)
#
# zero_perc_dict = calculate_zero_score_percentage(attr_dict_all)
# synth_feature_zero_perc, rest_zero_perc = separate_zero_score_percentage(zero_perc_dict, feature_list)
#
# specified_avg = np.mean(list(synth_feature_zero_perc.values()))
# remaining_avg = np.mean(list(rest_zero_perc.values()))
#
# print("Specified Features Average:", specified_avg)
# print("Remaining Features Average:", remaining_avg)


# top_features = get_top_features(attribution_dict, top_n=10)
# #print(top_features)
#
# with open(f"../data/raw/feature_list_10_base1.txt", "r") as f:
#     feature_list = f.read().splitlines()
#
# feature_list = [feature.replace("RNA_", "") for feature in feature_list]
# positions = [i for i, feature in enumerate(top_features) if feature in feature_list]
#
# print('Positions of top features that were modified:', positions)
# print('Number of top features that were modified:', len(positions))

# gene_metadata = get_cf_metadata(top_features)
# print(gene_metadata)
#
# gene_clin_data = get_cf_clin_data(top_features)
# print(gene_clin_data)



# if dataset == 'cf':
#     gene_metadata = get_cf_metadata(top_f)
#     feature_count = 1
#     for feature in top_f:
#         print("Feature ", feature_count, ": ", feature, " - ", gene_metadata[feature])
#         feature_count += 1
# else:
#     print(top_f)

# df = pd.read_csv('/Users/kallutox/Documents/projects/autoencoder/data/raw/random_features_5.txt', delimiter='\t')
#
# # loop over dimensions
# for i in range(0, 7):
#     attribution_values, delta_values = captum_importance_values(run_id=run_id, data_types=data_types, model_type='varix',
#                                                             dimension=i, latent_space_explain=True, xai_method='deepshap',
#                                                             visualize=False, return_delta=True)
#     top_f = get_top_features(attribution_values, get_processed_data(run_id), top_n=10)
#
#     top_f = [col.replace('RNA_', '') for col in top_f]
#     column_presence = [col in df.columns for col in top_f]
#     counter = 0
#     for i in column_presence:
#         if i:
#             counter += 1
#     print('Number of top features that was modified: ', str(counter))



# state_dict = shapHelper.get_state_dict(run_id=run_id, data_types=data_types, model_type='varix')
# input_data_rna = shapHelper.get_processed_data(run_id, data_type='RNA')
# input_data_mut = shapHelper.get_processed_data(run_id, data_type='MUT')
# input_data_meth = shapHelper.get_processed_data(run_id, data_type='METH')
#
# print("Input data rna shape: ", input_data_rna.shape)
# print("Input data mut shape: ", input_data_mut.shape)
# print("Input data meth shape: ", input_data_meth.shape)
#
#
# print(state_dict.keys())
# print("Weights shape: ", state_dict['encoder.0.weight'].shape)
# print("Get_input_dim: ", shapHelper.get_input_dim(state_dict))
