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

from xai_methods import captum_importance_values, shap_importance_values
from src.models.models import Vanillix, Varix

# captum attribution methods
run_id = 'CF_beta_decr'
data_types = 'RNA'

attribution_values, delta_values = captum_importance_values(run_id=run_id, data_types=data_types, model_type='varix',
    dimension=0, latent_space_explain=True, xai_method='deepshap', visualize=True, return_delta=True)


top_f = get_top_features(attribution_values, get_processed_data(run_id), top_n=10)
#print('Top features: ', top_f)
gene_metadata = get_cf_metadata(top_f)

feature_count = 1
for feature in top_f:
    print("Feature ", feature_count, ": ", feature, " - ", gene_metadata[feature])
    feature_count += 1


## overlap: top features - synthetic signal
# df = pd.read_csv('/Users/kallutox/Documents/projects/autoencoder/data/raw/random_features_1.txt', delimiter='\t')
# top_f = [col.replace('RNA_', '') for col in top_f]
# column_presence = [col in df.columns for col in top_f]
# counter = 0
# for i in column_presence:
#     if i:
#         counter += 1
# print('Number of top features that was modified: ', str(counter))


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
