import src.training as training
import src.load as load
import numpy as np
import pandas as pd


train_dataset, test_dataset, mappings, reverse_mappings, train_meta, test_meta  = load.separate_year('../data/processed/')

hyperparameters = {
    'lr': [1e-4, 1e-3, 1e-2, 1e-5],
    'weight_decay': [1e-4, 1e-3, 1e-2, 1e-5],
    'dropout': [0.1, 0.2, 0.3, 0.4],
    'step_size': [10, 20, 30, 40],
    'gamma': [0.1, 0.2, 0.3, 0.4],
    'batch_size': [32,64, 128, 256],
    'patience': [10, 20, 30]
}