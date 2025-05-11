import torch
import torch.nn as nn
from encoder import ClimateEncoder

class HarvestModel(nn.Module):
    def __init__(self,
                 input_dim=5,
                 hidden_dim=64,
                 n_ranches=13,
                 n_classes=2,
                 n_types=14,
                 n_varieties=59,
                 climate_input_dim=3,
                 climate_hidden_dim=32,
                 output_dim=20):
        super().__init__()

        self.encoder = ClimateEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_ranches=n_ranches,
            n_classes=n_classes,
            n_types=n_types,
            n_varieties=n_varieties,
            climate_input_dim=climate_input_dim,
            climate_hidden_dim=climate_hidden_dim
        )

        self.final_kilos = nn.Sequential(
            nn.Linear(self.encoder.combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, features, ranch_id, class_id, type_id, variety_id, climate_data):
        """
        features: (batch_size, 5)
        climate_data: (batch_size, 100, 3)
        """
        encoded = self.encoder(features, ranch_id, class_id, type_id, variety_id, climate_data)

        return self.final_kilos(encoded)
