import torch
import torch.nn as nn
from src.encoder import ClimateEncoder

class HarvestModel(nn.Module):
    def __init__(self,
                 input_dim=5,
                 hidden_dim=64,
                 embedding_dim=4,
                 n_ranches=13,
                 n_classes=2,
                 n_types=14,
                 n_varieties=59,
                 climate_input_dim=3,
                 climate_hidden_dim=32,
                 harvest_clamp_hidden_dim=16,
                 output_dim=20):
        super().__init__()
        
        self.output_dim = output_dim
        self.encoder = ClimateEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            n_ranches=n_ranches,
            n_classes=n_classes,
            n_types=n_types,
            n_varieties=n_varieties,
            climate_input_dim=climate_input_dim,
            climate_hidden_dim=climate_hidden_dim
        )

        self.harvest_clamp = nn.Sequential(
            nn.Linear(self.encoder.combined_dim,harvest_clamp_hidden_dim),
            nn.ReLU(),
            nn.Linear(harvest_clamp_hidden_dim,2)
        )

        self.kilo_gru = nn.GRU(
            input_size=3,
            hidden_size=output_dim,
            batch_first=True
        )

        self.kilo_output = nn.Sequential(
            nn.Linear(self.encoder.combined_dim+output_dim+2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, features, ranch_id, class_id, type_id, variety_id, climate_data, kilo_gru_input):
        """
        features: (batch_size, 5)z
        climate_data: (batch_size, 100, 3)
        kilo_gru_input: (batch_size, [5:20], 3)
        """
        encoded = self.encoder(features, ranch_id, class_id, type_id, variety_id, climate_data)
        batch_size = climate_data.size(0)
        
        h0 = torch.zeros(1, batch_size, self.kilo_gru.hidden_size)
        out, _ = self.kilo_gru(kilo_gru_input, h0)
        kilos = out[:, -1, :]
        clamp = torch.clamp(self.harvest_clamp(encoded),min=0,max=self.output_dim-1)
        kilo_output = self.kilo_output(torch.cat((encoded,kilos,clamp),dim=1))
        return kilo_output, clamp



