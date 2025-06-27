import torch
import torch.nn as nn
from src.encoder import ClimateEncoder

class HarvestModel(nn.Module):
    def __init__(self,
                 climate_encoder: ClimateEncoder,
                 output_dim=40):
        super().__init__()
        
        self.output_dim = output_dim
        
        self.encoder = climate_encoder


        self.kilo_gru = nn.GRU(
            input_size=3,
            hidden_size=output_dim,
            batch_first=True
        )

        self.kilo_output = nn.Sequential(
            nn.Linear(self.encoder.combined_dim+output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, features, encoded_features, climate_data, kilo_gru_input):
        """
        features: (batch_size, 5)
        climate_data: (batch_size, 100, 10)
        kilo_gru_input: (batch_size, [5:20], 3)
        """
        encoded = self.encoder(features, encoded_features, climate_data)
        batch_size = climate_data.size(0)
        
        # Ensure h0 is on the same device as the input
        h0 = torch.zeros(1, batch_size, self.kilo_gru.hidden_size, device=kilo_gru_input.device)
        out, _ = self.kilo_gru(kilo_gru_input, h0)
        kilos = out[:, -1, :]
    
        kilo_output = self.kilo_output(torch.cat((encoded,kilos),dim=1))
        return kilo_output

class HarvestScheduleModel(nn.Module):
    def __init__(self,
                 climate_encoder: ClimateEncoder,
                 output_dim=5):
        super().__init__()
        
        self.output_dim = output_dim
        self.encoder = climate_encoder


        self.kilo_gru = nn.GRU(
            input_size=3,
            hidden_size=output_dim,
            batch_first=True
        )

        self.schedule_output = nn.Sequential(
            nn.Linear(self.encoder.combined_dim+output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, features, encoded_features, climate_data, kilo_gru_input):
        """
        features: (batch_size, 5)
        climate_data: (batch_size, 100, 10)
        kilo_gru_input: (batch_size, [5:20], 3)
        """
        encoded = self.encoder(features, encoded_features, climate_data)
        batch_size = climate_data.size(0)
        
        # Ensure h0 is on the same device as the input
        h0 = torch.zeros(1, batch_size, self.kilo_gru.hidden_size, device=kilo_gru_input.device)
        out, _ = self.kilo_gru(kilo_gru_input, h0)
        kilos = out[:, -1, :]
    
        schedule = self.schedule_output(torch.cat((encoded,kilos),dim=1))
        return schedule