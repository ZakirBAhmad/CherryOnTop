import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from src.encoder import ClimateEncoder

class KiloModel(nn.Module):
    def __init__(self,
                 climate_encoder: ClimateEncoder,
                 output_dim=40,
                 hidden_size=32,
                 batch_size=32):
        super().__init__()
        
        self.output_dim = output_dim
        
        self.encoder = climate_encoder

        self.kilo_gru = nn.GRU(
            input_size=3,
            num_layers=2,
            dropout=0.2,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.kilo_output = nn.Sequential(
            weight_norm(nn.Linear(self.encoder.combined_dim+hidden_size, batch_size)),
            nn.ReLU(),
            weight_norm(nn.Linear(batch_size, output_dim))
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
        h0 = torch.zeros(2, batch_size, self.kilo_gru.hidden_size, device=kilo_gru_input.device)
        out, _ = self.kilo_gru(kilo_gru_input, h0)
        kilos = out[:, -1, :]
    
        kilo_output = self.kilo_output(torch.cat((encoded,kilos),dim=1))
        return kilo_output

class ScheduleModel(nn.Module):
    def __init__(self,
                 climate_encoder: ClimateEncoder,
                 output_dim=5,
                 batch_size=32,
                 hidden_size=32):
        super().__init__()
        
        self.output_dim = output_dim
        self.encoder = climate_encoder

        self.kilo_gru = nn.GRU(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        self.schedule_output = nn.Sequential(
            weight_norm(nn.Linear(self.encoder.combined_dim+hidden_size, batch_size)),
            nn.ReLU(),
            weight_norm(nn.Linear(batch_size, output_dim))
        )

    def forward(self, features, encoded_features, climate_data, kilo_gru_input):
        """
        features: (batch_size, 5)
        climate_data: (batch_size, 100, 10)
        kilo_gru_input: (batch_size, [5:20], 3)
        kilo_model_output: (batch_size, 40)
        """
        encoded = self.encoder(features, encoded_features, climate_data)
        batch_size = climate_data.size(0)
        
        # Ensure h0 is on the same device as the input
        h0 = torch.zeros(2, batch_size, self.kilo_gru.hidden_size, device=kilo_gru_input.device)
        out, _ = self.kilo_gru(kilo_gru_input, h0)
        kilos = out[:, -1, :]
    
        schedule = self.schedule_output(torch.cat((encoded,kilos),dim=1))
        return schedule
    
class FinalModel(nn.Module):
    def __init__(self,
                 kilo_output_dim=40,
                 schedule_output_dim=5,
                 batch_size=32):
        super().__init__()
        
        self.final_output = nn.Sequential(
            weight_norm(nn.Linear(kilo_output_dim+schedule_output_dim+1, batch_size)),
            nn.ReLU(),
            weight_norm(nn.Linear(batch_size, kilo_output_dim))
        )
        
    def forward(self, week_number, kilo_model_output, schedule_model_output):
        return self.final_output(torch.cat((week_number, kilo_model_output, schedule_model_output), dim=1))