import torch
import torch.nn as nn
from src.encoder import ClimateEncoder
from src.stats_predictor import StatsPredictor

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
                 output_dim=20):
        super().__init__()
        

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

        self.stats_predictor = StatsPredictor(
            encoder_dim=self.encoder.combined_dim
        )
        self.t = torch.arange(output_dim, dtype=torch.float)
        
        self.final_kilos = nn.Sequential(
            nn.Linear(self.encoder.combined_dim + output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, features, ranch_id, class_id, type_id, variety_id, climate_data):
        """
        features: (batch_size, 5)
        climate_data: (batch_size, 100, 3)
        """
        encoded = self.encoder(features, ranch_id, class_id, type_id, variety_id, climate_data)

        o2 = self.stats_predictor(encoded)
        pmf = torch.stack([self.logistic_pmf(o) for o in o2])
        together = torch.cat((encoded,pmf),dim=1)
        o1 = self.final_kilos(together)
        return torch.cat((o1,o2),dim=1)


    def logistic_pmf(self, X) -> torch.Tensor:
        # Safe clamp ranges to prevent NaNs
        K = torch.clamp(X[0], min=1e-3)
        r = torch.clamp(X[1], min=1e-4, max=5.0)
        t0 = torch.clamp(X[2], min=8, max=float(self.t[-1]))


        t = self.t
        cumulative = K * torch.sigmoid(r * (t - t0))


        prepend_val = torch.zeros(1, dtype=cumulative.dtype)
        pmf = torch.diff(cumulative, prepend=prepend_val)

        return pmf
