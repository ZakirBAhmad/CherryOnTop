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

        self.t = torch.arange(output_dim)

        self.final_kilos = nn.Sequential(
            nn.Linear(self.encoder.combined_dim + output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim+self.stats_predictor.output_dim)
        )

    def forward(self, features, ranch_id, class_id, type_id, variety_id, climate_data):
        """
        features: (batch_size, 5)
        climate_data: (batch_size, 100, 3)
        """
        encoded = self.encoder(features, ranch_id, class_id, type_id, variety_id, climate_data)

        o2 = self.stats_predictor(encoded)
        pmf = self.logistic_pmf(o2)

        together = torch.cat((encoded,pmf),dim=1)
        o1 = self.final_kilos(together)

        return torch.cat((o1,o2),dim=1)


    def logistic_pmf(self, X) -> torch.Tensor:
        # Step 1: compute cumulative logistic
        K = X[0]
        r = X[1]
        t0 = X[2]
        cumulative = K / (1 + torch.exp(-r * (self.t - t0)))
        
        # Step 2: approximate PMF as discrete difference
        pmf = torch.diff(cumulative, prepend=torch.tensor([0.0], dtype=cumulative.dtype))
        
        return pmf