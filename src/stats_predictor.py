import torch
import torch.nn as nn

class StatsPredictor(nn.Module):
    def __init__(self,
                 encoder_dim,
                 hidden_dim = 32,   
                 output_dim = 6):
        super().__init__()

        self.output_dim = output_dim

        self.stats_predictor = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, encoding):
        """
        encoding: (batch_size, encoder_dim)
        """
        return self.stats_predictor(encoding)