import torch
import torch.nn as nn

class ClimateEncoder(nn.Module):
    def __init__(self,
                 input_dim=5,
                 embedding_dim=4,
                 hidden_dim=64,
                 n_ranches=13,
                 n_classes=2,
                 n_types=14,
                 n_varieties=59,
                 climate_input_dim=3,
                 climate_hidden_dim=32):
        super().__init__()

        # Feature processing
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # Climate GRU
        self.climate_gru = nn.GRU(
            input_size=climate_input_dim,      # 3 features: temp_max, temp_min, precipitation
            hidden_size=climate_hidden_dim,     # You choose (maybe 32)
            batch_first=True
        )

        # Embedding dimensions
        self.ranch_dim = embedding_dim  # 12 ranches
        self.class_dim = embedding_dim  # 2 classes
        self.type_dim = embedding_dim  # 14 types
        self.variety_dim = embedding_dim  # 38 varieties

        self.ranch_emb = nn.Embedding(n_ranches, self.ranch_dim)
        self.class_emb = nn.Embedding(n_classes, self.class_dim)
        self.type_emb = nn.Embedding(n_types, self.type_dim)
        self.variety_emb = nn.Embedding(n_varieties, self.variety_dim)

        self.type_to_class = nn.Linear(self.type_dim, self.class_dim)
        self.variety_to_type = nn.Linear(self.variety_dim, self.type_dim)

        self.combined_dim = (
            hidden_dim +             # static features
            climate_hidden_dim +     # output from GRU
            self.ranch_dim + 
            self.class_dim + 
            self.type_dim + 
            self.variety_dim
        )


    def forward(self, features, ranch_id, class_id, type_id, variety_id, climate_data):
        """
        features: (batch_size, 5)
        climate_data: (batch_size, 100, 3)
        """

        # Static feature encoder
        h_features = self.feature_encoder(features)

        # Climate GRU
        batch_size = climate_data.size(0)
        h0 = torch.zeros(1, batch_size, self.climate_gru.hidden_size).to(climate_data.device)
        out, _ = self.climate_gru(climate_data, h0)  # out: (batch_size, seq_len, hidden_size)

        # Take last timestep
        climate_out = out[:, -1, :]  # (batch_size, climate_hidden_dim)

        # Embeddings
        r_emb = self.ranch_emb(ranch_id)
        c_emb = self.class_emb(class_id)
        t_emb = self.type_emb(type_id)
        v_emb = self.variety_emb(variety_id)

        # Hierarchy
        v_influence_on_type = self.variety_to_type(v_emb)
        t_emb = t_emb + v_influence_on_type

        t_influence_on_class = self.type_to_class(t_emb)
        c_emb = c_emb + t_influence_on_class

        # Combine all features
        combined = torch.cat([
            h_features,
            climate_out,
            r_emb,
            c_emb,
            t_emb,
            v_emb
        ], dim=-1)

        return combined
