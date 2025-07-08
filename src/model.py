import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ClimateEncoder(nn.Module):
    def __init__(self,
                 input_dim=5,
                 embedding_dim=12,
                 hidden_dim=32,
                 n_ranches=13,
                 n_parcels=44,
                 n_classes=2,
                 n_types=14,
                 n_varieties=59,
                 climate_input_dim=9,
                 climate_hidden_dim=32):
        super().__init__()

        # Feature processing
        self.feature_encoder = nn.Sequential(
            weight_norm(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU()
        )

        # Climate GRU
        self.climate_gru = nn.GRU(
            input_size=climate_input_dim, 
            hidden_size=climate_hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        # Embedding dimensions
        self.ranch_dim = embedding_dim  # 12 ranches
        self.parcel_dim = embedding_dim  # 12 ranches
        self.class_dim = embedding_dim  # 2 classes
        self.type_dim = embedding_dim  # 14 types
        self.variety_dim = embedding_dim  # 38 varieties

        self.ranch_emb = nn.Embedding(n_ranches, self.ranch_dim)
        self.parcel_emb = nn.Embedding(n_parcels, self.parcel_dim)

        self.parcel_to_ranch = weight_norm(nn.Linear(self.parcel_dim, self.ranch_dim))

        self.class_emb = nn.Embedding(n_classes, self.class_dim)
        self.type_emb = nn.Embedding(n_types, self.type_dim)
        self.variety_emb = nn.Embedding(n_varieties, self.variety_dim)

        self.type_to_class = weight_norm(nn.Linear(self.type_dim, self.class_dim))
        self.variety_to_type = weight_norm(nn.Linear(self.variety_dim, self.type_dim))

        self.combined_dim = (
            hidden_dim +             
            climate_hidden_dim +     
            self.ranch_dim + 
            self.parcel_dim + 
            self.class_dim + 
            self.type_dim + 
            self.variety_dim
        )


    def forward(self, features, encoded_features, climate_data):
        """
        features: (batch_size, 5)
        climate_data: (batch_size, 100, )
        """

        # Static feature encoder
        h_features = self.feature_encoder(features)

        # Climate GRU
        batch_size = climate_data.size(0)
        h0 = torch.zeros(2, batch_size, self.climate_gru.hidden_size).to(climate_data.device)
        out, _ = self.climate_gru(climate_data, h0)  # out: (batch_size, seq_len, hidden_size)

        # Take last timestep
        climate_out = out[:, -1, :]  # (batch_size, climate_hidden_dim)

        ranch_id, parcel_id, class_id, type_id, variety_id = encoded_features.T

        # Embeddings
        r_emb = self.ranch_emb(ranch_id)
        p_emb = self.parcel_emb(parcel_id)

        p_influence_on_ranch = self.parcel_to_ranch(p_emb)
        r_emb = r_emb + p_influence_on_ranch

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
            p_emb,
            c_emb,
            t_emb,
            v_emb
        ], dim=-1)

        return combined

class DistModel(nn.Module):
    def __init__(self,
                 output_dim=40,
                 hidden_size=32,
                 batch_size=32):
        super().__init__()
        
        self.output_dim = output_dim
        
        self.encoder = ClimateEncoder()

        self.kilo_gru = nn.GRU(
            input_size=1,
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
        kilo_dist_input: (batch_size, 40, 2)
        """
        encoded = self.encoder(features, encoded_features, climate_data)
        batch_size = climate_data.size(0)
        
        # Ensure h0 is on the same device as the input
        h0 = torch.zeros(2, batch_size, self.kilo_gru.hidden_size, device=kilo_gru_input.device)
        out, _ = self.kilo_gru(kilo_gru_input, h0)
        result = out[:, -1, :]
    
        kilo_output = self.kilo_output(torch.cat((encoded,result),dim=1))
        return kilo_output
    
class ScheduleModel(nn.Module):
    def __init__(self,
                 output_dim=5,
                 batch_size=32,
                 hidden_size=32):
        super().__init__()
        
        self.output_dim = output_dim
        self.encoder = ClimateEncoder()

        self.kilo_gru = nn.GRU(
            input_size=1,
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
        result = out[:, -1, :]
    
        schedule = self.schedule_output(torch.cat((encoded,result),dim=1))
        return schedule
    
class YieldModel(nn.Module):
    def __init__(self,
                    batch_size=32,
                    hidden_size=32):
        super().__init__()

        self.encoder = ClimateEncoder()
        self.yield_gru = nn.GRU(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        self.final_output = nn.Sequential(
            weight_norm(nn.Linear(self.encoder.combined_dim+hidden_size, batch_size)),
            nn.ReLU(),
            weight_norm(nn.Linear(batch_size, 1))
        )
        
    def forward(self, features, encoded_features, climate_data, yield_gru_input):
        encoded = self.encoder(features, encoded_features, climate_data)
        batch_size = climate_data.size(0)
        # Ensure h0 is on the same device as the input
        h0 = torch.zeros(2, batch_size, self.yield_gru.hidden_size, device=yield_gru_input.device)
        out, _ = self.yield_gru(yield_gru_input, h0)
        result = out[:, -1, :]
        yield_output = self.final_output(torch.cat((encoded, result), dim=1))
        return yield_output