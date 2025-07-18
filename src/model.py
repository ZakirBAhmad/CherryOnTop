import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class ClimateEncoder(nn.Module):
    def __init__(self,
                 input_dim=5,
                 embedding_dim=12,
                 ranch_dim=None,
                 parcel_dim=None,
                 class_dim=None,
                 type_dim=None,
                 variety_dim=None,
                 hidden_dim=32,
                 n_ranches=13,
                 n_parcels=44,
                 n_classes=2,
                 n_types=14,
                 n_varieties=59,
                 climate_input_dim=9,
                 climate_hidden_dim=32,
                 num_layers=2,
                 dropout=0.2):
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
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Embedding dimensions
        self.ranch_dim = ranch_dim if ranch_dim is not None else embedding_dim  # 12 ranches
        self.parcel_dim = parcel_dim if parcel_dim is not None else embedding_dim  # 12 ranches
        self.class_dim = class_dim if class_dim is not None else embedding_dim  # 2 classes
        self.type_dim = type_dim if type_dim is not None else embedding_dim  # 14 types
        self.variety_dim = variety_dim if variety_dim is not None else embedding_dim  # 38 varieties

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
                 batch_size=32,
                 dropout=0.2,
                 num_layers=2,
                 climate_hidden_dim=32,
                 climate_num_layers=2,
                 climate_dropout=0.2,
                 embedding_dim=12,
                 ranch_dim=None,
                 parcel_dim=None,
                 class_dim=None,
                 type_dim=None,
                 variety_dim=None):
        super().__init__()
        
        self.output_dim = output_dim
        
        self.encoder = ClimateEncoder(
            climate_hidden_dim=climate_hidden_dim,
            num_layers=climate_num_layers,
            dropout=climate_dropout,
            ranch_dim=ranch_dim,
            parcel_dim=parcel_dim,
            class_dim=class_dim,
            type_dim=type_dim,
            variety_dim=variety_dim,
            embedding_dim=embedding_dim
        )

        self.kilo_gru = nn.GRU(
            input_size=2,
            num_layers=num_layers,
            dropout=dropout,
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
        climate_data: (batch_size, 100, 9)
        kilo_gru_input: (batch_size, 40, 2)
        """
        encoded = self.encoder(features, encoded_features, climate_data)
        batch_size = climate_data.size(0)
        
        # Ensure h0 is on the same device as the input
        h0 = torch.zeros(2, batch_size, self.kilo_gru.hidden_size)
        out, _ = self.kilo_gru(kilo_gru_input, h0)
        result = out[:, -1, :]
    
        kilo_output = self.kilo_output(torch.cat((encoded, result), dim=1))
        kilo_output = torch.clamp(kilo_output, min=0)
        kilo_output = kilo_output / kilo_output.sum(dim=1, keepdim=True)
        return kilo_output
    
class ScheduleModel(nn.Module):
    def __init__(self,
                 output_dim=5,
                 batch_size=32,
                 hidden_size=32,
                 dropout=0.2,
                 num_layers=2,
                 climate_hidden_dim=32,
                 climate_num_layers=2,
                 climate_dropout=0.2,
                 embedding_dim=12,
                 ranch_dim=None,
                 parcel_dim=None,
                 class_dim=None,
                 type_dim=None,
                 variety_dim=None):
        super().__init__()
        
        self.output_dim = output_dim
        self.encoder = ClimateEncoder(
            climate_hidden_dim=climate_hidden_dim,
            num_layers=climate_num_layers,
            dropout=climate_dropout,
            ranch_dim=ranch_dim,
            parcel_dim=parcel_dim,
            class_dim=class_dim,
            type_dim=type_dim,
            variety_dim=variety_dim,
            embedding_dim=embedding_dim
        )

        self.kilo_gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
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
        climate_data: (batch_size, 100, 9)
        kilo_gru_input: (batch_size, [5:20], 2)
        """
        encoded = self.encoder(features, encoded_features, climate_data)
        batch_size = climate_data.size(0)
        
        # Ensure h0 is on the same device as the input
        h0 = torch.zeros(2, batch_size, self.kilo_gru.hidden_size)
        out, _ = self.kilo_gru(kilo_gru_input, h0)
        result = out[:, -1, :]
    
        schedule = self.schedule_output(torch.cat((encoded,result),dim=1))
        return schedule
    
class KiloModel(nn.Module):
    def __init__(self,
                    batch_size=32,
                    hidden_size=32,
                    dropout=0.2,
                    num_layers=2,
                    climate_hidden_dim=32,
                    climate_num_layers=2,
                    climate_dropout=0.2,
                    embedding_dim=12,
                    ranch_dim=None,
                    parcel_dim=None,
                    class_dim=None,
                    type_dim=None,
                    variety_dim=None):
        super().__init__()

        self.encoder = ClimateEncoder(
            climate_hidden_dim=climate_hidden_dim,
            num_layers=climate_num_layers,
            dropout=climate_dropout,
            ranch_dim=ranch_dim,
            parcel_dim=parcel_dim,
            class_dim=class_dim,
            type_dim=type_dim,
            variety_dim=variety_dim,
            embedding_dim=embedding_dim
        )
        self.kilo_gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.final_output = nn.Sequential(
            weight_norm(nn.Linear(self.encoder.combined_dim+hidden_size, batch_size)),
            nn.ReLU(),
            weight_norm(nn.Linear(batch_size, 1))
        )
        
    def forward(self, features, encoded_features, climate_data, kilo_gru_input):
        encoded = self.encoder(features, encoded_features, climate_data)
        batch_size = climate_data.size(0)
        # Ensure h0 is on the same device as the input
        h0 = torch.zeros(2, batch_size, self.kilo_gru.hidden_size)
        out, _ = self.kilo_gru(kilo_gru_input, h0)
        result = out[:, -1, :]
        kilo_output = self.final_output(torch.cat((encoded, result), dim=1))
        return kilo_output