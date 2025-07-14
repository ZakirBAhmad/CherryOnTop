import torch
from torch.utils.data import Dataset

class HarvestDataset(Dataset):


    def __init__(self,
                 features,
                 encoded_features,
                 climate_data,
                 kilo_dist,
                 log1p_kilos,      
                 log1p_schedule,
                 Y_kilos
                ):

        self.features = torch.tensor(features, dtype=torch.float32)
        self.encoded_features = torch.tensor(encoded_features, dtype=torch.long)
        self.climate_data = torch.tensor(climate_data, dtype=torch.float32)
        self.kilo_dist = torch.tensor(kilo_dist, dtype=torch.float32)
        week_tensor = torch.arange(1, 41).repeat(kilo_dist.shape[0], 1).unsqueeze(-1)
        self.kilo_gru_input = torch.cat((self.kilo_dist.unsqueeze(-1), week_tensor), dim=-1)
        self.Y_kilos = torch.tensor(Y_kilos, dtype=torch.float32)



        self.log1p_kilos = torch.tensor(log1p_kilos, dtype=torch.float32)
        self.log1p_schedule = torch.tensor(log1p_schedule, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        climate_data = self.climate_data[idx]

        return (
            self.features[idx],
            self.encoded_features[idx],
            climate_data,
            self.kilo_gru_input[idx],
            self.kilo_dist[idx],
            self.log1p_kilos[idx],
            self.log1p_schedule[idx],
            self.Y_kilos[idx],
            idx
        )
    
    def get_shapes(self):

        shapes = {
            'features': self.features.shape,
            'encoded_features': self.encoded_features.shape,
            'climate_data': self.climate_data.shape,
            'kilo_gru_input': self.kilo_gru_input.shape,
            'kilo_dist': self.kilo_dist.shape,
            'log1p_kilos': self.log1p_kilos.shape,
            'log1p_schedule': self.log1p_schedule.shape,
            'Y_kilos': self.Y_kilos.shape
        }
        return shapes