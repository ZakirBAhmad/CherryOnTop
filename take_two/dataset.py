import torch
from torch.utils.data import Dataset

class Data_Take2(Dataset):


    def __init__(self, 
                 features,
                 encoded_features,
                 climate_data,
                 yield_dist,       
                 kilo_dist,
                 yield_log,       
                 schedule
                ):

        self.features = torch.tensor(features, dtype=torch.float32)
        self.encoded_features = torch.tensor(encoded_features, dtype=torch.long)
        self.climate_data = torch.tensor(climate_data, dtype=torch.float32)
        self.yield_dist = torch.tensor(yield_dist, dtype=torch.float32).unsqueeze(2)
        self.kilo_dist = torch.tensor(kilo_dist, dtype=torch.float32).unsqueeze(2)

        self.Y_yield_log = torch.tensor(yield_log, dtype=torch.float32)
        self.Y_schedule = torch.tensor(schedule, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        climate_data = self.climate_data[idx]

        return (
            self.features[idx],
            self.encoded_features[idx],
            climate_data,
            self.yield_dist[idx],
            self.kilo_dist[idx],
            self.Y_yield_log[idx],
            self.Y_schedule[idx],
            idx
        )
    
    def get_shapes(self):

        shapes = {
            'features': self.features.shape,
            'encoded_features': self.encoded_features.shape,
            'climate_data': self.climate_data.shape,
            'yield_dist': self.yield_dist.shape,
            'kilo_dist': self.kilo_dist.shape,
            'yield_log': self.Y_yield_log.shape,
            'schedule': self.Y_schedule.shape
        }
        return shapes