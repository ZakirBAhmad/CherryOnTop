import torch
from torch.utils.data import Dataset
import numpy as np

#class
class HarvestDataset(Dataset):
    """
    A PyTorch Dataset class for handling harvest data.

    This dataset manages features and various categorical IDs (ranch, class, type, variety) 
    along with target variables for kilos measurements.

    Parameters
    ----------
    features : numpy.ndarray, shape (N, 4)
        Static features for N samples (hectares, plants_per_hectare, avg_plant_height, avg_leaf_count)
    ranch_ids : numpy.ndarray, shape (N,)
        Ranch identifier integers
    class_ids : numpy.ndarray, shape (N,)
        Class identifier integers
    type_ids : numpy.ndarray, shape (N,)
        Type identifier integers
    variety_ids : numpy.ndarray, shape (N,)
        Variety identifier integers
    Y_kilos : numpy.ndarray, shape (N, 20)
        Target kilos measurements for 20 timesteps

    Returns
    -------
    tuple
        Contains tensors for features, IDs and targets when indexed
    """
    def __init__(self, 
                 features,         # (N, 5)
                 ranch_ids,        # (N,)
                 class_ids,        # (N,)
                 type_ids,         # (N,)
                 variety_ids,      # (N,)
                 climate_data,     # (N, 100, 3)
                 Y_kilos,         # (N, 20)
                 mean=None,
                 std=None
                ):
    
       

        # Convert to tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.ranch_ids = torch.tensor(ranch_ids, dtype=torch.long)
        self.class_ids = torch.tensor(class_ids, dtype=torch.long)
        self.type_ids = torch.tensor(type_ids, dtype=torch.long)
        self.variety_ids = torch.tensor(variety_ids, dtype=torch.long)
        self.climate_data = torch.tensor(climate_data, dtype=torch.float32)
        self.Y = torch.tensor(Y_kilos, dtype=torch.float32)
        nonzero = self.Y != 0

        idx = torch.arange(self.Y.size(1)).expand_as(self.Y)
        start = torch.where(nonzero, idx, torch.full_like(idx, self.Y.size(1))).min(dim=1).values
        end = torch.where(nonzero, idx, torch.full_like(idx, -1)).max(dim=1).values

        self.bounds = torch.stack([start, end], dim=1)

        self.climate_mean = mean
        self.climate_std = std

             

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        climate_data = self.climate_data[idx]
        if self.climate_mean is not None and self.climate_std is not None:
            climate_data = (climate_data - self.climate_mean) / (self.climate_std + 1e-6)
        return (
            self.features[idx],
            self.ranch_ids[idx],
            self.class_ids[idx],
            self.type_ids[idx],
            self.variety_ids[idx],
            climate_data,
            self.Y[idx],
            self.bounds[idx],
            idx
        )
    
    
    def get_shapes(self):
        """
        Returns a dictionary containing the shapes of all data tensors
        
        Returns
        -------
        dict
            Dictionary with tensor names as keys and their shapes as values
        """
        shapes = {
            'features': self.features.shape,
            'ranch_ids': self.ranch_ids.shape,
            'class_ids': self.class_ids.shape,
            'type_ids': self.type_ids.shape,
            'variety_ids': self.variety_ids.shape,
            'climate_data': self.climate_data.shape,
            'Y_kilos': self.Y.shape
        }
        return shapes
    
def create_dataset(meta, y, mappings):
    features = np.column_stack([
        meta['Ha'].to_numpy(),                    # Hectares
        meta['WeekTransplanted_sin'].to_numpy(),  # Week sine
        meta['WeekTransplanted_cos'].to_numpy(),  # Week cosine
        meta['Year'].to_numpy() - 2010,                  # Year
        np.ones(len(meta))                    # Constant feature
        ])
    
    ranches = meta['Ranch'].to_numpy()
    varieties = meta['Variety'].to_numpy()
    classes = meta['Class'].to_numpy()
    types = meta['Type'].to_numpy()
    climate_data = np.array(meta.ClimateSeries.to_list())
    y_kilos = y.iloc[:,:20].to_numpy()

    dataset = HarvestDataset(
            features,
            ranches,
            classes,
            types,
            varieties,
            climate_data,
            y_kilos
        )
    return dataset