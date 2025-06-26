#imports
import torch
from torch.utils.data import Dataset


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
                 encoded_features,     # (N,)
                 climate_data,     # (N, 100, 3)
                 Y_kilos,         # (N, 20)
                ):

        # Convert to tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.encoded_features = torch.tensor(encoded_features, dtype=torch.float32)
        self.climate_data = torch.tensor(climate_data, dtype=torch.float32)
        self.Y = torch.tensor(Y_kilos, dtype=torch.float32)



    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        climate_data = self.climate_data[idx]

        return (
            self.features[idx],
            self.encoded_features[idx],
            climate_data,
            self.Y[idx],
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
            'encoded_features': self.encoded_features.shape,
            'climate_data': self.climate_data.shape,
            'Y_kilos': self.Y.shape
        }
        return shapes