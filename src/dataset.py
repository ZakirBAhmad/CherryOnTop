#imports
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
                 Y_kilos = None          # (N, 20)
                ):
    
        # Validate shapes
        N = len(features)
        if Y_kilos is None:
            Y_kilos = np.zeros((N, 20))
        assert features.shape == (N, 5), f"Expected features shape (N, 5), got {features.shape}"
        assert ranch_ids.shape == (N,), f"Expected ranch_ids shape (N,), got {ranch_ids.shape}"
        assert class_ids.shape == (N,), f"Expected class_ids shape (N,), got {class_ids.shape}"
        assert type_ids.shape == (N,), f"Expected type_ids shape (N,), got {type_ids.shape}"
        assert variety_ids.shape == (N,), f"Expected variety_ids shape (N,), got {variety_ids.shape}"
        assert climate_data.shape == (N, 100, 3), f"Expected climate_data shape (N, 100, 3), got {climate_data.shape}"
        assert Y_kilos.shape == (N, 20), f"Expected Y_kilos shape (N, 20), got {Y_kilos.shape}"

        # Convert to tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.ranch_ids = torch.tensor(ranch_ids, dtype=torch.long)
        self.class_ids = torch.tensor(class_ids, dtype=torch.long)
        self.type_ids = torch.tensor(type_ids, dtype=torch.long)
        self.variety_ids = torch.tensor(variety_ids, dtype=torch.long)
        self.climate_data = torch.tensor(climate_data, dtype=torch.float32)
        self.Y_kilos = torch.tensor(Y_kilos, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.ranch_ids[idx],
            self.class_ids[idx],
            self.type_ids[idx],
            self.variety_ids[idx],
            self.climate_data[idx],
            self.Y_kilos[idx])
    
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
            'Y_kilos': self.Y_kilos.shape
        }
        return shapes