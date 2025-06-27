#imports
import torch
from torch.utils.data import Dataset
from typing import Union


class HarvestDataset(Dataset):
    """
    A PyTorch Dataset class for handling harvest data with automatic device placement.

    This dataset manages features and various categorical IDs (ranch, class, type, variety) 
    along with target variables for kilos measurements. All tensors are automatically 
    placed on the specified device.

    Parameters
    ----------
    features : numpy.ndarray, shape (N, 5)
        Static features for N samples
    encoded_features : numpy.ndarray, shape (N, 6)
        Encoded categorical features
    climate_data : numpy.ndarray, shape (N, 100, 9)
        Climate time series data
    Y_kilos : numpy.ndarray, shape (N, 20)
        Target kilos measurements for 20 timesteps
    device : torch.device or str, optional
        Device to place tensors on ('cpu', 'cuda', or torch.device object)
        Defaults to 'cpu'

    Returns
    -------
    tuple
        Contains tensors for features, IDs and targets when indexed
    """
    def __init__(self, 
                 features,         # (N, 5)
                 encoded_features, # (N, 6)
                 climate_data,     # (N, 100, 9)
                 Y_kilos,         # (N, 20)
                 Y_schedule,      # (N, 3)
                 device = 'cpu'     # Device for tensor placement
                ):

        # Handle device specification
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Convert to tensors and move to specified device
        self.features = torch.tensor(features, dtype=torch.float32, device=self.device)
        self.encoded_features = torch.tensor(encoded_features, dtype=torch.long, device=self.device)
        self.climate_data = torch.tensor(climate_data, dtype=torch.float32, device=self.device)
        self.Y_kilos = torch.tensor(Y_kilos, dtype=torch.float32, device=self.device)

        self.Y_Schedule = torch.tensor(Y_schedule, dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        climate_data = self.climate_data[idx]

        return (
            self.features[idx],
            self.encoded_features[idx],
            climate_data,
            self.Y_kilos[idx],
            self.Y_Schedule[idx],
            idx
        )
    
    def to(self, device):
        """
        Move all tensors to the specified device.
        
        Parameters
        ----------
        device : torch.device or str
            Target device
            
        Returns
        -------
        HarvestDataset
            Returns self for method chaining
        """
        if isinstance(device, str):
            device = torch.device(device)
            
        self.device = device
        self.features = self.features.to(device)
        self.encoded_features = self.encoded_features.to(device)
        self.climate_data = self.climate_data.to(device)
        self.Y_kilos = self.Y_kilos.to(device)
        self.Y_Schedule = self.Y_Schedule.to(device)
        return self
    
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
            'Y_kilos': self.Y_kilos.shape,
            'Y_schedule': self.Y_Schedule.shape
        }
        return shapes
    
    def get_device(self):
        """
        Returns the device where tensors are stored
        
        Returns
        -------
        torch.device
            Current device of the dataset tensors
        """
        return self.device