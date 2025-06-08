import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.updated_model import HarvestModel

def create_model(train_dataset, val_dataset, epochs):

    full_train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    # Get all climate data from training set
    features, ranch_id, class_id, type_id, variety_id, climate_data, y, bounds, _ = next(iter(full_train_loader))

    # Flatten and compute mean/std on training climate data
    climate_data_flat = climate_data.view(-1, climate_data.shape[-1])
    climate_mean = climate_data_flat.mean(dim=0)
    climate_std = climate_data_flat.std(dim=0)

    # Assign these stats to train and val datasets
    train_dataset.dataset.climate_mean = climate_mean  # type: ignore
    train_dataset.dataset.climate_std = climate_std    # type: ignore
    val_dataset.dataset.climate_mean = climate_mean    # type: ignore
    val_dataset.dataset.climate_std = climate_std      # type: ignore

    # Now create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = HarvestModel()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            features, ranch_id, class_id, type_id, variety_id, climate_data, y, bounds, _ = batch
            totals = y.sum().to(dtype=torch.float32)
            bounds = bounds.to(dtype=torch.float32)
            looped_loss = criterion(y,y)
            batch_size = y.size(0)
            log_kilos = torch.log1p(y) 
            week_numbers = torch.arange(0, 20).unsqueeze(0).repeat(batch_size,1)
            inputs = torch.stack([y, log_kilos, week_numbers], dim=2)

            kilo_ranges = torch.arange(4,20,4)

            for kilo_range in kilo_ranges:
        
                kilo_inputs = inputs[:,:kilo_range,:]
                # Forward pass
                outputs, clamp = model(features, ranch_id, class_id, type_id, variety_id, climate_data, kilo_inputs)
                N = outputs.size(0)
                T = outputs.size(1)  # Should be 20

                # Create time indices [0, 1, ..., 19] and expand to shape (N, 20)
                time = torch.arange(T).unsqueeze(0).expand(N, T)

                # Get start and end indices
                start = clamp[:, 0].unsqueeze(1)  # Shape: (N, 1)
                end = clamp[:, 1].unsqueeze(1)    # Shape: (N, 1)

                # Create mask: 1 where i is within [start, end), 0 elsewhere
                mask = (time >= start) & (time < end)  # Shape: (N, 20)

                # Apply mask
                masked_harvests = outputs * mask
                loss_kilos = criterion(masked_harvests[:,kilo_range:], y[:,kilo_range:])
                loss_clamp = criterion(clamp/20, bounds/20)

                loss = loss_kilos + loss_clamp * totals
                looped_loss += loss

                # Backward and optimize
            optimizer.zero_grad()
            looped_loss.backward()
            optimizer.step()
            
            total_loss += looped_loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{200}], Loss: {avg_loss:.4f}")
    
    return model, val_loader