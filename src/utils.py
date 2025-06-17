import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from src.model import HarvestModel  


def create_model(train_dataset, num_epochs=30):

    # Now create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = HarvestModel()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            features, ranch_id, class_id, type_id, variety_id, climate_data, y, bounds, _ = batch
            totals = y.sum().to(dtype=torch.float32)
            bounds = bounds.to(dtype=torch.float32)
            batch_size = y.size(0)
            log_kilos = torch.log1p(y) 
            week_numbers = torch.arange(0, 20).unsqueeze(0).repeat(batch_size,1)
            inputs = torch.stack([y, log_kilos, week_numbers], dim=2)
            looped_loss = criterion(y, y)

            kilo_ranges = torch.arange(4,20,4)

            for kilo_range in kilo_ranges:
        
                kilo_inputs = inputs[:,:kilo_range,:]
                # Forward pass
                outputs= model(features, ranch_id, class_id, type_id, variety_id, climate_data, kilo_inputs)

                looped_loss += criterion(outputs, y)
            
            optimizer.zero_grad()
            looped_loss.backward()
            optimizer.step()

            total_loss += looped_loss.item()
        
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{200}], Loss: {avg_loss:.4f}")
    
    return model

def predict_harvest(_model, _test_dataset):
    _model.eval()
    with torch.no_grad():
        test_predictions = []
        
    
        test_loader = DataLoader(_test_dataset, batch_size=64, shuffle=False)
        for batch in test_loader:

                features, ranch_id, class_id, type_id, variety_id, climate_data, y, bounds, _ = batch

                batch_size = y.size(0)
                inputs = torch.stack([torch.zeros(batch_size,5), torch.ones(batch_size,5), torch.arange(5).unsqueeze(0).repeat(batch_size,1)], dim=2)
                
                outputs = _model(features, ranch_id, class_id, type_id, variety_id, climate_data, inputs)
                test_predictions.append(outputs)
    return torch.cat(test_predictions, dim=0).detach().numpy()

def predict_gridded_harvest(_model, _test_dataset):
    _model.eval()
    with torch.no_grad():
        test_predictions = np.zeros((len(_test_dataset), 20,20))
        
    
        test_loader = DataLoader(_test_dataset, batch_size=64, shuffle=False)
        for batch in test_loader:
            features, ranch_id, class_id, type_id, variety_id, climate_data, y, bounds, idx = batch
            batch_size = y.size(0)
            log_kilos = torch.log1p(y) 
            week_numbers = torch.arange(0, 20).unsqueeze(0).repeat(batch_size,1)
            inputs = torch.stack([y, log_kilos, week_numbers], dim=2)
            for i in range(5,20):
                kilo_inputs = inputs[:,:i,:]
                outputs = _model(features, ranch_id, class_id, type_id, variety_id, climate_data, kilo_inputs)       
                test_predictions[idx,i,:] = np.concat([y[:,:i],outputs.detach().numpy()[:,i:]],axis=1)

    return test_predictions

def train_partial_climate(_train_dataset, climate_step = 10,num_epochs=5, batch_size=32, lr=1e-4):
    # Create DataLoader
    train_loader = DataLoader(_train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model
    model = HarvestModel()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    seq_len = _train_dataset.get_shapes()['climate_data'][1]
    climate_steps = torch.arange(climate_step,seq_len + 1,climate_step)

    model.train()

    for epoch in range(num_epochs):
        total_loss = np.zeros(len(climate_steps))
        for batch in train_loader:
            features, ranch_id, class_id, type_id, variety_id, climate_data, y, _ = batch
            seq_loss = []
            for climate_step in climate_steps:
                
                climate_data_step = climate_data[:, :climate_step, :]

                # Forward pass
                outputs = model(features, ranch_id, class_id, type_id, variety_id, climate_data_step)


                loss = criterion(outputs, y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                seq_loss.append(loss.item())

            
            total_loss += seq_loss

        avg_start_loss = total_loss[0] / len(train_loader)
        avg_end_loss = total_loss[-1] / len(train_loader)
        avg_loss = str(total_loss / len(train_loader))
        min_loss = np.argmin(total_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Start Loss: {avg_start_loss:.4f}, End Loss: {avg_end_loss:.4f}, Avg Loss: {avg_loss}, Min Loss: {min_loss:.4f}")

    return model