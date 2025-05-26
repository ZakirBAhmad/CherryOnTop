import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from src.model import HarvestModel  


def train_harvest_model(_train_dataset, num_epochs=5, batch_size=32, lr=1e-3):
    
    # Create DataLoader
    train_loader = DataLoader(_train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model
    model = HarvestModel()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_loader:
            features, ranch_id, class_id, type_id, variety_id, climate_data, y = batch

    

            # Forward pass
            outputs = model(features, ranch_id, class_id, type_id, variety_id, climate_data)


            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model


def predict_harvest(_model, _test_dataset):
    _model.eval()
    with torch.no_grad():
        test_predictions = []
    
        test_loader = DataLoader(_test_dataset, batch_size=64, shuffle=False)
        for batch in test_loader:
                features, ranch_id, class_id, type_id, variety_id, climate_data, Y_kilos = batch
                outputs = _model(features, ranch_id, class_id, type_id, variety_id, climate_data)
                test_predictions.append(outputs)
    return torch.cat(test_predictions, dim=0).detach().numpy()

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
            features, ranch_id, class_id, type_id, variety_id, climate_data, y = batch
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