import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import streamlit as st
from src.model import HarvestModel  

@st.cache_resource
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

@st.cache_data
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