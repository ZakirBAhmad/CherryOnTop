import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import src.model as models


def dist_epoch(model, optimizer, scheduler, criterion, train_loader, val_loader, num_weeks):
    model.train()
    train_loss = 0
    for batch in train_loader:
        features, encoded_features, climate_data, kilo_gru_input, kilo_dist, log1p_kilos, _,_, _ = batch
        kilo_gru_input = kilo_gru_input[:,:num_weeks,:]

        dist_output = model(features, encoded_features, climate_data, kilo_gru_input)
        if torch.isnan(dist_output).any():
            raise ValueError('NaN in dist output')
        
        pred_input = torch.expm1(dist_output[:,num_weeks:].cumsum(dim=1) * log1p_kilos.unsqueeze(-1))
        actual_input = torch.expm1(kilo_dist[:,num_weeks:].cumsum(dim=1) * log1p_kilos.unsqueeze(-1))

        loss = criterion(pred_input, actual_input)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
    train_loss /= len(train_loader)
    scheduler.step()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            features, encoded_features, climate_data, kilo_gru_input, kilo_dist, log1p_kilos, log1p_schedule,_,_ = batch
            kilo_gru_input = kilo_gru_input[:,:num_weeks,:]
            kilo_output = model(features, encoded_features, climate_data, kilo_gru_input)
            pred_input = torch.expm1(kilo_output[:,num_weeks:].cumsum(dim=1) * log1p_kilos.unsqueeze(-1))
            actual_input = torch.expm1(kilo_dist[:,num_weeks:].cumsum(dim=1) * log1p_kilos.unsqueeze(-1))

            loss = criterion(pred_input, actual_input)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return train_loss, val_loss

def sched_epoch(model, optimizer, scheduler, criterion, train_loader, val_loader, num_weeks):
    model.train()
    train_loss = 0
    for batch in train_loader:
        features, encoded_features, climate_data, kilo_gru_input, _, _, log1p_schedule,_,_ = batch
        kilo_gru_input = kilo_gru_input[:,:num_weeks,:]

        sched_output = model(features, encoded_features, climate_data, kilo_gru_input)
        if torch.isnan(sched_output).any():
            raise ValueError('NaN in sched output')
        
        pred_input = torch.expm1(sched_output)
        actual_input = torch.expm1(log1p_schedule)

        loss = criterion(pred_input, actual_input)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
    train_loss /= len(train_loader)
    scheduler.step()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            features, encoded_features, climate_data, kilo_gru_input, _, _, log1p_schedule,_,_ = batch
            kilo_gru_input = kilo_gru_input[:,:num_weeks,:]
            sched_output = model(features, encoded_features, climate_data, kilo_gru_input)


            loss = criterion(sched_output, log1p_schedule)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return train_loss, val_loss

def kilo_epoch(model, optimizer, scheduler, criterion, train_loader, val_loader, num_weeks):

    model.train()
    train_loss = 0
    for batch in train_loader:
        features, encoded_features, climate_data, kilo_gru_input, kilo_dist, log1p_kilos, _,_,_ = batch
        kilo_gru_input = kilo_gru_input[:,:num_weeks,:]

        kilo_output = model(features, encoded_features, climate_data, kilo_gru_input)
        if torch.isnan(kilo_output).any():
            raise ValueError('NaN in kilo output')
        
        pred_input = torch.expm1(kilo_output)
        actual_input = torch.expm1(log1p_kilos).unsqueeze(-1)

        loss = criterion(pred_input, actual_input)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    scheduler.step()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            features, encoded_features, climate_data, kilo_gru_input, kilo_dist, log1p_kilos, log1p_schedule,_,_ = batch
            kilo_gru_input = kilo_gru_input[:,:num_weeks,:]
            kilo_output = model(features, encoded_features, climate_data, kilo_gru_input)
            pred_input = torch.expm1(kilo_output)
            actual_input = torch.expm1(log1p_kilos).unsqueeze(-1)

            loss = criterion(pred_input, actual_input)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return train_loss, val_loss

def train_model(model, 
                optimizer, 
                scheduler, 
                criterion, 
                train_loader,
                val_loader, 
                num_weeks,
                epoch_func,
                num_epochs,
                patience=30,
                min_epochs=50):
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss, val_loss = epoch_func(model, optimizer, scheduler, criterion, train_loader, val_loader, num_weeks)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
            epochs_no_improve = 0
        elif epoch > (min_epochs-patience):
            epochs_no_improve += 1
        
        # Early stopping after minimum epochs
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    print(f'Week {num_weeks} Best epoch: {best_epoch}')
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model,epoch

def train_trio(train_dataset,
                test_dataset,
                num_weeks, 
                num_epochs,
                tolerance = 5,
                lr = 1e-4,
                weight_decay = 1e-4,
                step_size = 10,
                gamma = 0.1,
                batch_size = 64,
                patience = 30,
                min_epochs = 50):
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()

    dist_model = models.DistModel()
    dist_optimizer = torch.optim.Adam(dist_model.parameters(), lr=lr,weight_decay=weight_decay)
    dist_scheduler = torch.optim.lr_scheduler.StepLR(dist_optimizer, step_size=step_size, gamma=gamma)

    sched_model = models.ScheduleModel()
    sched_optimizer = torch.optim.Adam(sched_model.parameters(), lr=lr,weight_decay=weight_decay)
    sched_scheduler = torch.optim.lr_scheduler.StepLR(sched_optimizer, step_size=step_size, gamma=gamma)

    kilo_model = models.KiloModel()
    kilo_optimizer = torch.optim.Adam(kilo_model.parameters(), lr=lr,weight_decay=weight_decay)
    kilo_scheduler = torch.optim.lr_scheduler.StepLR(kilo_optimizer, step_size=step_size, gamma=gamma)

    for i in range(tolerance):
        try:
            train_model(kilo_model, kilo_optimizer, kilo_scheduler, criterion, train_dataloader, test_dataloader, num_weeks=num_weeks, num_epochs=num_epochs,epoch_func=kilo_epoch,patience=patience,min_epochs=min_epochs)
            break
        except ValueError as e:
            print(f"Yield Attempt {i + 1} failed with error: {e}")
            if i == tolerance - 1:
                print("All yield attempts failed.")
                raise
    for i in range(tolerance):
        try:
            train_model(dist_model, dist_optimizer, dist_scheduler, criterion, train_dataloader, test_dataloader, num_weeks=num_weeks, num_epochs=num_epochs,epoch_func=dist_epoch,patience=patience,min_epochs=min_epochs)
            break
        except ValueError as e:
            print(f"Dist Attempt {i + 1} failed with error: {e}")
            if i == tolerance - 1:
                print("All dist attempts failed.")
                raise

    for i in range(tolerance):
        try:
            train_model(sched_model, sched_optimizer, sched_scheduler, criterion, train_dataloader, test_dataloader, num_weeks=num_weeks, num_epochs=num_epochs,epoch_func=sched_epoch,patience=patience,min_epochs=min_epochs)
            break
        except ValueError as e:
            print(f"SchedAttempt {i + 1} failed with error: {e}")
            if i == tolerance - 1:
                print("All sched attempts failed.")
                raise
    return dist_model, sched_model,kilo_model

def train_full(train_dataset, test_dataset, num_weeks, num_epochs,tolerance = 5,lr = 1e-4,weight_decay = 1e-4,step_size = 10,gamma = 0.1,batch_size = 64,patience = 30,min_epochs = 50):
    models = {}
    for week in range(num_weeks):
        try:
            dist_model, sched_model, kilo_model = train_trio(train_dataset, test_dataset, week + 1, num_epochs,tolerance,lr,weight_decay,step_size,gamma,batch_size,patience,min_epochs)
            models[week + 1] = {'dist_model': dist_model, 'sched_model': sched_model, 'kilo_model': kilo_model}
            print(f"Week {week + 1} trained successfully")
        except ValueError as e:
            print(f"Week {week + 1} failed with error: {e}")
            raise
        
    return models