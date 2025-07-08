import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import src.model as models

def check_for_nans(tensor, name):
    """Helper function to check for NaNs and print debugging info"""
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Min: {tensor.min()}, Max: {tensor.max()}")
        print(f"  Mean: {tensor.mean()}")
        return True
    return False

def train_dist_model(dist_model, dist_optimizer, dist_scheduler, criterion, dataloader, val_dataloader, num_weeks,num_epochs):
    for epoch in range(num_epochs):
        dist_model.train()
        total_loss = 0
        for batch in dataloader:
            features, encoded_features, climate_data, yield_dist, kilo_dist, yield_log, schedule,_ = batch
            
            kilo_gru_input = kilo_dist[:,:num_weeks,:]
            dist_output = dist_model(features, encoded_features, climate_data, kilo_gru_input)
            dist_output = dist_output.squeeze(-1)

            total_kilos = (torch.exp(yield_log) * features[:,0]).view(-1,1).detach()
                        # Check model output for NaNs
            if check_for_nans(dist_output, "dist_output"):
                print("Skipping batch due to NaN in model output")
                print(f"  dist_output: {dist_output.detach().numpy()}")
                raise ValueError("NaN detected in model output")
            
            loss = criterion(total_kilos * dist_output, total_kilos * kilo_dist.squeeze(-1) )

            dist_optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(dist_model.parameters(), max_norm=2.0)
            
            dist_optimizer.step()
            total_loss += loss.item()
        dist_scheduler.step()

        val_loss = 0
        dist_model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                features, encoded_features, climate_data, yield_dist, kilo_dist, yield_log, schedule,_ = batch
                kilo_gru_input = kilo_dist[:,:num_weeks,:]
                dist_output = dist_model(features, encoded_features, climate_data, kilo_gru_input)
                dist_output = dist_output.squeeze(-1)
                total_kilos = (torch.exp(yield_log) * features[:,0]).view(-1,1).detach()
                loss = criterion(total_kilos * dist_output, total_kilos * kilo_dist.squeeze(-1))
                val_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}, Val Loss: {val_loss/len(val_dataloader)}")

def train_sched_model(sched_model, sched_optimizer, sched_scheduler, criterion, dataloader, val_dataloader, num_weeks,num_epochs):
    for epoch in range(num_epochs):
        sched_model.train()
        total_loss = 0
        for batch in dataloader:
            features, encoded_features, climate_data, yield_dist, kilo_dist, yield_log, schedule,_ = batch
            
            kilo_gru_input = kilo_dist[:,:num_weeks,:]
            sched_output = sched_model(features, encoded_features, climate_data, kilo_gru_input)
            sched_output = sched_output.squeeze(-1)
            total_kilos = (torch.exp(yield_log) * features[:,0]).view(-1,1).detach()
            loss = criterion(total_kilos * sched_output, total_kilos * schedule)

            sched_optimizer.zero_grad()
            loss.backward()
            
                        # Check model output for NaNs
            if check_for_nans(sched_output, "sched_output"):
                print("Skipping batch due to NaN in model output")
                print(f"  sched_output: {sched_output.detach().numpy()}")
                raise ValueError("NaN detected in model output")
            
            torch.nn.utils.clip_grad_norm_(sched_model.parameters(), max_norm=2.0)
            
            sched_optimizer.step()
            total_loss += loss.item()
        sched_scheduler.step()

        val_loss = 0
        sched_model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                features, encoded_features, climate_data, yield_dist, kilo_dist, yield_log, schedule,_ = batch
                kilo_gru_input = kilo_dist[:,:num_weeks,:]
                sched_output = sched_model(features, encoded_features, climate_data, kilo_gru_input)
                sched_output = sched_output.squeeze(-1)
                total_kilos = (torch.exp(yield_log) * features[:,0]).view(-1,1).detach()
                loss = criterion(total_kilos * sched_output, total_kilos * schedule)

                val_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}, Val Loss: {val_loss/len(val_dataloader)}")


def train_yield_model(yield_model, yield_optimizer, yield_scheduler, criterion, dataloader, val_dataloader, num_weeks,num_epochs):
    for epoch in range(num_epochs):
        yield_model.train()
        total_loss = 0
        for batch in dataloader:
            features, encoded_features, climate_data, yield_dist, kilo_dist, yield_log, schedule,_ = batch
            
            
            yield_gru_input = yield_dist[:,:num_weeks,:]
            yield_output = yield_model(features, encoded_features, climate_data, yield_gru_input)
            yield_output = yield_output.squeeze(-1)
            
            # Check model output for NaNs
            if check_for_nans(yield_output, "yield_output"):
                print("Skipping batch due to NaN in model output")
                print(f"  yield_output: {yield_output.detach().numpy()}")
                raise ValueError("NaN detected in model output")
            
            # Better clamping to prevent extreme values
            yield_output = torch.clamp(yield_output, min=0, max=12)

            predicted_kilos = torch.exp(yield_output) * features[:,0]

            total_kilos = (torch.exp(yield_log) * features[:,0]).detach()
            # Use MSE loss in log space instead of exponential
            loss = criterion(predicted_kilos, total_kilos)

            if torch.isnan(loss).any():
                print("NaN detected in training loss, skipping this batch.")
                print(f"  yield_output range: {yield_output.min()} to {yield_output.max()}")
                print(f"  yield_log range: {yield_log.min()} to {yield_log.max()}")
                raise ValueError("NaN detected in training loss")

            yield_optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(yield_model.parameters(), max_norm=2.0)
            
            # Check gradients for NaNs
            has_nan_grad = False
            for name, param in yield_model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
                    has_nan_grad = True
            
            if has_nan_grad:
                print("Skipping optimizer step due to NaN gradients")
                raise ValueError("NaN detected in gradients")
                
            yield_optimizer.step()
            total_loss += loss.item()
        yield_scheduler.step()

        val_loss = 0
        yield_model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                features, encoded_features, climate_data, yield_dist, kilo_dist, yield_log, schedule,_ = batch
                
                # Check inputs for NaNs
                if check_for_nans(features, "val_features") or check_for_nans(climate_data, "val_climate_data") or check_for_nans(yield_dist, "val_yield_dist") or check_for_nans(yield_log, "val_yield_log"):
                    print("Skipping validation batch due to NaN in inputs")
                    raise ValueError("NaN detected in validation inputs")
                    
                yield_gru_input = yield_dist[:,:num_weeks,:]
                yield_output = yield_model(features, encoded_features, climate_data, yield_gru_input)
                yield_output = yield_output.squeeze(-1)
                # Better clamping to prevent extreme values
                yield_output = torch.clamp(yield_output, min=0, max=12)

                predicted_kilos = torch.exp(yield_output) * features[:,0]
                total_kilos = (torch.exp(yield_log) * features[:,0]).detach()
                loss = criterion(predicted_kilos, total_kilos)
                
                
                # Use MSE loss in log space instead of exponential
                #loss = criterion(torch.exp(yield_output), torch.exp(yield_log))

                if torch.isnan(loss).any():
                    print("NaN detected in validation loss, skipping this batch.")
                    raise ValueError("NaN detected in validation loss")

                val_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}, Val Loss: {val_loss/len(val_dataloader)}")

def train_error(train_dataset,
                test_dataset,
                num_weeks, 
                num_epochs,
                tolerance = 5,
                lr = 1e-4,
                weight_decay = 1e-4,
                step_size = 10,
                gamma = 0.1,
                batch_size = 64):
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()

    dist_model = models.DistModel()
    dist_optimizer = torch.optim.Adam(dist_model.parameters(), lr=lr,weight_decay=weight_decay)
    dist_scheduler = torch.optim.lr_scheduler.StepLR(dist_optimizer, step_size=step_size, gamma=gamma)

    yield_model = models.YieldModel()
    yield_optimizer = torch.optim.Adam(yield_model.parameters(), lr=lr,weight_decay=weight_decay)
    yield_scheduler = torch.optim.lr_scheduler.StepLR(yield_optimizer, step_size=step_size, gamma=gamma)

    sched_model = models.ScheduleModel()
    sched_optimizer = torch.optim.Adam(sched_model.parameters(), lr=lr,weight_decay=weight_decay)
    sched_scheduler = torch.optim.lr_scheduler.StepLR(sched_optimizer, step_size=step_size, gamma=gamma)

    for i in range(tolerance):
        try:

            train_yield_model(yield_model, yield_optimizer, yield_scheduler, criterion, train_dataloader, test_dataloader, num_weeks=num_weeks, num_epochs=num_epochs)
            break
        except ValueError as e:
            print(f"Yield Attempt {i + 1} failed with error: {e}")
            if i == tolerance - 1:
                print("All yield attempts failed.")
                raise
    for i in range(tolerance):
        try:
            train_dist_model(dist_model, dist_optimizer, dist_scheduler, criterion, train_dataloader, test_dataloader, num_weeks=num_weeks, num_epochs=num_epochs)
            break
        except ValueError as e:
            print(f"Dist Attempt {i + 1} failed with error: {e}")
            if i == tolerance - 1:
                print("All dist attempts failed.")
                raise

    for i in range(tolerance):
        try:
            train_sched_model(sched_model, sched_optimizer, sched_scheduler, criterion, train_dataloader, test_dataloader, num_weeks=num_weeks, num_epochs=num_epochs)
            break
        except ValueError as e:
            print(f"SchedAttempt {i + 1} failed with error: {e}")
            if i == tolerance - 1:
                print("All sched attempts failed.")
                raise
    return dist_model, yield_model, sched_model

def train_full(train_dataset, test_dataset, num_weeks, num_epochs,tolerance = 5,lr = 1e-4,weight_decay = 1e-4,step_size = 10,gamma = 0.1):
    models = {}
    for week in range(num_weeks):
        try:
            dist_model, yield_model, sched_model = train_error(train_dataset, test_dataset, week + 1, num_epochs,tolerance,lr,weight_decay,step_size,gamma)
            models[week + 1] = {'dist_model': dist_model, 'yield_model': yield_model, 'sched_model': sched_model}
            print(f"Week {week + 1} trained successfully")
        except ValueError as e:
            print(f"Week {week + 1} failed with error: {e}")
            raise
        
    return models