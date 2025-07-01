import torch
import numpy as np
from torch.utils.data import DataLoader
from src.model import HarvestScheduleModel, KiloModel, FinalModel
from src.encoder import ClimateEncoder
import torch.nn as nn

def train_trial(train_loader, kilo_model, schedule_model, final_model, criterion, optimizer, num_weeks):
    total_kilo_loss = 0
    total_schedule_loss = 0
    total_final_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        features, encoded_features, climate_data, y_kilos, y_combined, schedule, _= batch
        
        # Tensors are already on the correct device from the dataset
        climate_data = climate_data[:,:num_weeks * 7,:]
        kilo_input = y_combined[:,:num_weeks,:]

        kilo_outputs = kilo_model(features, encoded_features, climate_data, kilo_input)
        kilo_loss = criterion(kilo_outputs, y_kilos)
        kilo_loss.backward()

        schedule_outputs = schedule_model(features.detach(), encoded_features.detach(), climate_data.detach(), kilo_input.detach(),kilo_outputs.detach())
        schedule_loss = criterion(schedule_outputs, schedule)
        schedule_loss.backward()
        
        batch_size = len(y_kilos)
        weeks = (torch.ones(batch_size, device=y_kilos.device) * num_weeks).unsqueeze(1)
        final_outputs = final_model(weeks,kilo_outputs.detach(),schedule_outputs.detach())
        final_loss = criterion(final_outputs, y_kilos)
        final_loss.backward()
        
        optimizer.step()

        total_kilo_loss += kilo_loss.item()
        total_schedule_loss += schedule_loss.item()
        total_final_loss += final_loss.item()

    avg_kilo_loss = total_kilo_loss / len(train_loader)
    avg_schedule_loss = total_schedule_loss / len(train_loader)
    avg_final_loss = total_final_loss / len(train_loader)
    return avg_kilo_loss, avg_schedule_loss, avg_final_loss

def evaluate(val_loader, kilo_model, schedule_model, final_model, criterion, num_weeks):
    total_kilo_loss = 0
    total_schedule_loss = 0
    total_final_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            features, encoded_features, climate_data, y_kilos, y_combined, schedule, _= batch

            climate_data = climate_data[:,:num_weeks * 7,:]
            kilo_input = y_combined[:,:num_weeks,:]

            kilo_outputs = kilo_model(features, encoded_features, climate_data, kilo_input)
            schedule_outputs = schedule_model(features, encoded_features, climate_data, kilo_input, kilo_outputs)

            batch_size = len(y_kilos)
            weeks = (torch.ones(batch_size, device=y_kilos.device) * num_weeks).unsqueeze(1)
            final_outputs = final_model(weeks, kilo_outputs, schedule_outputs)

    
            kilo_loss = criterion(kilo_outputs, y_kilos)
            schedule_loss = criterion(schedule_outputs, schedule)
            final_loss = criterion(final_outputs, y_kilos)

            total_kilo_loss += kilo_loss.item()
            total_schedule_loss += schedule_loss.item()
            total_final_loss += final_loss.item()

    avg_kilo_loss = total_kilo_loss / len(val_loader)
    avg_schedule_loss = total_schedule_loss / len(val_loader)
    avg_final_loss = total_final_loss / len(val_loader)
    return avg_kilo_loss, avg_schedule_loss, avg_final_loss

def run_trial(train_loader, val_loader, kilo_model, schedule_model, final_model, criterion, optimizer, scheduler, num_epochs, save_destination):
    
    patience = 10
    
    trigger_times_kilo = 0
    total_kilo_trigger_times = 0
    trigger_times_schedule = 0
    total_schedule_trigger_times = 0
    trigger_times_final = 0
    total_final_trigger_times = 0

    best_val_loss_kilo = float('inf')
    best_val_loss_schedule = float('inf')
    best_val_loss_final = float('inf')

    losses = np.zeros((num_epochs,13,3))

    best_kilo_model = kilo_model.state_dict()
    kilo_trigger_epoch = -1
    best_kilo_epoch = -1

    best_schedule_models = (kilo_model.state_dict(),schedule_model.state_dict())
    schedule_trigger_epoch = -1
    best_schedule_epoch = -1

    best_final_models = (kilo_model.state_dict(),schedule_model.state_dict(),final_model.state_dict())
    best_final_epoch = -1

    for epoch in range(num_epochs):
        kilo_model.train()
        schedule_model.train()
        final_model.train()
        # Phase 1 - Transplant_Date
        losses[epoch,0,:] = train_trial(train_loader, kilo_model, schedule_model, final_model, criterion, optimizer, 1)
        
        # Phase 2 - End Climate
        week_num = np.random.randint(9,13)
        losses[epoch,1,:] = train_trial(train_loader, kilo_model, schedule_model, final_model, criterion, optimizer, week_num)

        # Phase 3 - Curve Adjustment
        sample = np.random.randint(14,25,5)
        for i, week in enumerate(sample):
            losses[epoch,2+i,:] = train_trial(train_loader, kilo_model, schedule_model, final_model, criterion, optimizer, week)

        # Phase 4 - Validation
        kilo_model.eval()
        schedule_model.eval()
        final_model.eval()
        
        losses[epoch,7,:] = evaluate(val_loader, kilo_model, schedule_model, final_model, criterion, 1)
        losses[epoch,8,:] = evaluate(val_loader, kilo_model, schedule_model, final_model, criterion, 6)
        losses[epoch,9,:] = evaluate(val_loader, kilo_model, schedule_model, final_model, criterion, 11)
        losses[epoch,10,:] = evaluate(val_loader, kilo_model, schedule_model, final_model, criterion, 16)
        losses[epoch,11,:] = evaluate(val_loader, kilo_model, schedule_model, final_model, criterion, 21)
        losses[epoch,12,:] = evaluate(val_loader, kilo_model, schedule_model, final_model, criterion, 26)

        
        scheduler.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch} completed. Total Trigger Times: Kilo - {total_kilo_trigger_times}, Schedule - {total_schedule_trigger_times}, Final - {total_final_trigger_times}')

        if epoch % 50 == 0:
            torch.save(kilo_model.state_dict(), f'{save_destination}/kilo_model_epoch_{epoch}.pth')
            torch.save(schedule_model.state_dict(), f'{save_destination}/schedule_model_epoch_{epoch}.pth')
            torch.save(final_model.state_dict(), f'{save_destination}/final_model_epoch_{epoch}.pth')
            print('Epoch', epoch, 'completed, models saved')

        avg_val_loss_kilo = np.mean(losses[epoch, 7:, 0])
        avg_val_loss_schedule = np.mean(losses[epoch, 7:, 1])
        avg_val_loss_final = np.mean(losses[epoch, 7:, 2])

        # Check for improvement
        if avg_val_loss_kilo < best_val_loss_kilo:
            best_val_loss_kilo = avg_val_loss_kilo
            best_kilo_model = kilo_model.state_dict()
            best_kilo_epoch = epoch
            trigger_times_kilo = 0
            trigger_times_schedule = 0
            trigger_times_final = 0
        else:
            trigger_times_kilo += 1
            total_kilo_trigger_times += 1

        if trigger_times_kilo >= patience and epoch > 100 and kilo_trigger_epoch == -1:
            kilo_trigger_epoch = epoch
        
        if avg_val_loss_schedule < best_val_loss_schedule:
            best_val_loss_schedule = avg_val_loss_schedule
            best_schedule_models = (kilo_model.state_dict(),schedule_model.state_dict())
            best_schedule_epoch = epoch
            trigger_times_schedule = 0
            trigger_times_final = 0
        else:
            trigger_times_schedule += 1
            total_schedule_trigger_times += 1

        if trigger_times_schedule >= patience and epoch > 200 and schedule_trigger_epoch == -1:
            schedule_trigger_epoch = epoch

        if avg_val_loss_final < best_val_loss_final:
            best_val_loss_final = avg_val_loss_final
            best_final_models = (kilo_model.state_dict(),schedule_model.state_dict(),final_model.state_dict())
            best_final_epoch = epoch
            trigger_times_final = 0
        else:
            trigger_times_final += 1
            total_final_trigger_times += 1
        
        if trigger_times_final >= patience and epoch > 300:
            print(f'Final triggered at epoch {epoch}, overfitting detected')
            break

    print(f'Kilo triggered at epoch {kilo_trigger_epoch} and best kilo model saved at epoch {best_kilo_epoch}. Overall, it triggered {total_kilo_trigger_times} times')
    print(f'Schedule triggered at epoch {schedule_trigger_epoch} and best schedule model saved at epoch {best_schedule_epoch}. Overall, it triggered {total_schedule_trigger_times} times')
    print(f'Final triggered at epoch {best_final_epoch}. Overall, it triggered {total_final_trigger_times} times')

    return losses, best_kilo_model, best_schedule_models, best_final_models

def test_model(train_dataset, test_dataset, num_epochs, device, save_destination):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    encoder_schedule = ClimateEncoder().to(device)
    encoder_kilo = ClimateEncoder().to(device)

    kilo_model = KiloModel(encoder_kilo).to(device)
    schedule_model = HarvestScheduleModel(encoder_schedule).to(device)
    final_model = FinalModel().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam([
        {"params": kilo_model.parameters(), "lr": 5e-4},
        {"params": schedule_model.parameters(), "lr": 5e-4},
        {"params": final_model.parameters(), "lr": 1e-3}
        
    ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    report, kilo_state, schedule_models, final_models = run_trial(
        train_loader, 
        val_loader,
        kilo_model, 
        schedule_model, 
        final_model, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs,
        save_destination
        )

    return report, kilo_state, schedule_models, final_models