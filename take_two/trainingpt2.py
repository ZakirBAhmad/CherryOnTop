# imports
import take_two.training as utils
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import take_two.models as models


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

            utils.train_yield_model(yield_model, yield_optimizer, yield_scheduler, criterion, train_dataloader, test_dataloader, num_weeks=num_weeks, num_epochs=num_epochs)
            break
        except ValueError as e:
            print(f"Yield Attempt {i + 1} failed with error: {e}")
            if i == tolerance - 1:
                print("All yield attempts failed.")
                raise
    for i in range(tolerance):
        try:
            utils.train_dist_model(dist_model, dist_optimizer, dist_scheduler, criterion, train_dataloader, test_dataloader, num_weeks=num_weeks, num_epochs=num_epochs)
            break
        except ValueError as e:
            print(f"Dist Attempt {i + 1} failed with error: {e}")
            if i == tolerance - 1:
                print("All dist attempts failed.")
                raise

    for i in range(tolerance):
        try:
            utils.train_sched_model(sched_model, sched_optimizer, sched_scheduler, criterion, train_dataloader, test_dataloader, num_weeks=num_weeks, num_epochs=num_epochs)
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







    