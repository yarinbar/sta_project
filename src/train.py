import torch
from torch.utils.data import Dataset, DataLoader
import util
from tqdm import tqdm

def train_session(model_dict, dataloader, num_epochs, lr, momentum, device, verbose=False, losses=None, snr=None):
    
    n = len(model_dict)
    
    criterion = torch.nn.MSELoss()    
    
    sgd_params = {
        "lr": lr,
        "momentum": momentum
    }
    
    optimizer_dict = {}
    train_losses, snr_db = {}, {}
    
    for model_name, model in model_dict.items():
        
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
        optimizer_dict[model_name] = optimizer
        
        if losses is None:
            train_losses[model_name] = []
        else:
            train_losses[model_name] = losses[model_name]
            
        if snr is None:
            snr_db[model_name] = []
        else:
            snr_db[model_name] = snr[model_name]    
        
    
    
    if verbose:
        print("STARTED TRAINING")
        
    for epoch in tqdm(range(num_epochs)):
        
        for t_clean, t_noisy in dataloader:
            
            t_clean = t_clean.unsqueeze(dim=1)
            t_noisy = t_noisy.unsqueeze(dim=1)
            
            for model_name, model in model_dict.items():

                # ===================forward=====================
                output = model(t_noisy.float())

                loss = criterion(output.float(), t_clean.float())

                train_losses[model_name].append(loss.data)
                
                # ===================backward====================
                optimizer_dict[model_name].zero_grad()
                loss.backward()
                optimizer_dict[model_name].step()
        
        
        if verbose and (epoch == 0 or (epoch + 1) % (num_epochs // 5) == 0):
            print("Current epoch {}".format(epoch))
            for model_name, model in model_dict.items():
                print("{0:25}".format(model_name), end='')
                print("train loss:  {:.6f}".format(train_losses[model_name][-1]))

            print()
            
    if verbose:       
        print("DONE TRAINING")
    
    return train_losses, snr_db

