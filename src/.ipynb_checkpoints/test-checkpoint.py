import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

import util
import train


        
def save_result(exp_path, model_dict, dataloader):
    
    rows = 2
    cols = len(model_dict) // rows
    
    fig, my_plots = plt.subplots(rows, cols)

    fig.set_figheight(8)
    fig.set_figwidth(16)
    fig.tight_layout(pad=6)
    
    
    data = {"model": ["out_of_the_box"],
            "l2_loss": [],
            "l1_loss": [],
            "snr_db": []}
    
    l2 = 0
    l1 = 0
    snr_db = 0
    n_batches = len(dataloader)
    
    for batch, (clean, noisy) in enumerate(dataloader):
        
        if batch == 0:
            
            clean = clean.unsqueeze(dim=1)
            noisy = noisy.unsqueeze(dim=1)
            
            plot_clean = clean[0, 0, :]
            plot_noisy = noisy[0, 0, :]
            
        
        l2 += float(F.mse_loss(noisy.float(), clean.float()).data)
        l1 += float(F.l1_loss(noisy.float(), clean.float()).data)
        snr_db += float(util.snr_db(clean.cpu(), noisy.cpu()))
    
    
    # adding out of the box loss and snr
    data["l2_loss"].append(l2 / n_batches)
    data["l1_loss"].append(l1 / n_batches)
    data["snr_db"].append(snr_db / n_batches)
    
    
    curr_row = 0
    for i, (model_name, model) in enumerate(model_dict.items()): 
        
        
        l2 = 0
        l1 = 0
        snr_db = 0
        
        
        for batch, (clean, noisy) in enumerate(dataloader):
            
            clean = clean.unsqueeze(dim=1)
            noisy = noisy.unsqueeze(dim=1)
            
            curr_denoised = model(noisy.float())
            
            if batch == 0:
                plot_denoised = model(plot_noisy.view((1, 1, -1)).float())
                
                curr_loss = torch.nn.functional.mse_loss(plot_denoised.view(-1).float(), plot_clean.float())
                curr_plt = my_plots[curr_row][i % cols]
                
                axis = torch.arange(0, plot_clean.shape[-1], 1)
                
                curr_plt.plot(axis, plot_noisy.cpu(), label="noisy", color="lightsteelblue")
                curr_plt.plot(axis, plot_clean.cpu(), label="clean", color="yellowgreen")
                curr_plt.plot(axis, plot_denoised.view(-1).detach().cpu(), label=model_name, color="salmon")
                
                curr_plt.legend()
                curr_plt.axis(xmin=0, xmax=128)
                curr_plt.axis(ymin=-1, ymax=2)
                
                
                exp_data = str(exp_path.name).split("_")
                
                curr_plt.set_title("{}\ntested on {}, noise std {}\nL2 Loss {:.6f}".format(model_name, exp_data[0], exp_data[-1], curr_loss), fontsize=12)
            
            l2 += float(F.mse_loss(curr_denoised, clean.float()).data)
            l1 += float(F.l1_loss(curr_denoised, clean.float()).data)
            snr_db += float(util.snr_db(clean.cpu(), curr_denoised.cpu()))
        
        data["model"].append(model_name)
        data["l2_loss"].append(l2 / n_batches)
        data["l1_loss"].append(l1 / n_batches)
        data["snr_db"].append(snr_db / n_batches)
        
        if (i + 1) % cols == 0:
            curr_row += 1
    
    
    #-------------- DATA LOGGING ------------
    
    fig_path = exp_path / "result_figure.pdf"
    fig.savefig(fig_path)
    
    fig_path = exp_path / "result_figure.svg"
    fig.savefig(fig_path)
    
    csv_path = exp_path / "result_table.csv"
    df = pd.DataFrame(data)
    df.to_csv(csv_path)
    

    


