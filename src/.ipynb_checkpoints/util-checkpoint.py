import numpy as np
import torch

def add_noise(x: torch.tensor, mean=0, std=0.5):
    noise = torch.empty(x.shape).normal_(mean=mean, std=std)
    return x + noise


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(np.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe



def snr_db(clean, noisy):
    
    clean = np.abs(clean.detach())
    noisy = np.abs(noisy.detach())
    
    ratio = np.divide(clean, noisy + 1e-4) # avoidong zero division
    mean = torch.mean(ratio, dim=-1)
    mean = torch.mean(mean)
    res = 20 * np.log(mean)
    
    return res    
    

def lincomb_generate_data(batch_size, intervals, sample_length, functions, noise_dict, sample_offset, device)->torch.tensor:
    
    channels = 1
    mul_term = 2 * np.pi / sample_length
    
    random_offset = np.random.uniform(low=sample_offset[0], high=sample_offset[1], size=(batch_size, 1))
    random_offset = np.tile(random_offset, (1, sample_length * intervals))
    
    x_axis = np.linspace(0, 2 * np.pi * intervals, sample_length* intervals)

    X = np.tile(x_axis, (batch_size, 1)) + random_offset
    Y = np.expand_dims(X, axis=1)
    
    coef_lo  = -1
    coef_hi  =  1
    coef_mat = np.random.uniform(coef_lo, coef_hi, (batch_size, len(functions))) # creating a matrix of coefficients
    coef_mat = np.where(np.abs(coef_mat) < 10**-2, 0, coef_mat)

    for i in range(batch_size):
    
        curr_res = np.zeros((channels, sample_length * intervals))
        for func_id, function in enumerate(functions):
            curr_func = functions[func_id]
            curr_coef = coef_mat[i][func_id]
            curr_res += curr_coef * curr_func(Y[i, :, :])
            
        Y[i, :, :] = curr_res
        
    clean = Y
    
    # normalization
    clean -= clean.min(axis=2, keepdims=2)
    clean /= clean.max(axis=2, keepdims=2) + 1e-5 #avoiding zero division
    
    if noise_dict["type"] == "gaussian":
        noise_mean = noise_dict["mean"]
        noise_std  = noise_dict["std"]
        
        noise = np.random.normal(noise_mean, noise_std, Y.shape)
        
    if noise_dict["type"] == "uniform":
        noise_low  = noise_dict["low"]
        noise_high = noise_dict["high"]
        
        noise = np.random.uniform(noise_low, noise_high, Y.shape)
    
    
    noisy = clean + noise    
   
    clean = torch.from_numpy(clean).to(device)
    noisy = torch.from_numpy(noisy).to(device)
    
    return x_axis, clean, noisy
