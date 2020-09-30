
import torch
import torch.nn.functional as F
import torch.nn as nn
import custom_layers
from reference_sta import SpectroTemporalAttention
import numpy as np
from scipy import signal


class LinAutoencoder(nn.Module):
    def __init__(self, in_channels, K, B, z_dim, out_channels, device, fft=False):
        super(LinAutoencoder, self).__init__()

        if fft:
            in_channels *= 2 # spectral channel
            
        self.in_channels = in_channels
        self.K = K
        self.B = B
        self.out_channels = out_channels
        
        self.device = device
        
        self.fft = fft
        
            

        encoder_layers = []
        decoder_layers = []
        
        encoder_layers += [
            nn.Linear(in_channels * K * B, 2 * in_channels * K * B, bias=True),
            nn.Linear(2 * in_channels * K * B, in_channels * K * B, bias=True),
            nn.Linear(in_channels * K * B, 2 * z_dim, bias=True),
            nn.Linear(2 * z_dim, z_dim, bias=True),
        ]
        

        decoder_layers += [          
            nn.Linear(z_dim, 2 * z_dim, bias=True),
            nn.Linear(2 * z_dim, in_channels * K * B, bias=True),
            nn.Linear(in_channels * K * B, 2 * in_channels * K * B, bias=True),
            nn.Linear(2 * in_channels * K * B, out_channels * K * B, bias=True),
        ]


        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        

    def forward(self, x):
        
        batch_size  = x.shape[0]
        
        # adds the spectral info
        if self.fft:
            spectral = torch.from_numpy(signal.welch(x.cpu(), axis=-1, return_onesided=False, nperseg=self.K * self.B)[1]).to(self.device).float()
            x = torch.cat((x, spectral), dim=1)
        
        x_flat = x.view((batch_size, -1))
        enc = self.encoder(x_flat)
        dec = self.decoder(enc)
        res = dec.view((batch_size, self.out_channels, self.K * self.B))
        return res

    
class NonLinAutoencoder(nn.Module):
    def __init__(self, in_channels, K, B, z_dim, out_channels, device, fft=False):
        super(NonLinAutoencoder, self).__init__()
        
        if fft:
            in_channels *= 2 # spectral channel

        self.in_channels = in_channels
        self.K = K
        self.B = B
        self.out_channels = out_channels
        
        self.device = device
        
        self.fft = fft

        encoder_layers = []
        decoder_layers = []
        
        encoder_layers += [
            nn.Linear(in_channels * K * B, 2 * in_channels * K * B, bias=True),
            nn.Tanh(),
            nn.Linear(2 * in_channels * K * B, in_channels * K * B, bias=True),
            nn.Tanh(),
            nn.Linear(in_channels * K * B, 2 * z_dim, bias=True),
            nn.Tanh(),
            nn.Linear(2 * z_dim, z_dim, bias=True),
            nn.Tanh(),
        ]
        

        decoder_layers += [          
            nn.Linear(z_dim, 2 * z_dim, bias=True),
            nn.Tanh(),
            nn.Linear(2 * z_dim, in_channels * K * B, bias=True),
            nn.Tanh(),
            nn.Linear(in_channels * K * B, 2 * in_channels * K * B, bias=True),
            nn.Tanh(),
            nn.Linear(2 * in_channels * K * B, out_channels * K * B, bias=True),
            nn.Tanh(),
        ]


        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        

    def forward(self, x):
        
        batch_size  = x.shape[0]
        
        # adds the spectral info
        if self.fft:
            spectral = torch.from_numpy(signal.welch(x.cpu(), axis=-1, return_onesided=False, nperseg=self.K * self.B)[1]).to(self.device).float()
            x = torch.cat((x, spectral), dim=1)
        
        x_flat = x.view((batch_size, -1))
        enc = self.encoder(x_flat)
        dec = self.decoder(enc)
        res = dec.view((batch_size, self.out_channels, self.K * self.B))
        return res


class DenoiserSTA(nn.Module):
    def __init__(self, L, K, B, z_dim, device, linear=True):
        super(DenoiserSTA, self).__init__()

        self.L = L
        self.K = K
        self.B = B
        
        self.device = device

        self.sta = custom_layers.STA(L, K, B, device)
        
        if linear:
            self.autoencoder = LinAutoencoder(4*L, K, B, z_dim, L, device)
        else:
            self.autoencoder = NonLinAutoencoder(4*L, K, B, z_dim, L, device) # with non linear activation

    def forward(self, x):
        temporal = x
        spectral = torch.from_numpy(signal.welch(temporal.cpu(), axis=-1, return_onesided=False, nperseg=self.K * self.B)[1]).to(self.device).float()
        tmp = self.sta(temporal.float(), spectral.float())
        res = self.autoencoder(tmp)

        return res


class RefDenoiserSTA(nn.Module):
    def __init__(self, L, K, B, z_dim, device, linear=True):
        super(RefDenoiserSTA, self).__init__()

        self.L = L
        self.K = K
        self.B = B
        
        self.device = device

        self.sta = SpectroTemporalAttention((2*L, B, K), L)
        
        if linear:
            self.autoencoder = LinAutoencoder(L, K, B, z_dim, L, device)
        else:
            self.autoencoder = NonLinAutoencoder(L, K, B, z_dim, L, device) # with non linear activation
            
        self.attn = None

    def forward(self, x):
        batch_size, channels, sample_len = x.size()
        
        fft = torch.from_numpy(signal.welch(x.cpu(), axis=-1, return_onesided=False, nperseg=self.K * self.B)[1]).to(self.device).float()
        x = torch.stack((x, fft), dim=1)

        x = x.view(batch_size, 2*channels, self.B, self.K)
        
        tmp, attn = self.sta(x.float())
        self.attn = attn
        res = self.autoencoder(tmp)

        return res    
    