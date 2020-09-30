
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import util

__author__ = 'Yarin Bar'


class ECGPositionalEncoding(nn.Module):
    """

    """

    def __init__(self, features_dimensionality: int, num_of_beats: int = 60, dropout_rate: float = 0.1):
        """

        :param features_dimensionality:
        :param num_of_beats:
        :param dropout_rate:
        """

        super(ECGPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(features_dimensionality, num_of_beats)
        position = torch.arange(0, features_dimensionality, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_of_beats, 2).float() * (-math.log(10000.0) /
                                                                         num_of_beats))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        pe = self.pe.transpose(0, 2).reshape((x.shape[-1]))
        x = x + pe.repeat(x.shape[0], 2, 1)
        
        return self.dropout(x)
    
    

class STA(nn.Module):
    def __init__(self, L: int, K: int, B: int, device):
        super(STA, self).__init__()

        self.L = L
        self.K = K
        self.B = B
        
        self.device = device

        KB = K * B
        LB = L * B
        
        

#         self.temporal_transform = nn.Linear(KB, KB, bias=True)
#         self.spectral_transform = nn.Linear(KB, KB, bias=True)
        
        self.temporal_transform = self.embedding()
        self.spectral_transform = self.embedding()
    
        self.temporal_mask = nn.Linear(LB, LB, bias=True)
        self.spectral_mask = nn.Linear(LB, LB, bias=True)

        self.temporal_conv = nn.Conv1d(2 * self.L, 2 * self.L, kernel_size=3, padding=1)
        self.spectral_conv = nn.Conv1d(2 * self.L, 2 * self.L, kernel_size=3, padding=1)
        
        self.spectral_pe = ECGPositionalEncoding(features_dimensionality=K, num_of_beats=B)
        
        self.last_temporal_conv = nn.Conv1d(2 * self.L, 2 * self.L, kernel_size=3, padding=1)
        self.last_spectral_conv = nn.Conv1d(2 * self.L, 2 * self.L, kernel_size=3, padding=1)
        
        self.attn_dict = {}
        
    def embedding(self):
        layers = []
        
        layers = [
            nn.Conv1d(1, 8, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
            nn.Conv1d(8, 8, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        ]
        
        return nn.Sequential(*layers)
        
        
    def forward(self, temporal, spectral):

        # shape - [batch, channels, 1, sample_length]
        assert temporal.shape == spectral.shape
        batch_size = temporal.shape[0]
        
        Xe = self.temporal_transform(spectral)
        Se = self.spectral_transform(temporal)

        # Softmax should happen on the first dimension of each sample, not of the bathes, i.e. dim=1
        XeTXe = (Xe.squeeze(2).permute(0, 2, 1).matmul(Xe.squeeze(2)))
        Ax = torch.softmax(XeTXe, 1)
        SeTSe = (Se.squeeze(2).permute(0, 2, 1).matmul(Se.squeeze(2)))
        As = torch.softmax(SeTSe, 1)

        Xa = torch.matmul(Xe, Ax)
        Sa = torch.matmul(Se, As)

        # ------ Make sure that the reshaping keeps the original elements in the same orderin
        Xa_tag = Xa.transpose(2,1).reshape((batch_size, self.K, self.L * self.B))
        Sa_tag = Sa.transpose(2,1).reshape((batch_size, self.K, self.L * self.B))

        Mx_tag = self.temporal_mask(Xa_tag)
        Ms_tag = self.spectral_mask(Sa_tag)

        Mx = Mx_tag.view((batch_size, self.L, self.K * self.B))
        Ms = Ms_tag.view((batch_size, self.L, self.K * self.B))

        Xm = torch.mul(Mx, temporal)
        Sm = torch.mul(Ms, spectral)

        X_hat = torch.cat((Xm, temporal), dim=1)
        S_hat = torch.cat((Sm, spectral), dim=1)

        Xe_hat = self.temporal_conv(X_hat)
        Se_hat = self.spectral_pe(self.spectral_conv(S_hat))
        
        
        Xe_hat = self.last_temporal_conv(Xe_hat)
        Se_hat = self.last_spectral_conv(Se_hat)
        
        res = torch.cat((Xe_hat, Se_hat), dim=1)
        
        self.attn_dict["Ax"] = Ax
        self.attn_dict["As"] = As
        self.attn_dict["Xa"] = Xa
        self.attn_dict["Xs"] = Sa
        self.attn_dict["Mx"] = Mx
        self.attn_dict["Ms"] = Ms

        """
        print("Xe.shape {}".format(Xe.shape))
        print("Se.shape {}".format(Se.shape))
        print("Ax.shape {}".format(Ax.shape))
        print("As.shape {}".format(As.shape))
        print("Xa.shape {}".format(Xa.shape))
        print("Sa.shape {}".format(Sa.shape))
        print("attempting reshaping K={}, L={}, B={}".format(self.K, self.L, self.B))
        print("Xa_tag.shape {}".format(Xa_tag.shape))
        print("Sa_tag.shape {}".format(Sa_tag.shape))
        print("Mx_tag.shape {}".format(Mx_tag.shape))
        print("Ms_tag.shape {}".format(Ms_tag.shape))
        print("Mx.shape {}".format(Mx.shape))
        print("Ms.shape {}".format(Ms.shape))
        print("Xm.shape {}".format(Xm.shape))
        print("Sm.shape {}".format(Sm.shape))
        print("X_hat.shape {}".format(X_hat.shape))
        print("S_hat.shape {}".format(S_hat.shape))
        print("Xe_hat.shape {}".format(Xe_hat.shape))
        print("Se_hat.shape {}".format(Se_hat.shape))
        print("T.shape {}".format(T.shape))
        """

        return res

