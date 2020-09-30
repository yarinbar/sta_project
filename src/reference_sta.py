import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        x = x + self.pe.squeeze(1).repeat(x.shape[0], x.shape[1], 1, 1)

        return self.dropout(x)


class TCNActivationBlock(nn.Module):
    """

    """

    def __init__(self, bn_channels: int, activ_type: str = 'relu', lrelu_alpha: float = 0.2,
                 dropout_rate: float = 0.4):
        """

        :param bn_channels:
        :param activ_type:
        :param lrelu_alpha:
        :param dropout_rate:
        """

        super(TCNActivationBlock, self).__init__()

        self.dropout_rate = dropout_rate

        self.bn = nn.BatchNorm2d(bn_channels)

        if activ_type == 'lrelu':
            self.activation = nn.LeakyReLU(lrelu_alpha)

        elif activ_type == 'elu':
            self.activation = nn.ELU()

        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        """

        :param x:
        :return:
        """

        out = self.bn(x)
        out = self.activation(out)
        out = F.dropout(input=out, p=self.dropout_rate, training=True, inplace=False)

        return out


class SpectroTemporalAttention(nn.Module):
    """

    """

    def __init__(self, input_shape: [int, int, int], out_channels: int, dropout_rate: float = 0.3):

        """

        :param input_shape:
        :param out_channels:
        """

        super(SpectroTemporalAttention, self).__init__()

        self.input_shape = input_shape
        self.k = out_channels
        self.dropout_rate = dropout_rate
        self.softmax = nn.Softmax(dim=1)

        self.ecg_attention_channels = self._get_attention_layers(input_channels=(input_shape[0] // 2))
        self.fft_attention_channels = self._get_attention_layers(input_channels=(input_shape[0] // 2))

        self.ecg_fc = nn.Linear(in_features=input_shape[1], out_features=input_shape[1], bias=True)
        nn.init.kaiming_normal_(self.ecg_fc._parameters['weight'])

        self.fft_fc = nn.Linear(in_features=input_shape[1], out_features=input_shape[1], bias=True)
        nn.init.kaiming_normal_(self.fft_fc._parameters['weight'])

        self.ecg_conv = self.TCNConvBlock(in_channels=input_shape[0], out_channels=input_shape[0],
                                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), bias=True,
                                          activation_type='relu', dropout_rate=dropout_rate)

        self.fft_conv = self.TCNConvBlock(in_channels=input_shape[0], out_channels=input_shape[0],
                                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1),
                                          bias=True, activation_type='relu', dropout_rate=dropout_rate)

        self.fft_positional_encoding = ECGPositionalEncoding(features_dimensionality=input_shape[1],
                                                             num_of_beats=input_shape[2],
                                                             dropout_rate=dropout_rate)

        self.merge_conv = self.TCNConvBlock(in_channels=(input_shape[0] * 2), out_channels=out_channels,
                                            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1),
                                            bias=True, activation_type='relu', dropout_rate=dropout_rate)

    @staticmethod
    def _get_attention_layers(input_channels: int = 3):
        """

        :param input_channels:
        :return:
        """

        attention_channels = []
        for i in range(input_channels):
            attention_channels.append(nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1,
                                                stride=1, padding=0, dilation=1, bias=False))
            nn.init.kaiming_normal_(attention_channels[-1]._parameters['weight'])

        return nn.ModuleList(attention_channels)

    @staticmethod
    def TCNConvBlock(in_channels: int, out_channels: int, kernel_size: (int, int) = (3, 3),
                     stride: (int, int) = (1, 1), padding: (int, int) = (0, 0), dilation: (int, int) = (1, 1),
                     bias: bool = True, activation_type: str = 'relu', leaky_relu_alpha: float = 0.2,
                     dropout_rate: float = 0.4):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param bias:
        :param activation_type:
        :param leaky_relu_alpha:
        :param dropout_rate:
        :return:
        """

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(conv._parameters['weight'])

        activation = TCNActivationBlock(bn_channels=out_channels, activ_type=activation_type,
                                        lrelu_alpha=leaky_relu_alpha, dropout_rate=dropout_rate)

        return nn.Sequential(conv, activation)

    def forward(self, x: torch.Tensor):
        """

        :param x:
        :return:
        """

        batch_size, channels, h, w = x.size()
        channels_per_modality = channels // 2

        ecg = x[:, 0::2, :, :]
        fft = x[:, 1::2, :, :]

        # Convolve inputs with 1 features map per channel
        ecg_embeddings = []
        fft_embeddings = []
        
        for i in range((self.input_shape[0] // 2)):
            out_ecg = self.ecg_attention_channels[i](fft)
            out_ecg = out_ecg.view(batch_size, (h * w))
            ecg_embeddings.append(out_ecg)

            out_fft = self.fft_attention_channels[i](ecg)
            out_fft = out_fft.view(batch_size, (h * w))
            fft_embeddings.append(out_fft)

        # [out] = [BatchSize, ((Channels / 2) * h * w)]
        #print(ecg_embeddings)
        out_ecg = torch.stack(ecg_embeddings, dim=0).view(batch_size, (channels_per_modality * h * w)).t().contiguous().view(batch_size, (channels_per_modality * h * w))
        out_fft = torch.stack(fft_embeddings, dim=0).view(batch_size, (channels_per_modality * h * w)).t().contiguous().view(batch_size, (channels_per_modality * h * w))

        # [out] = [BatchSize, (Channels / 2), (h * w)]
        out_ecg = out_ecg.view(batch_size, channels_per_modality, (h * w))
        out_fft = out_fft.view(batch_size, channels_per_modality, (h * w))

        # [out_] = [BatchSize, (h * w), (h * w)]
        out_ecg_ = torch.bmm(out_ecg.transpose(1, 2), out_ecg)
        out_fft_ = torch.bmm(out_fft.transpose(1, 2), out_fft)

        # [out_] = [BatchSize, Softmax((h * w), (h * w))]
        out_ecg_ = out_ecg_.view((batch_size * (h * w)), (h * w))
        out_ecg_ = self.softmax(out_ecg_)
        out_ecg_ = out_ecg_.view(batch_size, (h * w), (h * w))

        out_fft_ = out_fft_.view((batch_size * (h * w)), (h * w))
        out_fft_ = self.softmax(out_fft_)
        out_fft_ = out_fft_.view(batch_size, (h * w), (h * w))

        # [attention_mask] = [BatchSize, (Channels / 2), (h * w)]
        ecg_attention_mask = torch.bmm(out_ecg, out_ecg_)
        fft_attention_mask = torch.bmm(out_fft, out_fft_)

        # [attention_mask] = [BatchSize, (Channels / 2), h, w]
        ecg_attention_mask = ecg_attention_mask.view(batch_size, (channels // 2), h, w)
        fft_attention_mask = fft_attention_mask.view(batch_size, (channels // 2), h, w)

        # FC per row
        ecg_attention_mask = ecg_attention_mask.permute(0, 1, 3, 2)
        ecg_attention_mask = ecg_attention_mask.reshape((batch_size * channels_per_modality * w), h)
        ecg_attention_mask = self.ecg_fc(ecg_attention_mask)
        ecg_attention_mask = ecg_attention_mask.reshape(batch_size, channels_per_modality, w, h)
        ecg_attention_mask = ecg_attention_mask.permute(0, 1, 3, 2)

        fft_attention_mask = fft_attention_mask.permute(0, 1, 3, 2)
        fft_attention_mask = fft_attention_mask.reshape((batch_size * channels_per_modality * w), h)
        fft_attention_mask = self.fft_fc(fft_attention_mask)
        fft_attention_mask = fft_attention_mask.reshape(batch_size, channels_per_modality, w, h)
        fft_attention_mask = fft_attention_mask.permute(0, 1, 3, 2)

        # Save the complete attention_mask
        attention_mask = torch.cat((ecg_attention_mask, fft_attention_mask), 1)

        # Apply the attention masks
        out_ecg = ecg_attention_mask * ecg
        out_fft = fft_attention_mask * fft

        ecg = torch.cat((ecg, out_ecg), 1)
        fft = torch.cat((fft, out_fft), 1)

        # First Layers
        ecg = self.ecg_conv(ecg)
        fft = self.fft_positional_encoding(fft)
        fft = self.fft_conv(fft)

        out = torch.cat((ecg, fft), 1)
        out = self.merge_conv(out)

        return out, attention_mask

