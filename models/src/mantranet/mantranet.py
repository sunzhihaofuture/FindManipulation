import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .module import ConvLSTM, ConvGRU
from .module import symm_pad, batch_norm

CHECKPOINTS_ROOT = '/data/sunzhihao/season23/workspace/FindManipulation/models/src/mantranet/checkpoints'


# --------------------------------------------------
# IMAGE MANIPULATION TRACE FEATURE EXTRACTOR
# --------------------------------------------------
class IMTFE(nn.Module):
    def __init__(self, in_channel=3):
        super(IMTFE, self).__init__()

        self.relu = nn.ReLU()

        self.init_conv = nn.Conv2d(in_channel, 4, 5, 1, padding=0, bias=False)

        self.BayarConv2D = nn.Conv2d(in_channel, 3, 5, 1, padding=0, bias=False)
        self.bayar_mask = torch.tensor(np.ones(shape=(5, 5)))
        self.bayar_mask[2, 2] = 0

        self.bayar_final = torch.tensor(np.zeros((5, 5)))
        self.bayar_final[2, 2] = -1

        self.SRMConv2D = nn.Conv2d(in_channel, 9, 5, 1, padding=0, bias=False)
        self.SRMConv2D.weight.data=torch.load(os.path.join(CHECKPOINTS_ROOT, 'IMTFEv4.pt'))['SRMConv2D.weight']

        # SRM filters (fixed)
        for param in self.SRMConv2D.parameters():
            param.requires_grad = False

        self.middle_and_last_block = nn.ModuleList([
            nn.Conv2d(16, 32, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0)]
        )

    def forward(self, x: torch.Tensor):
        device = x.device
        self.bayar_mask = self.bayar_mask.to(device)
        self.bayar_final = self.bayar_final.to(device)
    
        # Normalization
        x = x / 255. * 2 - 1

        # Bayar constraints
        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        self.BayarConv2D.weight.data += self.bayar_final

        x = symm_pad(x, (2, 2, 2, 2))

        conv_init = self.init_conv(x)
        conv_bayar = self.BayarConv2D(x)
        conv_srm = self.SRMConv2D(x)

        first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
        first_block = self.relu(first_block)

        last_block = first_block

        for layer in self.middle_and_last_block:
            if isinstance(layer, nn.Conv2d):
                last_block = symm_pad(last_block, (1, 1, 1, 1))
            last_block = layer(last_block)

        # L2 normalization
        last_block = F.normalize(last_block, dim=1, p=2)

        return last_block


# --------------------
# ANOMALY DETECTOR
# --------------------
class AnomalyDetector(nn.Module):
    def __init__(self, eps=10**(-6), with_GRU=False):
        super(AnomalyDetector, self).__init__()

        self.eps = eps
        self.relu = nn.ReLU()
        self.with_GRU=with_GRU

        self.adaptation = nn.Conv2d(256, 64, 1, 1, padding=0, bias=False)

        self.sigma_F = nn.Parameter(torch.zeros((1, 64, 1, 1)), requires_grad=True)

        self.pool31 = nn.AvgPool2d(31, stride=1, padding=15, count_include_pad=False)
        self.pool15 = nn.AvgPool2d(15, stride=1, padding=7, count_include_pad=False)
        self.pool7 = nn.AvgPool2d(7, stride=1, padding=3, count_include_pad=False)

        if not(self.with_GRU):
            self.conv_lstm =ConvLSTM(input_dim=64,
                                    hidden_dim=8,
                                    kernel_size=(7, 7),
                                    num_layers=1,
                                    batch_first=False,
                                    bias=True,
                                    return_all_layers=False)
        else:
            self.conv_gru=ConvGRU(input_dim=64,
                                 hidden_dim=8,
                                 kernel_size=(7, 7),
                                 num_layers=1,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)

        self.end = nn.Sequential(nn.Conv2d(8, 1, 7, 1, padding=3),nn.Sigmoid())
        
    def forward(self, IMTFE_output):
        _, _, H, W = IMTFE_output.shape
        
        if not(self.training):
            self.GlobalPool = nn.AvgPool2d((H, W), stride=1)
        else:
            if not hasattr(self, 'GlobalPool'):
                self.GlobalPool = nn.AvgPool2d((H, W), stride=1)

        # Local anomaly feature extraction
        X_adapt = self.adaptation(IMTFE_output)
        X_adapt = batch_norm(X_adapt)

        # Z-pool concatenation
        mu_T = self.GlobalPool(X_adapt)
        sigma_T = torch.sqrt(self.GlobalPool(torch.square(X_adapt - mu_T)))
        sigma_T = torch.max(sigma_T, self.sigma_F + self.eps)
        inv_sigma_T = torch.pow(sigma_T, -1)
        zpoolglobal = torch.abs((mu_T - X_adapt) * inv_sigma_T)

        mu_31 = self.pool31(X_adapt)
        zpool31 = torch.abs((mu_31 - X_adapt) * inv_sigma_T)

        mu_15 = self.pool15(X_adapt)
        zpool15 = torch.abs((mu_15 - X_adapt) * inv_sigma_T)

        mu_7 = self.pool7(X_adapt)
        zpool7 = torch.abs((mu_7 - X_adapt) * inv_sigma_T)

        input_rnn = torch.cat([zpool7.unsqueeze(0), zpool15.unsqueeze(0), zpool31.unsqueeze(0), zpoolglobal.unsqueeze(0)], axis=0)

        if not(self.with_GRU):
            # Conv2DLSTM
            _, output_lstm = self.conv_lstm(input_rnn)
            output_lstm = output_lstm[0][0]

            final_output = self.end(output_lstm)

        else:
            # Conv2DLSTM
            _, output_gru = self.conv_gru(input_rnn)
            output_gru = output_gru[0]

            final_output = self.end(output_gru)

        return final_output


class MantraNet(nn.Module):
    def __init__(self, in_channel=3, eps=10**(-6), with_GRU=False):
        super(MantraNet, self).__init__()

        self.eps = eps
        self.relu = nn.ReLU()

        self.IMTFE=IMTFE(in_channel=in_channel)
        self.AnomalyDetector=AnomalyDetector(eps=eps, with_GRU=with_GRU)
        
    def forward(self, x):
        return self.AnomalyDetector(self.IMTFE(x))
            

def get_mantranet(pretrained=True, checkpoint_path=os.path.join(CHECKPOINTS_ROOT, 'MantraNetv4.pt')):
    model = MantraNet()
    if pretrained:
        model.load_state_dict(torch.load(checkpoint_path))
    return model
