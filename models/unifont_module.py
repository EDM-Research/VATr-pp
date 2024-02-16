import math
import os

import cv2
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np


def gauss(x, sigma=1.0):
    return (1.0 / math.sqrt(2.0 * math.pi) * sigma) * math.exp(-x**2 / (2.0 * sigma**2))


class UnifontModule(torch.nn.Module):
    def __init__(self, out_dim, alphabet, device='cuda', input_type='unifont', projection='linear'):
        super(UnifontModule, self).__init__()
        self.projection_type = projection
        self.device = device
        self.alphabet = alphabet
        self.symbols = self.get_symbols('unifont')
        self.symbols_repr = self.get_symbols(input_type)

        if projection == 'linear':
            self.linear = torch.nn.Linear(self.symbols_repr.shape[1], out_dim)
        else:
            self.linear = torch.nn.Identity()

    def get_symbols(self, input_type):
        with open(f"files/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        all_symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        symbols = []
        for char in self.alphabet:
            im = all_symbols[ord(char)]
            im = im.flatten()
            symbols.append(im)

        symbols.insert(0, np.zeros_like(symbols[0]))
        symbols = np.stack(symbols)
        return torch.from_numpy(symbols).float().to(self.device)

    def forward(self, QR):
        if self.projection_type != 'cnn':
            return self.linear(self.symbols_repr[QR])
        else:
            result = []
            symbols = self.symbols_repr[QR]
            for b in range(QR.size(0)):
                result.append(self.linear(torch.unsqueeze(symbols[b], dim=1)))

            return torch.stack(result)


class LearnableModule(torch.nn.Module):
    def __init__(self, out_dim, device='cuda'):
        super(LearnableModule, self).__init__()
        self.device = device
        self.param = torch.nn.Parameter(torch.zeros(1, 1, 256, device=device))
        self.linear = torch.nn.Linear(256, out_dim)

    def forward(self, QR):
        return self.linear(self.param).repeat((QR.shape[0], 1, 1))


if __name__ == "__main__":
    module = UnifontModule(512, "bluuuuurp", 'cpu', projection='cnn')