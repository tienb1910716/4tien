import torch.nn as nn
import torch

class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        self.rnn = nn.LSTM(inp, hidden, bidirectional=True) if lstm else nn.GRU(inp, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden * 2, out)

    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)
        return out

import torchvision.models as models

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.imgH = imgH
        self.nc = nc
        self.nclass = nclass
        self.nh = nh

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H/8
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H/16
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU()  # h = 1
        )

        # RNN layers
        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True, num_layers=2),
            nn.LSTM(nh * 2, nh, bidirectional=True, num_layers=2)
        )

        # Điều chỉnh fc để khớp với H = 104
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"Chiều cao phải bằng 1, got {h}"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        output, _ = self.rnn(conv)
        output = self.fc(output)
        return output
