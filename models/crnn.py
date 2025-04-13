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
    def __init__(self, in_channels, output):
        super(CRNN, self).__init__()

        # CNN Layers with reduced number of channels
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, stride=1, padding=1),  # 3x3 conv 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # 3x3 conv 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # 3x3 conv 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        
        # Sau 3 lần MaxPool2d với kernel (2, 2), chiều cao và chiều rộng sẽ giảm đi một nửa mỗi lần
        # Giả sử kích thước ảnh đầu vào là 128x128, chiều cao sẽ giảm còn 16 (128 / 2^3), chiều rộng tương tự.
        # input_size = channels * height, so input_size = 32 * 16 = 512
        self.rnn = nn.LSTM(input_size=32 * 13, hidden_size=1024, num_layers=2, batch_first=False, bidirectional=True)

        # Output Layer
        self.output_layer = nn.Linear(1024 * 2, output + 1)  # +1 for the blank token in CTC

    def forward(self, X, y=None, criterion=None):
        # Áp dụng các lớp CNN
        out = self.cnn(X)
        batch_size, channels, height, width = out.size()
        
        # Reshape đầu ra của CNN để đưa vào RNN: (width, batch_size, channels * height)
        out = out.permute(3, 0, 2, 1).reshape(width, batch_size, channels * height)
        
        # Áp dụng các lớp RNN
        out, _ = self.rnn(out)
        
        # Áp dụng lớp output
        out = self.output_layer(out)

        # Nếu có nhãn (y), tính toán độ mất mát CTC
        if y is not None:
            T = out.size(0)  # độ dài chuỗi (width)
            N = out.size(1)  # kích thước batch
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int64, device=out.device)
            target_lengths = torch.tensor([len(label[label > 0]) for label in y], dtype=torch.int64, device=out.device)
            loss = criterion(out, y, input_lengths, target_lengths)
            return out, loss
        
        return out, None