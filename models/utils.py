from sklearn.model_selection import train_test_split
import json
import pandas as pd
from CaptchaDataset.dataset import CaptchaDataset, custom_collate_fn, transform
from torch.utils.data import DataLoader




from CaptchaDataset.dataset import HandwritingDataset, custom_collate_fn, transform
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import editdistance
class strLabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'  # Thêm blank
        self.char2idx = {c: i for i, c in enumerate(self.alphabet)}
    
    def encode(self, texts):
        lengths = [len(t) for t in texts]
        max_length = max(lengths)
        encoded = torch.zeros(len(texts), max_length).long()
        for i, t in enumerate(texts):
            for j, c in enumerate(t):
                encoded[i, j] = self.char2idx[c]
        return encoded, torch.IntTensor(lengths)
    
    def decode(self, preds, preds_size):
        texts = []
        for pred, size in zip(preds, preds_size):
            _, pred = pred[:size].max(1)
            text = ''
            for p in pred:
                if p != len(self.alphabet) - 1:  # Bỏ blank
                    text += self.alphabet[p]
            texts.append(text)
        return texts

class CTCLoss(nn.Module):
    def __init__(self):
        super(CTCLoss, self).__init__()
        self.loss = nn.CTCLoss(zero_infinity=True)
    
    def forward(self, preds, labels, preds_size, label_lengths):
        return self.loss(preds, labels, preds_size, label_lengths)

def cer(pred, gt):
    return editdistance.eval(pred, gt) / max(len(gt), 1)

def wer(pred, gt):
    pred_words = pred.split()
    gt_words = gt.split()
    return editdistance.eval(pred_words, gt_words) / max(len(gt_words), 1)



