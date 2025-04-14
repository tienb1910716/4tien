import torch, json
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from CaptchaDataset.dataset import HandwritingDataset, custom_collate_fn
from models.crnn import CRNN
from models.utils import strLabelConverter, CTCLoss, cer, wer
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pandas as pd

# Mở và đọc nội dung file JSON chứa nhãn (labels.json).
label_file = 'data/captcha-version-2-images/labels.json'

try:
    with open(label_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        if not content.strip():  # Kiểm tra xem file có rỗng không
            print("File is empty.")
        else:
            labels_data = json.loads(content)  # Dùng json.loads thay vì json.load
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Định nghĩa các ký tự cho ánh xạ
all_letters = " !$%&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}ÀÁÂÃÈÉÊÌÍÐÒÓÔÕÖÙÚÜÝàáâãèéêìíðòóôõöùúüýĀāĂăĐđĨĩŌōŨũŪūƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ–—’“”…™−"
mapping = {ch: idx + 1 for idx, ch in enumerate(all_letters)}
mapping_inv = {idx + 1: ch for idx, ch in enumerate(all_letters)}
num_class = len(mapping)

# Tạo dataframe từ dữ liệu nhãn
images = []
labels = []
for filename, text in labels_data.items():
    images.append(filename)
    labels.append([mapping[char] for char in text if char in mapping])


# Tham số
imgH = 104
nc = 1
nh = 256
batch_size = 16
epochs = 50
lr = 0.001
val_split = 0.2  # Tỷ lệ validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
transform = transforms.Compose([
    transforms.Resize((imgH, 1853)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#dataset = HandwritingDataset('data/captcha-version-2-images/samples2', 'data/labels.txt', transform=transform)
df = pd.DataFrame({'images': images, 'label': labels})

# Chia dữ liệu thành 80% train, 10% validation và 10% test
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))
test_size = len(df) - train_size - val_size

# Đầu tiên chia thành 80% train và 20% (validation + test)
df_train, df_temp = train_test_split(df, test_size=val_size + test_size, shuffle=True)

# Sau đó chia 20% còn lại thành 50% cho validation và 50% cho test
df_val, df_test = train_test_split(df_temp, test_size=test_size / (val_size + test_size), shuffle=True)
# Kiểm tra DataFrame sau khi chia
print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")
if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
    print("One of the splits is empty. Please check data splitting.")
    exit()
# Tạo dataset và dataloader
train_data = HandwritingDataset(df_train, root_dir='data/captcha-version-2-images/samples2', transform=transform)
val_data = HandwritingDataset(df_val, root_dir='data/captcha-version-2-images/samples2', transform=transform)
test_data = HandwritingDataset(df_test, root_dir='data/captcha-version-2-images/samples2', transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_data, batch_size=4, collate_fn=custom_collate_fn)

# Mô hình
alphabet = open('models/alphabet.txt', 'r', encoding='utf-8').read().strip()
converter = strLabelConverter(alphabet)
nclass = len(alphabet) + 1
model = CRNN(imgH, nc, nclass, nh).to(device)
criterion = CTCLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# Lưu lịch sử
train_losses = []
val_losses = []
train_cers = []
train_wers = []
val_cers = []
val_wers = []

# Huấn luyện
for epoch in range(epochs):
    # Train
    model.train()
    train_loss = 0
    train_cer = 0
    train_wer = 0
    train_batches = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        batch_size = images.size(0)
        
        # Mã hóa nhãn
        labels_encoded, lengths = converter.encode(labels)
        labels_encoded = labels_encoded.to(device)
        
        optimizer.zero_grad()
        preds = model(images)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size).to(device)
        loss = criterion(preds, labels_encoded, preds_size, lengths)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Tính CER/WER
        preds = preds.log_softmax(2)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        pred_texts = converter.decode(preds, preds_size)
        for pred, gt in zip(pred_texts, labels):
            train_cer += cer(pred, gt)
            train_wer += wer(pred, gt)
        
        train_batches += 1
    
    train_loss /= train_batches
    train_cer /= (train_batches * batch_size)
    train_wer /= (train_batches * batch_size)
    
    # Validation
    model.eval()
    val_loss = 0
    val_cer = 0
    val_wer = 0
    val_batches = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            labels_encoded, lengths = converter.encode(labels)
            labels_encoded = labels_encoded.to(device)
            
            preds = model(images)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size).to(device)
            loss = criterion(preds, labels_encoded, preds_size, lengths)
            
            val_loss += loss.item()
            
            # Tính CER/WER
            preds = preds.log_softmax(2)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            pred_texts = converter.decode(preds, preds_size)
            for pred, gt in zip(pred_texts, labels):
                val_cer += cer(pred, gt)
                val_wer += wer(pred, gt)
            
            val_batches += 1
    
    val_loss /= val_batches
    val_cer /= (val_batches * batch_size)
    val_wer /= (val_batches * batch_size)
    
    # Lưu lịch sử
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_cers.append(train_cer)
    train_wers.append(train_wer)
    val_cers.append(val_cer)
    val_wers.append(val_wer)
    
    # In kết quả sau mỗi epoch
    print(f'Epoch [{epoch+1}/{epochs}]')
    print(f'Train Loss: {train_loss:.4f}, CER: {train_cer:.4f}, WER: {train_wer:.4f}')
    print(f'Val Loss: {val_loss:.4f}, CER: {val_cer:.4f}, WER: {val_wer:.4f}')

# In kết quả cuối cùng
print('\nFinal Results:')
print(f'Train Loss: {train_losses[-1]:.4f}, CER: {train_cers[-1]:.4f}, WER: {train_wers[-1]:.4f}')
print(f'Val Loss: {val_losses[-1]:.4f}, CER: {val_cers[-1]:.4f}, WER: {val_wers[-1]:.4f}')

# Vẽ biểu đồ
plt.figure(figsize=(15, 10))

# Loss
plt.subplot(3, 1, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# CER
plt.subplot(3, 1, 2)
plt.plot(train_cers, label='Train CER')
plt.plot(val_cers, label='Val CER')
plt.xlabel('Epoch')
plt.ylabel('CER')
plt.legend()
plt.title('Training and Validation CER')

# WER
plt.subplot(3, 1, 3)
plt.plot(train_wers, label='Train WER')
plt.plot(val_wers, label='Val WER')
plt.xlabel('Epoch')
plt.ylabel('WER')
plt.legend()
plt.title('Training and Validation WER')

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()
