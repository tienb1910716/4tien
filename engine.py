from tqdm import tqdm
import torch
from PIL import Image
import torchvision.transforms as T
import collections
import matplotlib.pyplot as plt
import numpy as np
import jiwer
from jiwer import wer
from models.utils import mapping

class Engine:
    def __init__(self, model, optimizer, criterion, epochs=70, early_stop=False, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device
        self.idx_to_char = {v: k for k, v in mapping.items()}
        self.train_losses = []  # Danh sách lưu train loss
        self.val_losses = []  # Danh sách lưu validation loss
        self.train_cers = []  # Danh sách lưu train CER
        self.val_cers = []  # Danh sách lưu validation CER
        self.train_wers = []  # Danh sách lưu train WER
        self.val_wers = []  # Danh sách lưu validation WER

    # Huấn luyện mô hình qua nhiều epoch
    # Huấn luyện mô hình qua nhiều epoch
    def fit(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            self.model.train()
            tk = tqdm(train_loader, total=len(train_loader))
            epoch_loss = 0  # Loss trung bình của epoch
            for data, target in tk:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                out, loss = self.model(data, target, criterion=self.criterion)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()
                self.train_losses.append(loss.item())
                epoch_loss += loss.item()
                tk.set_postfix({'Epoch': epoch + 1, 'Batch Loss': loss.item()})

            epoch_loss /= len(train_loader)
            self.train_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

            # **Đánh giá trên validation set sau mỗi epoch**
            val_outs, val_loss, avg_val_loss, val_wer = self.evaluate(val_loader)

            # Cập nhật các giá trị loss và error rate cho validation
            self.val_losses.append(avg_val_loss)
            self.val_wers.append(val_wer)

            # Vẽ biểu đồ sau mỗi epoch
            self.plot_metrics()


    # Đánh giá mô hình trên dữ liệu không huấn luyện (validation/test set)
   # Đánh giá mô hình trên dữ liệu không huấn luyện (validation/test set)
    def evaluate(self, dataloader):
        self.model.eval()
        hist_loss = []
        all_predictions = []
        all_references = []
        tk = tqdm(dataloader, total=len(dataloader))
        with torch.no_grad():
            for data, target in tk:
                data, target = data.to(self.device), target.to(self.device)
                out, loss = self.model(data, target, criterion=self.criterion)
                decoded_preds = self.decode_predictions(out)
                decoded_targets = self.decode_labels(target)
                all_predictions.extend(decoded_preds)
                all_references.extend(decoded_targets)
                hist_loss.append(loss.item())
                tk.set_postfix({'Loss': loss.item()})

        avg_loss = sum(hist_loss) / len(hist_loss)
        cer = self.calculate_cer(all_references, all_predictions)
        wer = self.calculate_wer(all_references, all_predictions)
        print(f"Validation Loss: {avg_loss:.4f}, CER: {cer:.4f}, WER: {wer:.4f}")
        return all_predictions, hist_loss, avg_loss, wer


    # Tính toán CER (Character Error Rate)
    def calculate_cer(self, references, predictions):
        total_cer = 0
        for ref, pred in zip(references, predictions):
            total_cer += jiwer.cer(ref, pred)  # Tính CER
        return total_cer / len(references)

    # Tính toán WER (Word Error Rate)
    def calculate_wer(self, references, predictions):
        total_wer = 0
        for ref, pred in zip(references, predictions):
            total_wer += wer(ref, pred)  # Sử dụng thư viện WER để tính toán
        return total_wer / len(references)

    # Dự đoán chuỗi ký tự từ một ảnh đầu vào
    def predict(self, image_path):
        image = Image.open(image_path).convert('L')
        image_tensor = T.ToTensor()(image).unsqueeze(0).to(self.device)
        out, _ = self.model(image_tensor)
        out = out.permute(1, 0, 2).log_softmax(2).argmax(2)
        out = out.cpu().detach().numpy()
        return out

    # Decode các đầu ra của mô hình
    def decode_predictions(self, outputs):
        probabilities = torch.softmax(outputs, dim=-1)
        predicted_indices = torch.argmax(probabilities, dim=-1)
        decoded_texts = []

        for indices in predicted_indices:
            text = []
            prev_idx = None
            for idx in indices:
                if idx != prev_idx and idx != 0:
                    text.append(self.idx_to_char[idx.item()])
                prev_idx = idx
            decoded_texts.append(''.join(text))
        
        return decoded_texts

    # Decode nhãn thực từ batch
    def decode_labels(self, targets):
        decoded_texts = []
        for target in targets:
            text = [self.idx_to_char[idx.item()] for idx in target if idx != 0]
            decoded_texts.append(''.join(text))
        return decoded_texts

    # Vẽ biểu đồ loss, CER, WER
    def plot_metrics(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Loss Plot
        axs[0].plot(self.train_losses, label='Train Loss', color='blue')
        axs[0].plot(self.val_losses, label='Validation Loss', color='orange')
        axs[0].set_title('Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # CER Plot
        axs[1].plot(self.train_cers, label='Train CER', color='blue')
        axs[1].plot(self.val_cers, label='Validation CER', color='orange')
        axs[1].set_title('Character Error Rate (CER)')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('CER')
        axs[1].legend()

        # WER Plot
        axs[2].plot(self.train_wers, label='Train WER', color='blue')
        axs[2].plot(self.val_wers, label='Validation WER', color='orange')
        axs[2].set_title('Word Error Rate (WER)')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('WER')
        axs[2].legend()

        plt.tight_layout()
        plt.show()
