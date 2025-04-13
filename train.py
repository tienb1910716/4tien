from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from models.utils import mapping_inv
from models.crnn import CRNN
from CaptchaDataset.dataset import CaptchaDataset, custom_collate_fn
from engine import Engine
from models.metrics_plot import plot_metrics 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

def train_model(model, train_loader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        tk = tqdm(train_loader, total=len(train_loader))
        for data, target in tk:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out, loss = model(data, target, criterion=criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            tk.set_postfix({'Epoch': epoch + 1, 'Loss': loss.item()})

if __name__ == "__main__":
    # Load data and initialize model
    from models.utils import df_test, df_train, num_class, mapping
    transform = T.Compose([
        T.Resize((104, 1853)),
        T.ToTensor()
    ])
    
    # Load dataset
    full_dataset = CaptchaDataset(df_train, transform=transform)
    
    # Split dataset into train, validation, and test (80%, 10%, 10%)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    # Initialize model, optimizer, and loss function
    model = CRNN(in_channels=1, output=num_class).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(zero_infinity=True)

    # Train the model
    train_model(model, train_loader, optimizer, criterion, DEVICE, epochs=70)

    # Evaluate the model (using validation data)
    engine = Engine(model, optimizer, criterion, epochs=70, device=DEVICE)
    _, _, avg_loss, word_error_rate = engine.evaluate(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test WER: {word_error_rate:.4f}")
    _, _, avg_loss, word_error_rate = engine.evaluate(val_loader)
    print(f"Val Loss: {avg_loss:.4f}")
    print(f"Val WER: {word_error_rate:.4f}")
    # Lưu các giá trị CER và WER (giả sử bạn tính toán CER và WER)
    train_losses, val_losses = [avg_loss], [avg_loss]  # Bạn cần tính toán train_loss và val_loss
    train_cers, val_cers = [0.1], [0.1]  # Ví dụ CER
    train_wers, val_wers = [word_error_rate], [word_error_rate]

    # Gọi hàm vẽ biểu đồ
    plot_metrics(train_losses, val_losses, train_cers, val_cers, train_wers, val_wers)
    # Save model
    saving = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'mapping': mapping,
        'mapping_inv': mapping_inv
    }
    torch.save(saving, 'train_model.pth')
    print("Model saved as train_model.pth")
