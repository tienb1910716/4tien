import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, train_cers, val_cers, train_wers, val_wers):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 10))

    # Biểu đồ Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Biểu đồ Train CER và Validation CER
    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_cers, label='Train CER')
    plt.plot(epochs, val_cers, label='Validation CER')
    plt.title("Character Error Rate (CER)")
    plt.xlabel("Epochs")
    plt.ylabel("CER")
    plt.legend()
    plt.grid(True)

    # Biểu đồ Train WER và Validation WER
    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_wers, label='Train WER')
    plt.plot(epochs, val_wers, label='Validation WER')
    plt.title("Word Error Rate (WER)")
    plt.xlabel("Epochs")
    plt.ylabel("WER")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
