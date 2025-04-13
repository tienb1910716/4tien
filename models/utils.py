from sklearn.model_selection import train_test_split
import json
import pandas as pd
from CaptchaDataset.dataset import CaptchaDataset, custom_collate_fn, transform
from torch.utils.data import DataLoader


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

df = pd.DataFrame({'images': images, 'label': labels})

# Chia dữ liệu thành 80% train, 10% validation và 10% test
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))
test_size = len(df) - train_size - val_size

# Đầu tiên chia thành 80% train và 20% (validation + test)
df_train, df_temp = train_test_split(df, test_size=val_size + test_size, shuffle=True)

# Sau đó chia 20% còn lại thành 50% cho validation và 50% cho test
df_val, df_test = train_test_split(df_temp, test_size=test_size / (val_size + test_size), shuffle=True)

# Tạo dataset và dataloader
train_data = CaptchaDataset(df_train, transform=transform)
val_data = CaptchaDataset(df_val, transform=transform)
test_data = CaptchaDataset(df_test, transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_data, batch_size=4, collate_fn=custom_collate_fn)



