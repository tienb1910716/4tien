from PIL import Image
import torch
import os
import torchvision.transforms as T
image_path = 'data/captcha-version-2-images/samples2'
# Các phép biến đổi ảnh
transform = T.Compose([
    #T.Resize((105, 1853)), #note lại kich thước góc của ảnh cần train thôi ạ
    T.Resize((104,1853)),
    T.ToTensor()
    
])
#Resize ảnh trong thư mục train và đưa vào thư mục resized_train
def batch_resize(input_folder, output_folder, size):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.png')):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            try:
                with Image.open(input_path) as img:
                    print(f"Resizing {file_name}...")
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    img.save(output_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    print(f"Resized all images and saved to {output_folder}")
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HandwritingDataset(Dataset):
    def __init__(self, df, root_dir='data/captcha-version-2-images/samples2', transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['images']
        label = self.df.iloc[idx]['label']
        img_path = os.path.join(self.root_dir, img_name)
       
        image = Image.open(img_path).convert('L')  # Grayscale
        if self.transform:
            image = self.transform(image)
        # Chuyển label thành chuỗi ký tự (nếu cần)
        label_str = ''.join([chr(c) for c in label])  # Chuyển từ danh sách chỉ số về chuỗi
        return image, label_str

# Transform để resize và chuẩn hóa ảnh
transform = transforms.Compose([
    transforms.Resize((104, 1853)),  # Giữ kích thước mong muốn
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Chuẩn hóa cho ảnh grayscale
])
#chuẩn bị batch dữ liệu với padding cho nhãn để đảm bảo kích thước đồng nhất.
def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    
    return images, labels
