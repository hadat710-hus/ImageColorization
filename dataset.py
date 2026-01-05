import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.color import rgb2lab

class ColorizationDataset(Dataset):
    # Khởi tạo dataset với danh sách đường dẫn ảnh
    def __init__(self, paths, split='train'):
        self.split = split # Lưu loại tập dữ liệu ('train' hoặc 'val')
        self.paths = paths # Lưu danh sách đường dẫn ảnh
        
        # Định nghĩa các phép biến đổi ảnh (Pre-processing)
        if split == 'train':
            self.transforms = transforms.Compose([
                # Resize về 256x256 dùng thuật toán Bicubic (giữ chi tiết tốt hơn)
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(), # Lật ngang ngẫu nhiên để tăng dữ liệu (Augmentation)
            ])
        else:
            self.transforms = transforms.Compose([
                # Tập validation chỉ cần resize, không lật ảnh
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            ])

    # Hàm lấy một mẫu dữ liệu tại vị trí idx
    def __getitem__(self, idx):
        try:
            # 1. Đọc ảnh và tiền xử lý
            img = Image.open(self.paths[idx]).convert("RGB") # Đọc ảnh, đảm bảo đủ 3 kênh màu
            img = self.transforms(img) # Resize và Augmentation
            img = np.array(img) # Chuyển sang mảng Numpy để dùng thư viện skimage
            
            # 2. Chuyển đổi không gian màu RGB -> LAB
            img_lab = rgb2lab(img).astype("float32") # Chuyển sang LAB
            img_lab = transforms.ToTensor()(img_lab) # Chuyển sang Tensor PyTorch (C, H, W)
            
            # 3. Chuẩn hóa dữ liệu (Normalization) về khoảng [-1, 1]
            # Kênh L (Lightness): giá trị gốc [0, 100] -> Chia 50 trừ 1 -> [-1, 1]
            L = img_lab[[0], ...] / 50. - 1. 
            
            # Kênh ab (Color): giá trị gốc khoảng [-128, 127] -> Chia 110 -> xấp xỉ [-1, 1]
            ab = img_lab[[1, 2], ...] / 110. 
            
            # Trả về dictionary: L là đầu vào, ab là nhãn (target)
            return {'L': L, 'ab': ab}
            
        except Exception as e:
            # Nếu ảnh lỗi (file hỏng), đệ quy lấy ảnh tiếp theo để không ngắt quãng training
            return self.__getitem__((idx + 1) % len(self.paths))

    # Trả về tổng số lượng ảnh trong dataset
    def __len__(self):
        return len(self.paths)