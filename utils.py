import torch
import numpy as np
from skimage.color import lab2rgb

def lab_to_rgb(L, ab):
    """Hàm chuyển đổi tensor Lab đã chuẩn hóa sang ảnh RGB để hiển thị"""
    
    # 1. Hoàn tác chuẩn hóa (Denormalize)
    L = (L + 1.) * 50. # Đưa L từ [-1, 1] về [0, 100]
    ab = ab * 110.     # Đưa ab từ [-1, 1] về [-110, 110]
    
    # 2. Ghép kênh
    # Nối L và ab lại, tách khỏi đồ thị (detach), chuyển về CPU và NumPy
    lab = torch.cat([L, ab], dim=0).detach().cpu().numpy()
    
    # 3. Đổi chiều dữ liệu
    # PyTorch dùng (Channels, H, W) -> Skimage cần (H, W, Channels)
    lab = lab.transpose(1, 2, 0)
    
    # 4. Chuyển đổi sang RGB
    # Chuyển kiểu dữ liệu sang float64 để tránh lỗi thư viện skimage
    rgb = lab2rgb(lab.astype("float64"))
    
    return rgb # Trả về mảng ảnh RGB hoàn chỉnh