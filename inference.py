import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.color import rgb2lab

# Import module tự viết
from model import UnetColor
from utils import lab_to_rgb

# --- CẤU HÌNH ---
MODEL_PATH = "colorization_model.pth" # Đường dẫn file model đã train
IMAGE_PATH = "test_image.jpg"         # Ảnh muốn tô màu
IMG_SIZE = 256                        # Kích thước ảnh đầu vào cho model

def colorize_image(img_path, model_path):
    # Kiểm tra và sử dụng GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = UnetColor().to(device) # Khởi tạo kiến trúc mạng U-Net
    try:
        # Load trọng số (weights) đã train vào mô hình
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Đã load model thành công!")
    except FileNotFoundError:
        print("Chưa có file model. Hãy chạy train.py trước!")
        return

    model.eval() # Chuyển sang chế độ đánh giá (tắt Dropout, khóa BatchNorm)
    
    # 2. Xử lý ảnh Input
    img = Image.open(img_path).convert("RGB") # Mở ảnh và đảm bảo chuẩn RGB
    original_size = img.size
    
    # Resize & Preprocess
    transform = transforms.Compose([
        # Resize về 256x256 để khớp với input của mạng
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    img_resized = transform(img)
    img_array = np.array(img_resized)
    
    # Convert sang Lab và lấy kênh L
    img_lab = rgb2lab(img_array).astype("float32") # Chuyển đổi không gian màu RGB -> Lab
    L = img_lab[:, :, 0] # Tách riêng kênh L (Lightness/Đen trắng)
    
    # Chuẩn hóa L
    # Thêm chiều Batch và Channel: (H, W) -> (1, 1, H, W)
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0) 
    L_tensor = L_tensor / 50. - 1. # Chuẩn hóa L từ [0, 100] về [-1, 1]
    L_tensor = L_tensor.to(device) # Đưa dữ liệu lên GPU
    
    # 3. Predict (Dự đoán)
    with torch.no_grad(): # Tắt tính toán gradient để tiết kiệm bộ nhớ
        ab_pred = model(L_tensor) # Model nhận L, trả về 2 kênh màu ab
    
    # 4. Hiển thị kết quả
    # Gọi hàm tiện ích để ghép L gốc + ab dự đoán -> Ảnh màu RGB
    rgb_img = lab_to_rgb(L_tensor.squeeze(0), ab_pred.squeeze(0))
    
    # Plot (Vẽ hình)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Hiển thị ảnh input (đen trắng)
    ax[0].imshow(img_resized.convert("L"), cmap='gray')
    ax[0].set_title("Đầu vào (Grayscale)")
    ax[0].axis("off")
    
    # Hiển thị ảnh output (đã tô màu)
    ax[1].imshow(rgb_img)
    ax[1].set_title("Kết quả (Colorized)")
    ax[1].axis("off")
    
    plt.show()
    
    # Lưu ảnh kết quả ra file
    plt.imsave("result.png", rgb_img)
    print("Đã lưu kết quả vào result.png")

if __name__ == "__main__":
    # Block chính để chạy chương trình
    try:
        colorize_image(IMAGE_PATH, MODEL_PATH)
    except Exception as e:
        print(f"Lỗi: {e}")
        print(f"Hãy đảm bảo file ảnh '{IMAGE_PATH}' tồn tại.")