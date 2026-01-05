import os
import glob
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import các module tự viết
from model import UnetColor
from dataset import ColorizationDataset

# --- CẤU HÌNH ---
BATCH_SIZE = 32         # Số lượng ảnh xử lý trong 1 lần cập nhật trọng số
EPOCHS = 50             # Số lần model học lại toàn bộ tập dữ liệu
LEARNING_RATE = 2e-4    # Tốc độ học (bước nhảy gradient)
IMAGE_DIR = "images/train" # Đường dẫn folder chứa ảnh huấn luyện
MODEL_SAVE_PATH = "colorization_model.pth" # Tên file model sẽ lưu

def train():
    # 1. Setup thiết bị
    # Kiểm tra máy có GPU không, nếu có thì dùng (cuda), không thì dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 2. Load Data
    paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) # Lấy danh sách đường dẫn tất cả file .jpg
    if not paths:
        print(f"Không tìm thấy ảnh nào trong {IMAGE_DIR}")
        return

    train_ds = ColorizationDataset(paths, split='train') # Khởi tạo Dataset
    # DataLoader giúp chia nhỏ dữ liệu thành batch và xáo trộn (shuffle) ngẫu nhiên
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    
    # 3. Setup Model
    model = UnetColor().to(device) # Khởi tạo model và đưa vào bộ nhớ GPU
    # Dùng Adam Optimizer để tối ưu hóa trọng số (learning rate = 2e-4)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.L1Loss() # Dùng hàm mất mát L1 (Mean Absolute Error) để tính sai số
    
    # 4. Training Loop
    print("Bắt đầu training...")
    for epoch in range(EPOCHS): # Vòng lặp chính qua từng Epoch
        model.train() # Bật chế độ training (kích hoạt Dropout, BatchNorm)
        epoch_loss = 0
        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}") # Tạo thanh hiển thị tiến độ
        
        for data in loop: # Vòng lặp qua từng Batch dữ liệu
            L = data['L'].to(device)   # Input: Kênh L (Đen trắng) -> GPU
            ab = data['ab'].to(device) # Target: Kênh ab (Màu gốc) -> GPU
            
            optimizer.zero_grad() # Xóa sạch gradient của bước trước đó
            preds = model(L)      # Lan truyền xuôi (Forward pass): Model dự đoán màu
            loss = criterion(preds, ab) # Tính toán sai số (Loss) giữa dự đoán và thực tế
            loss.backward()       # Lan truyền ngược (Backpropagation): Tính đạo hàm
            optimizer.step()      # Cập nhật trọng số của Model dựa trên đạo hàm
            
            epoch_loss += loss.item() # Cộng dồn loss để theo dõi
            loop.set_postfix(loss=loss.item()) # Hiển thị loss tức thời trên thanh tiến độ
        
        # Lưu checkpoint (dự phòng) sau mỗi 10 epoch
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Đã lưu model tại epoch {epoch+1}")

    # Lưu file model hoàn chỉnh sau khi train xong
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Hoàn tất training!")

if __name__ == "__main__":
    # Kiểm tra folder ảnh đầu vào có tồn tại hay không
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Vui lòng copy ảnh vào thư mục '{IMAGE_DIR}' rồi chạy lại!")
    else:
        train() # Chạy hàm train