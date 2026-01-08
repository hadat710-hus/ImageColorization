import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
import os

# --- IMPORT MODEL ---
# Đảm bảo bạn có file model.py chứa class UnetColor cùng thư mục
from model import UnetColor

# --- CẤU HÌNH ---
MODEL_PATH = "colorization_model.pth"
IMG_SIZE = 256

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="AI Colorizer Pro",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 AI Image Colorization (Full Resolution)")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Tự động chọn GPU nếu có, không thì dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnetColor().to(device)
    
    if not os.path.exists(MODEL_PATH):
        return None, device
        
    try:
        # Load weights
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None, device

# Gọi hàm load
model, device = load_model()

# Kiểm tra trạng thái model
if model is None:
    st.error(f"❌ Không tìm thấy file '{MODEL_PATH}' hoặc file bị lỗi.")
    st.info("👉 Hãy tải file .pth từ Colab về và đặt vào cùng thư mục với app.py")
else:
    st.sidebar.success(f"✅ System Ready! Device: {device}")

# --- GIAO DIỆN UPLOAD ---
uploaded_file = st.file_uploader("Chọn ảnh đen trắng (hoặc ảnh màu) để test...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Đọc ảnh và chuyển về RGB
    image = Image.open(uploaded_file).convert("RGB")
    
    # Lấy kích thước gốc (Width, Height) để lát nữa phóng to lại
    orig_w, orig_h = image.size
    
    # Chia cột giao diện
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Ảnh gốc")
        st.image(image.convert("L"), use_container_width=True)

    # Nút bấm xử lý
    if st.button("🚀 Tô màu ngay", use_container_width=True):
        with st.spinner("AI đang tô màu... vui lòng đợi..."):
            try:
                # --- GIAI ĐOẠN 1: XỬ LÝ ẢNH GỐC (LẤY NÉT) ---
                # Chuyển ảnh gốc sang mảng Numpy
                img_original_np = np.array(image)
                # Chuyển sang không gian Lab
                img_lab_original = rgb2lab(img_original_np).astype("float32")
                # Tách lấy lớp L (Lightness) ở độ phân giải gốc -> Dùng cái này để ảnh nét
                L_original = img_lab_original[:, :, 0] 

                # --- GIAI ĐOẠN 2: CHUẨN BỊ INPUT CHO AI (RESIZE) ---
                # AI chỉ nhận ảnh vuông 256x256, nên phải resize một bản copy
                transform = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
                ])
                img_resized = transform(image)
                img_array_resized = np.array(img_resized)
                
                # Chuyển bản resize sang Lab để lấy đầu vào cho Model
                img_lab_resized = rgb2lab(img_array_resized).astype("float32")
                L_input = img_lab_resized[:, :, 0]
                
                # Chuẩn hóa về khoảng [-1, 1] và tạo Tensor
                L_tensor = torch.from_numpy(L_input).unsqueeze(0).unsqueeze(0) # (1, 1, 256, 256)
                L_tensor = L_tensor / 50. - 1.
                L_tensor = L_tensor.to(device)
                
                # --- GIAI ĐOẠN 3: AI DỰ ĐOÁN MÀU ---
                with torch.no_grad():
                    ab_pred = model(L_tensor) # Kết quả ra size (1, 2, 256, 256)

                # --- GIAI ĐOẠN 4: HẬU XỬ LÝ (QUAN TRỌNG) ---
                # Phóng to lớp màu ab từ 256x256 lên kích thước gốc (orig_h, orig_w)
                ab_pred_upscaled = torch.nn.functional.interpolate(
                    ab_pred, 
                    size=(orig_h, orig_w), 
                    mode='bilinear', 
                    align_corners=True
                )
                
                # Chuyển về Numpy và nhân với 128 để khôi phục độ bão hòa màu
                # (Vì output của Tanh là -1 đến 1, còn màu Lab là -128 đến 128)
                ab_final = ab_pred_upscaled.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 128.0
                
                # --- GIAI ĐOẠN 5: GHÉP ẢNH ---
                # Tạo ảnh Lab rỗng kích thước gốc
                final_lab_image = np.zeros((orig_h, orig_w, 3))
                
                # Kênh 0: Lấy từ ảnh gốc (để giữ độ nét chi tiết)
                final_lab_image[:, :, 0] = L_original
                
                # Kênh 1, 2: Lấy từ AI đã phóng to (để lấy màu)
                final_lab_image[:, :, 1:] = ab_final
                
                # Chuyển ngược từ Lab sang RGB để hiển thị
                final_rgb_image = lab2rgb(final_lab_image)
                
                # --- HIỂN THỊ KẾT QUẢ ---
                with col2:
                    st.subheader("🎨 Kết quả AI (Full HD)")
                    # clamp=True giúp cắt bỏ các giá trị màu bị nhiễu vượt quá giới hạn
                    st.image(final_rgb_image, use_container_width=True, clamp=True)
                    
                st.balloons() # Pháo hoa chúc mừng
                
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")
                st.write("Chi tiết lỗi:", e)