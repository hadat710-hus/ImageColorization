import torch
from torch import nn

class UnetBlock(nn.Module):
    # Khởi tạo Block con của U-Net
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False, innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost # Lưu cờ đánh dấu lớp ngoài cùng
        if input_c is None: input_c = nf # Mặc định số kênh vào bằng số filter
        
        # Định nghĩa các layer cơ bản
        downconv = nn.Conv2d(input_c, ni, kernel_size=4, stride=2, padding=1, bias=False) # Conv giảm size (Encoder)
        downrelu = nn.LeakyReLU(0.2, True) # Hàm kích hoạt cho Encoder
        downnorm = nn.BatchNorm2d(ni) # Chuẩn hóa dữ liệu
        uprelu = nn.ReLU(True) # Hàm kích hoạt cho Decoder
        upnorm = nn.BatchNorm2d(nf) # Chuẩn hóa dữ liệu decoder
        
        # Xử lý logic ghép nối các layer tùy vị trí
        if outermost: # Nếu là lớp vỏ ngoài cùng (Input/Output)
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1) # Conv tăng size
            down = [downconv] # Phần đi xuống
            up = [uprelu, upconv, nn.Tanh()] # Phần đi lên + Tanh (để output [-1, 1])
            model = down + [submodule] + up # Ghép: Xuống -> Con -> Lên
            
        elif innermost: # Nếu là lớp lõi trong cùng (Bottleneck)
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4, stride=2, padding=1, bias=False) # Conv tăng size
            down = [downrelu, downconv] # Phần đi xuống
            up = [uprelu, upconv, upnorm] # Phần đi lên
            model = down + up # Ghép: Xuống -> Lên (không có con)
            
        else: # Các lớp ở giữa
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1, bias=False) # Conv tăng size
            down = [downrelu, downconv, downnorm] # Phần đi xuống + BN
            up = [uprelu, upconv, upnorm] # Phần đi lên + BN
            if dropout: up += [nn.Dropout(0.5)] # Thêm Dropout nếu cần
            model = down + [submodule] + up # Ghép: Xuống -> Con -> Lên
            
        self.model = nn.Sequential(*model) # Đóng gói thành Sequential
    
    def forward(self, x):
        if self.outermost: return self.model(x) # Lớp vỏ trả về kết quả luôn
        return torch.cat([x, self.model(x)], 1) # Các lớp khác: Nối tắt (Skip Connection)

class UnetColor(nn.Module):
    # Khởi tạo mô hình chính
    def __init__(self, input_c=1, output_c=2, n_filters=64):
        super().__init__()
        # Xây dựng từ trong ra ngoài (Đệ quy)
        
        # 1. Tạo lớp lõi (Bottleneck)
        unet_block = UnetBlock(n_filters * 8, n_filters * 8, innermost=True)
        
        # 2. Các lớp ở giữa (bọc lấy lớp lõi)
        unet_block = UnetBlock(n_filters * 8, n_filters * 8, submodule=unet_block, dropout=True)
        unet_block = UnetBlock(n_filters * 8, n_filters * 8, submodule=unet_block, dropout=True)
        unet_block = UnetBlock(n_filters * 8, n_filters * 8, submodule=unet_block, dropout=True)
        unet_block = UnetBlock(n_filters * 4, n_filters * 8, submodule=unet_block)
        unet_block = UnetBlock(n_filters * 2, n_filters * 4, submodule=unet_block)
        unet_block = UnetBlock(n_filters, n_filters * 2, submodule=unet_block)
        
        # 3. Tạo lớp vỏ ngoài cùng (chứa tất cả các lớp trên)
        self.model = UnetBlock(output_c, n_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x) # Truyền dữ liệu qua mô hình