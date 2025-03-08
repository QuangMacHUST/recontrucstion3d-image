# Dự án Tái tạo Hình ảnh 3D từ Dữ liệu Y tế

Dự án này tập trung vào việc tái tạo mô hình 3D từ dữ liệu hình ảnh y tế như CT, MRI, PET/CT và SPECT/CT. Mục tiêu là cung cấp một bộ công cụ mạnh mẽ và linh hoạt cho việc xử lý, phân đoạn, tái tạo và trực quan hóa dữ liệu hình ảnh y tế 3D.

## Tính năng chính

- **Thu thập và xử lý dữ liệu**: Nhập và xử lý dữ liệu từ các định dạng DICOM và NIfTI.
- **Phân đoạn hình ảnh**: Áp dụng các thuật toán phân đoạn đa dạng (thresholding, region growing, deep learning).
- **Tái tạo mô hình 3D**: Chuyển đổi từ dữ liệu 2D sang mô hình 3D bằng các thuật toán như Marching Cubes.
- **Trực quan hóa**: Hiển thị mô hình 3D với các công cụ tương tác trực quan.
- **Đánh giá và xác thực**: Công cụ đánh giá độ chính xác của mô hình 3D được tạo ra.

## Cài đặt

```bash
# Sao chép mã nguồn
git clone https://github.com/yourusername/recontrucstion3d-image.git
cd recontrucstion3d-image

# Cài đặt các phụ thuộc
pip install -r requirements.txt
```

## Cấu trúc dự án

```
recontrucstion3d-image/
├── data/                  # Thư mục chứa dữ liệu
│   ├── raw/               # Dữ liệu gốc (DICOM, NIfTI)
│   └── processed/         # Dữ liệu đã xử lý
├── src/                   # Mã nguồn chính
│   ├── data_processing/   # Module xử lý dữ liệu
│   ├── segmentation/      # Module phân đoạn hình ảnh
│   ├── reconstruction/    # Module tái tạo 3D
│   ├── visualization/     # Module trực quan hóa
│   └── evaluation/        # Module đánh giá và xác thực
│   └── main.py            # Mô-đun tích hợp chính
├── tests/                 # Thư mục chứa các bài kiểm tra
├── requirements.txt       # Danh sách các phụ thuộc
└── README.md              # Tài liệu dự án
```

## Cách sử dụng

### Sử dụng dòng lệnh

Dự án cung cấp giao diện dòng lệnh để dễ dàng sử dụng:

```bash
# Cơ bản: tải một tệp DICOM và tạo mô hình 3D
python src/main.py /đường/dẫn/đến/dữ/liệu/dicom /đường/dẫn/đến/thư/mục/đầu/ra

# Sử dụng với tùy chọn thêm
python src/main.py /đường/dẫn/đến/dữ/liệu --data-type dicom --segmentation otsu --step-size 2 --visualize
```

### Tùy chọn dòng lệnh

- `--data-type`: Loại dữ liệu đầu vào (`dicom`, `nifti`, `numpy`, mặc định: `auto`)
- `--segmentation`: Phương pháp phân đoạn (`threshold`, `otsu`, `adaptive`, `deep_learning`, mặc định: `otsu`)
- `--threshold`: Giá trị ngưỡng cho phương pháp phân đoạn threshold
- `--model`: Đường dẫn đến mô hình học sâu cho phương pháp phân đoạn deep_learning
- `--step-size`: Bước nhảy cho tái tạo 3D (giá trị lớn hơn tăng tốc độ nhưng giảm chất lượng)
- `--ground-truth`: Đường dẫn đến dữ liệu chuẩn để đánh giá
- `--visualize`: Hiển thị trực quan tương tác
- `--quiet`: Không hiển thị thông tin chi tiết

### Sử dụng như một thư viện Python

Bạn cũng có thể nhúng dự án vào mã Python của riêng bạn:

```python
from src.main import MedicalImageReconstruction

# Khởi tạo pipeline
pipeline = MedicalImageReconstruction(verbose=True)

# Tải dữ liệu
pipeline.load_data("data/raw/sample_ct", data_type="dicom")

# Phân đoạn
pipeline.segment_data(method="otsu")

# Tái tạo 3D
pipeline.reconstruct_3d(step_size=1)

# Lưu kết quả
pipeline.save_results("data/processed")

# Hiển thị trực quan tương tác
pipeline.visualize_interactive()
```

## Chi tiết triển khai

### 1. Thu thập và xử lý dữ liệu

Module này xử lý việc đọc và chuẩn hóa dữ liệu từ các định dạng y tế phổ biến:

- **DICOM**: Định dạng chuẩn trong hình ảnh y tế
- **NIfTI**: Định dạng phổ biến cho dữ liệu neuroimaging
- **Numpy**: Định dạng lưu trữ đơn giản cho các mảng dữ liệu

Quá trình xử lý bao gồm:
- Chuẩn hóa khoảng dữ liệu
- Điều chỉnh khoảng cách voxel
- Trích xuất metadata

### 2. Phân đoạn hình ảnh

Module này tách biệt các cấu trúc quan tâm khỏi hình ảnh nền:

- **Ngưỡng cơ bản**: Phân đoạn dựa trên giá trị pixel
- **Ngưỡng Otsu**: Tự động tìm ngưỡng tối ưu
- **Ngưỡng thích ứng**: Thay đổi ngưỡng dựa trên đặc điểm cục bộ
- **Deep Learning**: Sử dụng mô hình U-Net 3D cho các trường hợp phân đoạn phức tạp

### 3. Tái tạo mô hình 3D

Module này chuyển đổi dữ liệu phân đoạn thành mô hình 3D:

- **Marching Cubes**: Thuật toán tái tạo bề mặt từ dữ liệu thể tích phân đoạn
- **Đơn giản hóa lưới**: Giảm số lượng mặt trong khi vẫn giữ nguyên hình dạng
- **Lưu trữ**: Hỗ trợ lưu dưới dạng OBJ, STL và các định dạng 3D khác

### 4. Trực quan hóa

Module này hiển thị kết quả 3D:

- **Hiển thị trực quan tương tác**: Sử dụng VTK để hiển thị mô hình 3D
- **Cắt mặt phẳng**: Xem các mặt cắt của thể tích
- **Tạo ảnh và video**: Xuất kết quả dưới dạng ảnh chất lượng cao

### 5. Đánh giá và xác thực

Module này đánh giá chất lượng của mô hình 3D được tạo ra:

- **Hệ số Dice và Jaccard**: Đo độ tương đồng với dữ liệu chuẩn
- **Khoảng cách Hausdorff**: Đo độ chính xác của bề mặt
- **Độ tương đồng thể tích**: So sánh thể tích giữa mô hình tái tạo và dữ liệu chuẩn

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng xem tệp CONTRIBUTING.md để biết chi tiết về quy trình đóng góp.

## Tài liệu tham khảo

- Marching Cubes: Lorensen, W. E., & Cline, H. E. (1987). Marching cubes: A high resolution 3D surface construction algorithm. ACM SIGGRAPH computer graphics, 21(4), 163-169.
- U-Net: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241).
- VTK: Schroeder, W., Martin, K., & Lorensen, B. (2006). The visualization toolkit: an object-oriented approach to 3D graphics. Kitware.

## Giấy phép

Dự án này được cấp phép theo [Giấy phép MIT](LICENSE). 