"""
Module phân đoạn hình ảnh dựa trên ngưỡng.
Cung cấp các phương pháp ngưỡng cơ bản và tự động để phân đoạn hình ảnh y tế.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Union
from scipy import ndimage


class ThresholdSegmentation:
    """Lớp phân đoạn hình ảnh dựa trên ngưỡng."""
    
    def __init__(self, verbose: bool = False):
        """
        Khởi tạo ThresholdSegmentation.
        
        Args:
            verbose: Hiển thị thông tin chi tiết nếu True.
        """
        self.verbose = verbose
    
    def binary_threshold(self, volume: np.ndarray, threshold: float) -> np.ndarray:
        """
        Phân đoạn nhị phân đơn giản dựa trên một ngưỡng cố định.
        
        Args:
            volume: Mảng dữ liệu 3D đầu vào.
            threshold: Giá trị ngưỡng (từ 0 đến 1 cho dữ liệu đã chuẩn hóa).
            
        Returns:
            Mảng nhị phân 3D (0 và 1).
        """
        return (volume >= threshold).astype(np.int8)
    
    def otsu_threshold(self, volume: np.ndarray) -> np.ndarray:
        """
        Áp dụng phương pháp Otsu để xác định ngưỡng tự động.
        
        Args:
            volume: Mảng dữ liệu 3D đầu vào.
            
        Returns:
            Mảng nhị phân 3D (0 và 1).
        """
        from skimage.filters import threshold_otsu
        
        # Làm phẳng mảng 3D thành 1D để tính ngưỡng
        flat_volume = volume.flatten()
        
        # Tính ngưỡng Otsu
        thresh = threshold_otsu(flat_volume)
        
        if self.verbose:
            print(f"Ngưỡng Otsu được tính: {thresh}")
        
        # Áp dụng ngưỡng
        return (volume >= thresh).astype(np.int8)
    
    def adaptive_threshold(self, volume: np.ndarray, block_size: int = 11, 
                         offset: float = 0.05) -> np.ndarray:
        """
        Áp dụng ngưỡng thích ứng cho từng lát cắt 2D.
        
        Args:
            volume: Mảng dữ liệu 3D đầu vào.
            block_size: Kích thước cửa sổ cho ngưỡng thích ứng. Phải là số lẻ.
            offset: Giá trị bù được trừ từ giá trị trung bình.
            
        Returns:
            Mảng nhị phân 3D (0 và 1).
        """
        from skimage.filters import threshold_local
        
        if block_size % 2 == 0:
            block_size += 1  # Đảm bảo kích thước khối là số lẻ
        
        result = np.zeros_like(volume, dtype=np.int8)
        
        # Áp dụng ngưỡng thích ứng cho từng lát cắt
        for z in range(volume.shape[2]):
            slice_2d = volume[:, :, z]
            
            # Tính ngưỡng thích ứng
            thresh_map = threshold_local(slice_2d, block_size, offset=offset)
            
            # Áp dụng ngưỡng
            result[:, :, z] = (slice_2d > thresh_map).astype(np.int8)
        
        return result
    
    def multi_threshold(self, volume: np.ndarray, thresholds: List[float]) -> np.ndarray:
        """
        Áp dụng nhiều ngưỡng để tạo ra phân đoạn nhiều lớp.
        
        Args:
            volume: Mảng dữ liệu 3D đầu vào.
            thresholds: Danh sách các giá trị ngưỡng tăng dần.
            
        Returns:
            Mảng 3D đã phân đoạn với nhiều nhãn (0, 1, 2, ...).
        """
        # Kiểm tra thresholds được sắp xếp tăng dần
        if not all(thresholds[i] <= thresholds[i+1] for i in range(len(thresholds)-1)):
            raise ValueError("Danh sách ngưỡng phải được sắp xếp tăng dần")
        
        result = np.zeros_like(volume, dtype=np.int8)
        
        # Áp dụng từng ngưỡng theo thứ tự
        for i, thresh in enumerate(thresholds, start=1):
            result[volume >= thresh] = i
        
        return result
    
    def clean_binary_mask(self, mask: np.ndarray, min_size: int = 100, 
                        closing_size: int = 3) -> np.ndarray:
        """
        Làm sạch mặt nạ nhị phân bằng cách loại bỏ các thành phần nhỏ và lấp đầy lỗ hổng.
        
        Args:
            mask: Mảng nhị phân 3D (0 và 1).
            min_size: Kích thước tối thiểu (voxel) của thành phần để giữ lại.
            closing_size: Kích thước phần tử cấu trúc cho thao tác đóng hình thái học.
            
        Returns:
            Mảng nhị phân 3D đã được làm sạch.
        """
        # Tạo phần tử cấu trúc hình cầu
        struct = ndimage.generate_binary_structure(3, 1)
        
        # Mở rộng phần tử cấu trúc nếu cần
        if closing_size > 1:
            struct = ndimage.iterate_structure(struct, closing_size)
        
        # Thực hiện đóng hình thái học để lấp đầy lỗ hổng
        mask = ndimage.binary_closing(mask, structure=struct)
        
        # Gắn nhãn các thành phần liên thông
        labeled, num_components = ndimage.label(mask)
        
        if self.verbose:
            print(f"Số thành phần trước khi lọc: {num_components}")
        
        # Đếm kích thước của từng thành phần
        component_sizes = np.bincount(labeled.ravel())
        
        # Bỏ qua thành phần 0 (nền)
        component_sizes[0] = 0
        
        # Tạo mặt nạ cho các thành phần lớn hơn min_size
        big_components = component_sizes >= min_size
        
        # Xóa các thành phần nhỏ
        mask_cleaned = big_components[labeled]
        
        if self.verbose:
            remaining_components = np.sum(big_components)
            print(f"Số thành phần sau khi lọc: {remaining_components}")
        
        return mask_cleaned.astype(np.int8)
    
    def visualize_segmentation(self, volume: np.ndarray, segmentation: np.ndarray, 
                             slice_idx: Optional[int] = None, save_path: Optional[str] = None) -> None:
        """
        Trực quan hóa kết quả phân đoạn trên một lát cắt cụ thể.
        
        Args:
            volume: Mảng dữ liệu 3D gốc.
            segmentation: Mảng 3D phân đoạn.
            slice_idx: Chỉ số lát cắt cần hiển thị. Nếu None, sẽ chọn lát cắt giữa.
            save_path: Đường dẫn để lưu hình ảnh. Nếu None, sẽ hiển thị hình ảnh.
        """
        if slice_idx is None:
            slice_idx = volume.shape[2] // 2  # Lát cắt giữa theo trục z
        
        # Lấy lát cắt từ khối dữ liệu
        orig_slice = volume[:, :, slice_idx]
        seg_slice = segmentation[:, :, slice_idx]
        
        # Tạo hình
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Hiển thị lát cắt gốc
        axes[0].imshow(orig_slice, cmap='gray')
        axes[0].set_title('Lát cắt gốc')
        axes[0].axis('off')
        
        # Hiển thị lát cắt phân đoạn
        axes[1].imshow(orig_slice, cmap='gray')
        axes[1].imshow(seg_slice, cmap='hot', alpha=0.3)
        axes[1].set_title('Phân đoạn')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            if self.verbose:
                print(f"Đã lưu hình ảnh tại {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Ví dụ cách sử dụng
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.data_processing.nifti_loader import NiftiLoader
    
    # Tải dữ liệu mẫu
    loader = NiftiLoader(verbose=True)
    segmenter = ThresholdSegmentation(verbose=True)
    
    # Đường dẫn đến tệp dữ liệu mẫu
    sample_file = "../../data/raw/sample_mri.nii.gz"
    
    try:
        # Kiểm tra xem tệp có tồn tại không
        if not os.path.exists(sample_file):
            print(f"Không tìm thấy tệp mẫu tại {sample_file}")
            # Sử dụng dữ liệu mẫu giả
            volume = np.random.rand(100, 100, 50)
            print("Sử dụng dữ liệu mẫu ngẫu nhiên")
        else:
            # Tải dữ liệu thực
            volume, _ = loader.load_nifti_file(sample_file)
            volume = loader.normalize_volume(volume)
        
        # Áp dụng phân đoạn ngưỡng
        binary_seg = segmenter.binary_threshold(volume, 0.5)
        otsu_seg = segmenter.otsu_threshold(volume)
        
        # Làm sạch mặt nạ
        cleaned_seg = segmenter.clean_binary_mask(otsu_seg)
        
        # Đảm bảo thư mục đầu ra tồn tại
        os.makedirs("../../data/processed", exist_ok=True)
        
        # Trực quan hóa kết quả
        segmenter.visualize_segmentation(volume, cleaned_seg, 
                                       save_path="../../data/processed/segmentation_example.png")
        
        print("Đã hoàn thành phân đoạn.")
        
    except Exception as e:
        print(f"Lỗi: {e}") 