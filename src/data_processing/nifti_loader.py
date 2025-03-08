"""
Module xử lý dữ liệu NIfTI.
Cung cấp các chức năng đọc, ghi và chuyển đổi tệp NIfTI.
"""

import os
import numpy as np
import nibabel as nib
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm


class NiftiLoader:
    """Lớp xử lý và tải dữ liệu NIfTI."""
    
    def __init__(self, verbose: bool = False):
        """
        Khởi tạo NiftiLoader.
        
        Args:
            verbose: Hiển thị thông tin chi tiết nếu True.
        """
        self.verbose = verbose
    
    def load_nifti_file(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Tải một tệp NIfTI.
        
        Args:
            file_path: Đường dẫn đến tệp NIfTI.
            
        Returns:
            Tuple chứa mảng dữ liệu 3D và từ điển metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tệp {file_path} không tồn tại")
            
        if self.verbose:
            print(f"Đang tải tệp NIfTI từ {file_path}")
        
        # Tải tệp NIfTI
        nifti_img = nib.load(file_path)
        
        # Lấy dữ liệu hình ảnh
        volume_data = nifti_img.get_fdata()
        
        # Thu thập metadata
        metadata = {
            'affine': nifti_img.affine.tolist(),  # Ma trận biến đổi
            'header': {k: str(v) for k, v in nifti_img.header.items() if k != 'data_type'}
        }
        
        # Thêm thông tin về độ phân giải không gian
        voxel_dims = nifti_img.header.get_zooms()
        metadata['voxel_spacing'] = voxel_dims
        
        if self.verbose:
            print(f"Đã tải tệp NIfTI thành công. Kích thước: {volume_data.shape}")
            
        return volume_data, metadata
    
    def normalize_volume(self, volume: np.ndarray, percentile_low: float = 1.0, 
                       percentile_high: float = 99.0) -> np.ndarray:
        """
        Chuẩn hóa khối dữ liệu dựa trên phân vị.
        
        Args:
            volume: Mảng dữ liệu 3D.
            percentile_low: Phân vị thấp (%).
            percentile_high: Phân vị cao (%).
            
        Returns:
            Mảng đã được chuẩn hóa.
        """
        min_bound = np.percentile(volume, percentile_low)
        max_bound = np.percentile(volume, percentile_high)
        
        # Clip giá trị
        volume = np.clip(volume, min_bound, max_bound)
        
        # Chuẩn hóa về dải [0, 1]
        volume_normalized = (volume - min_bound) / (max_bound - min_bound)
        
        return volume_normalized
    
    def resample_volume(self, volume: np.ndarray, original_spacing: Tuple[float, float, float], 
                       target_spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Lấy mẫu lại khối lượng để có khoảng cách voxel đồng nhất.
        
        Args:
            volume: Mảng dữ liệu 3D.
            original_spacing: Khoảng cách voxel gốc (x, y, z).
            target_spacing: Khoảng cách voxel mục tiêu (x, y, z).
            
        Returns:
            Mảng đã được lấy mẫu lại.
        """
        from scipy import ndimage
        
        # Tính toán tỉ lệ resize
        resize_factor = [o / t for o, t in zip(original_spacing, target_spacing)]
        
        # Tính toán kích thước mới
        new_shape = [int(round(factor * size)) for factor, size in zip(resize_factor, volume.shape)]
        
        # Tính toán các tham số lấy mẫu lại
        real_resize = [new_size / old_size for new_size, old_size in zip(new_shape, volume.shape)]
        
        # Thực hiện nội suy
        resampled_volume = ndimage.zoom(volume, real_resize, order=1)
        
        return resampled_volume
    
    def save_as_nifti(self, volume: np.ndarray, output_path: str, affine: Optional[np.ndarray] = None) -> None:
        """
        Lưu dữ liệu dưới dạng tệp NIfTI.
        
        Args:
            volume: Mảng dữ liệu 3D.
            output_path: Đường dẫn đến tệp đầu ra.
            affine: Ma trận biến đổi (tùy chọn). Nếu None, sử dụng ma trận đơn vị.
        """
        if affine is None:
            affine = np.eye(4)  # Ma trận đơn vị 4x4
            
        # Tạo đối tượng NIfTI
        nifti_img = nib.Nifti1Image(volume, affine)
        
        # Lưu tệp
        nib.save(nifti_img, output_path)
        
        if self.verbose:
            print(f"Đã lưu dữ liệu tại {output_path}")
    
    def save_as_numpy(self, volume: np.ndarray, output_path: str, metadata: Optional[Dict] = None) -> None:
        """
        Lưu dữ liệu dưới dạng tệp numpy.
        
        Args:
            volume: Mảng dữ liệu 3D.
            output_path: Đường dẫn đến tệp đầu ra.
            metadata: Từ điển chứa metadata (tùy chọn).
        """
        if metadata:
            np.savez_compressed(output_path, volume=volume, **metadata)
        else:
            np.save(output_path, volume)
        
        if self.verbose:
            print(f"Đã lưu dữ liệu tại {output_path}")


if __name__ == "__main__":
    # Ví dụ cách sử dụng
    loader = NiftiLoader(verbose=True)
    
    # Đường dẫn đến tệp NIfTI
    sample_file = "../data/raw/sample_mri.nii.gz"
    
    try:
        # Tải dữ liệu
        volume_data, metadata = loader.load_nifti_file(sample_file)
        
        # Chuẩn hóa dữ liệu
        normalized_data = loader.normalize_volume(volume_data)
        
        # Lưu dữ liệu đã xử lý
        loader.save_as_numpy(normalized_data, "../data/processed/sample_mri.npy", metadata)
        
        # Lưu lại dưới dạng NIfTI
        loader.save_as_nifti(normalized_data, "../data/processed/sample_mri_normalized.nii.gz", 
                          np.array(metadata['affine']))
        
        print(f"Kích thước khối dữ liệu: {volume_data.shape}")
        print(f"Khoảng cách voxel: {metadata.get('voxel_spacing', 'Không có')}")
        
    except Exception as e:
        print(f"Lỗi: {e}") 