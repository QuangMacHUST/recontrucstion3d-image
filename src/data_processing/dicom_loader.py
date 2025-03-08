"""
Module xử lý dữ liệu DICOM.
Cung cấp các chức năng đọc, ghi và chuyển đổi tệp DICOM.
"""

import os
import numpy as np
import pydicom
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm


class DicomLoader:
    """Lớp xử lý và tải dữ liệu DICOM."""
    
    def __init__(self, verbose: bool = False):
        """
        Khởi tạo DicomLoader.
        
        Args:
            verbose: Hiển thị thông tin chi tiết nếu True.
        """
        self.verbose = verbose
    
    def load_dicom_series(self, directory: str) -> Tuple[np.ndarray, Dict]:
        """
        Tải một chuỗi tệp DICOM từ thư mục.
        
        Args:
            directory: Đường dẫn đến thư mục chứa chuỗi tệp DICOM.
            
        Returns:
            Tuple chứa mảng dữ liệu 3D và từ điển metadata.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Thư mục {directory} không tồn tại")
            
        # Lấy danh sách tất cả các tệp DICOM trong thư mục
        dicom_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                     if os.path.isfile(os.path.join(directory, f)) and f.endswith('.dcm')]
        
        if not dicom_files:
            raise ValueError(f"Không tìm thấy tệp DICOM nào trong thư mục {directory}")
            
        if self.verbose:
            print(f"Đã tìm thấy {len(dicom_files)} tệp DICOM.")
        
        # Đọc tệp đầu tiên để lấy thông tin
        first_dicom = pydicom.dcmread(dicom_files[0])
        
        # Lấy kích thước của hình ảnh
        pixel_dims = (int(first_dicom.Rows), int(first_dicom.Columns), len(dicom_files))
        
        # Khởi tạo mảng numpy để lưu trữ dữ liệu voxel
        volume_data = np.zeros(pixel_dims, dtype=first_dicom.pixel_array.dtype)
        
        # Từ điển để lưu trữ thông tin position của slice để sắp xếp theo thứ tự chính xác
        position_dict = {}
        
        # Đọc tất cả các tệp DICOM và lưu trữ dữ liệu hình ảnh
        for i, file_path in enumerate(tqdm(dicom_files, disable=not self.verbose)):
            ds = pydicom.dcmread(file_path)
            
            # Lấy thông tin vị trí slice trong không gian 3D
            try:
                position = float(ds.ImagePositionPatient[2])
            except (AttributeError, IndexError):
                # Nếu không có thông tin vị trí, sử dụng chỉ số tệp
                position = i
                
            position_dict[position] = ds.pixel_array
        
        # Sắp xếp slice theo vị trí chính xác
        sorted_positions = sorted(position_dict.keys())
        
        # Điền dữ liệu vào mảng 3D
        for i, position in enumerate(sorted_positions):
            volume_data[:, :, i] = position_dict[position]
        
        # Thu thập metadata
        metadata = {}
        for element in first_dicom:
            if element.keyword and element.keyword != "PixelData":
                try:
                    metadata[element.keyword] = element.value
                except:
                    pass
        
        # Thêm thông tin về độ phân giải không gian
        try:
            pixel_spacing = first_dicom.PixelSpacing
            slice_thickness = first_dicom.SliceThickness
            metadata['voxel_spacing'] = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness))
        except:
            metadata['voxel_spacing'] = (1.0, 1.0, 1.0)  # Giá trị mặc định
            
        return volume_data, metadata
    
    def normalize_volume(self, volume: np.ndarray, min_bound: float = -1000.0, max_bound: float = 400.0) -> np.ndarray:
        """
        Chuẩn hóa giá trị voxel cho dữ liệu CT (Hounsfield Units).
        
        Args:
            volume: Mảng dữ liệu 3D.
            min_bound: Giá trị HU tối thiểu.
            max_bound: Giá trị HU tối đa.
            
        Returns:
            Mảng đã được chuẩn hóa.
        """
        # Giới hạn giá trị Hounsfield Units
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
    loader = DicomLoader(verbose=True)
    
    # Đường dẫn đến thư mục chứa chuỗi DICOM
    sample_directory = "../data/raw/sample_ct"
    
    try:
        # Tải dữ liệu
        volume_data, metadata = loader.load_dicom_series(sample_directory)
        
        # Chuẩn hóa dữ liệu
        normalized_data = loader.normalize_volume(volume_data)
        
        # Lưu dữ liệu đã xử lý
        loader.save_as_numpy(normalized_data, "../data/processed/sample_ct.npy", metadata)
        
        print(f"Kích thước khối dữ liệu: {volume_data.shape}")
        print(f"Khoảng cách voxel: {metadata.get('voxel_spacing', 'Không có')}")
        
    except Exception as e:
        print(f"Lỗi: {e}") 