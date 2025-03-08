"""
Module chính tích hợp toàn bộ pipeline xử lý và tái tạo 3D.
Cung cấp giao diện dòng lệnh để dễ dàng sử dụng các chức năng của dự án.
"""

import os
import sys
import argparse
import time
import numpy as np
from typing import Tuple, Dict, List, Optional, Union

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import các module của dự án
from src.data_processing.dicom_loader import DicomLoader
from src.data_processing.nifti_loader import NiftiLoader
from src.segmentation.threshold_segmentation import ThresholdSegmentation
from src.segmentation.deep_learning_segmentation import UNet3D, preprocess_data, reconstruct_volume
from src.reconstruction.marching_cubes import MarchingCubesReconstruction
from src.visualization.vtk_visualizer import VTKVisualizer
from src.evaluation.model_evaluation import ModelEvaluator


class MedicalImageReconstruction:
    """Lớp tích hợp pipeline tái tạo 3D từ dữ liệu y tế."""
    
    def __init__(self, verbose: bool = True):
        """
        Khởi tạo pipeline tái tạo 3D.
        
        Args:
            verbose: Hiển thị thông tin chi tiết nếu True.
        """
        self.verbose = verbose
        
        # Khởi tạo các thành phần
        self.dicom_loader = DicomLoader(verbose=verbose)
        self.nifti_loader = NiftiLoader(verbose=verbose)
        self.segmenter = ThresholdSegmentation(verbose=verbose)
        self.reconstructor = MarchingCubesReconstruction(verbose=verbose)
        self.evaluator = ModelEvaluator(verbose=verbose)
        
        # Trạng thái của pipeline
        self.volume_data = None
        self.segmented_data = None
        self.mesh = None
        self.metadata = None
    
    def load_data(self, input_path: str, data_type: str = 'auto') -> bool:
        """
        Tải dữ liệu từ tệp hoặc thư mục.
        
        Args:
            input_path: Đường dẫn đến tệp hoặc thư mục dữ liệu.
            data_type: Loại dữ liệu ('dicom', 'nifti', 'numpy', 'auto').
            
        Returns:
            True nếu tải thành công, False nếu không.
        """
        try:
            # Xác định loại dữ liệu nếu data_type là 'auto'
            if data_type == 'auto':
                if os.path.isdir(input_path):
                    # Kiểm tra nếu là thư mục DICOM
                    dicom_files = [f for f in os.listdir(input_path) if f.endswith('.dcm')]
                    if dicom_files:
                        data_type = 'dicom'
                    else:
                        raise ValueError(f"Không thể tự động xác định loại dữ liệu trong thư mục {input_path}")
                else:
                    # Kiểm tra phần mở rộng tệp
                    ext = os.path.splitext(input_path)[1].lower()
                    if ext in ['.nii', '.nii.gz']:
                        data_type = 'nifti'
                    elif ext in ['.npy', '.npz']:
                        data_type = 'numpy'
                    else:
                        raise ValueError(f"Không hỗ trợ loại tệp {ext}")
            
            # Tải dữ liệu dựa trên loại
            if data_type == 'dicom':
                self.volume_data, self.metadata = self.dicom_loader.load_dicom_series(input_path)
                
                # Chuẩn hóa dữ liệu
                self.volume_data = self.dicom_loader.normalize_volume(self.volume_data)
                
            elif data_type == 'nifti':
                self.volume_data, self.metadata = self.nifti_loader.load_nifti_file(input_path)
                
                # Chuẩn hóa dữ liệu
                self.volume_data = self.nifti_loader.normalize_volume(self.volume_data)
                
            elif data_type == 'numpy':
                # Tải dữ liệu từ tệp numpy
                data = np.load(input_path, allow_pickle=True)
                
                # Kiểm tra xem có phải là tệp npz không
                if isinstance(data, np.lib.npyio.NpzFile):
                    # Trích xuất dữ liệu và metadata từ npz
                    if 'volume' in data:
                        self.volume_data = data['volume']
                    else:
                        # Lấy mảng đầu tiên
                        for key in data.keys():
                            self.volume_data = data[key]
                            break
                    
                    # Trích xuất metadata nếu có
                    self.metadata = {}
                    for key in data.keys():
                        if key != 'volume':
                            self.metadata[key] = data[key]
                else:
                    # Chỉ là một mảng numpy đơn giản
                    self.volume_data = data
                    self.metadata = {}
            else:
                raise ValueError(f"Loại dữ liệu không được hỗ trợ: {data_type}")
            
            if self.verbose:
                print(f"Đã tải dữ liệu từ {input_path}")
                print(f"Kích thước dữ liệu: {self.volume_data.shape}")
                
            return True
            
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return False
    
    def segment_data(self, method: str = 'otsu', threshold: Optional[float] = None, 
                  model_path: Optional[str] = None) -> bool:
        """
        Phân đoạn dữ liệu thể tích.
        
        Args:
            method: Phương pháp phân đoạn ('threshold', 'otsu', 'adaptive', 'deep_learning').
            threshold: Giá trị ngưỡng (chỉ dùng cho phương pháp 'threshold').
            model_path: Đường dẫn đến mô hình học sâu (chỉ dùng cho phương pháp 'deep_learning').
            
        Returns:
            True nếu phân đoạn thành công, False nếu không.
        """
        if self.volume_data is None:
            print("Lỗi: Chưa tải dữ liệu.")
            return False
        
        try:
            start_time = time.time()
            
            if method == 'threshold':
                if threshold is None:
                    threshold = 0.5  # Giá trị mặc định
                
                self.segmented_data = self.segmenter.binary_threshold(self.volume_data, threshold)
                
            elif method == 'otsu':
                self.segmented_data = self.segmenter.otsu_threshold(self.volume_data)
                
            elif method == 'adaptive':
                self.segmented_data = self.segmenter.adaptive_threshold(self.volume_data)
                
            elif method == 'deep_learning':
                if model_path is None or not os.path.exists(model_path):
                    print("Lỗi: Cần đường dẫn đến mô hình học sâu hợp lệ.")
                    return False
                
                # Tiền xử lý dữ liệu cho mô hình học sâu
                patches, _ = preprocess_data(self.volume_data)
                
                # Tạo mô hình U-Net 3D
                input_shape = patches.shape[1:]
                model = UNet3D(
                    input_shape=input_shape,
                    n_classes=1,
                    verbose=self.verbose
                )
                
                # Tải trọng số
                model.load_weights(model_path)
                
                # Dự đoán
                patches_pred = model.predict(patches)
                
                # Tái tạo khối dữ liệu từ các patch
                self.segmented_data = reconstruct_volume(patches_pred, self.volume_data.shape[:3])
                
            else:
                print(f"Lỗi: Phương pháp phân đoạn không được hỗ trợ: {method}")
                return False
            
            # Làm sạch kết quả phân đoạn
            self.segmented_data = self.segmenter.clean_binary_mask(self.segmented_data)
            
            if self.verbose:
                elapsed_time = time.time() - start_time
                print(f"Đã hoàn thành phân đoạn sử dụng phương pháp {method} trong {elapsed_time:.2f} giây")
            
            return True
            
        except Exception as e:
            print(f"Lỗi khi phân đoạn dữ liệu: {e}")
            return False
    
    def reconstruct_3d(self, step_size: int = 1) -> bool:
        """
        Tái tạo mô hình 3D từ dữ liệu phân đoạn.
        
        Args:
            step_size: Bước nhảy để giảm độ phân giải và tăng tốc độ xử lý.
            
        Returns:
            True nếu tái tạo thành công, False nếu không.
        """
        if self.segmented_data is None:
            print("Lỗi: Chưa phân đoạn dữ liệu.")
            return False
        
        try:
            start_time = time.time()
            
            # Lấy khoảng cách voxel từ metadata nếu có
            spacing = (1.0, 1.0, 1.0)  # Giá trị mặc định
            if self.metadata and 'voxel_spacing' in self.metadata:
                spacing = self.metadata['voxel_spacing']
            
            # Tạo lưới 3D
            self.mesh = self.reconstructor.generate_mesh(
                self.segmented_data,
                threshold=0.5,
                step_size=step_size,
                spacing=spacing
            )
            
            if self.verbose:
                elapsed_time = time.time() - start_time
                print(f"Đã hoàn thành tái tạo 3D trong {elapsed_time:.2f} giây")
                print(f"Số đỉnh: {len(self.mesh['vertices'])}, Số mặt: {len(self.mesh['faces'])}")
            
            return True
            
        except Exception as e:
            print(f"Lỗi khi tái tạo 3D: {e}")
            return False
    
    def save_results(self, output_dir: str, base_name: str = 'result') -> bool:
        """
        Lưu kết quả của pipeline.
        
        Args:
            output_dir: Thư mục đầu ra.
            base_name: Tên cơ sở cho các tệp đầu ra.
            
        Returns:
            True nếu lưu thành công, False nếu không.
        """
        try:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(output_dir, exist_ok=True)
            
            # Lưu dữ liệu gốc
            if self.volume_data is not None:
                np.save(os.path.join(output_dir, f"{base_name}_volume.npy"), self.volume_data)
            
            # Lưu dữ liệu phân đoạn
            if self.segmented_data is not None:
                np.save(os.path.join(output_dir, f"{base_name}_segmented.npy"), self.segmented_data)
                
                # Lưu hình ảnh phân đoạn
                self.segmenter.visualize_segmentation(
                    self.volume_data,
                    self.segmented_data,
                    save_path=os.path.join(output_dir, f"{base_name}_segmentation.png")
                )
            
            # Lưu lưới 3D
            if self.mesh is not None:
                # Lưu dưới dạng OBJ
                self.reconstructor.save_mesh_as_obj(
                    self.mesh,
                    os.path.join(output_dir, f"{base_name}_mesh.obj")
                )
                
                # Lưu hình ảnh lưới
                self.reconstructor.visualize_mesh(
                    self.mesh,
                    save_path=os.path.join(output_dir, f"{base_name}_mesh.png")
                )
            
            # Lưu metadata
            if self.metadata:
                np.savez(os.path.join(output_dir, f"{base_name}_metadata.npz"), **self.metadata)
            
            if self.verbose:
                print(f"Đã lưu kết quả tại {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"Lỗi khi lưu kết quả: {e}")
            return False
    
    def visualize_interactive(self) -> None:
        """
        Hiển thị trực quan tương tác của mô hình 3D.
        """
        if self.mesh is None:
            print("Lỗi: Chưa tái tạo mô hình 3D.")
            return
        
        try:
            # Tạo trình trực quan hóa
            visualizer = VTKVisualizer()
            
            # Thêm lưới
            visualizer.add_mesh(
                self.mesh['vertices'],
                self.mesh['faces'],
                color=(0.8, 0.2, 0.2)
            )
            
            # Thêm lưới dạng khung dây
            visualizer.add_mesh(
                self.mesh['vertices'],
                self.mesh['faces'],
                color=(1.0, 1.0, 1.0),
                opacity=0.1,
                wireframe=True
            )
            
            # Thêm trục tọa độ
            visualizer.add_axes()
            
            # Thêm văn bản
            visualizer.add_text("Mô hình 3D y tế", position=(0.05, 0.95))
            
            # Bắt đầu tương tác
            if self.verbose:
                print("Đang khởi động trình trực quan hóa tương tác...")
            
            visualizer.start_interaction()
            
        except Exception as e:
            print(f"Lỗi khi hiển thị trực quan tương tác: {e}")
    
    def evaluate_result(self, ground_truth_path: Optional[str] = None) -> Dict:
        """
        Đánh giá kết quả tái tạo 3D.
        
        Args:
            ground_truth_path: Đường dẫn đến dữ liệu chuẩn (tùy chọn).
            
        Returns:
            Từ điển chứa các chỉ số đánh giá.
        """
        results = {}
        
        # Nếu có dữ liệu chuẩn, so sánh với kết quả phân đoạn
        if ground_truth_path and os.path.exists(ground_truth_path):
            try:
                # Tải dữ liệu chuẩn
                if ground_truth_path.endswith('.nii') or ground_truth_path.endswith('.nii.gz'):
                    ground_truth, _ = self.nifti_loader.load_nifti_file(ground_truth_path)
                elif ground_truth_path.endswith('.npy'):
                    ground_truth = np.load(ground_truth_path)
                else:
                    print(f"Lỗi: Không hỗ trợ định dạng tệp dữ liệu chuẩn: {ground_truth_path}")
                    return results
                
                # Đảm bảo dữ liệu chuẩn là nhị phân
                ground_truth = (ground_truth > 0).astype(np.int8)
                
                # Đánh giá phân đoạn
                if self.segmented_data is not None:
                    results = self.evaluator.evaluate_segmentation(ground_truth, self.segmented_data)
                    
                    # So sánh trực quan
                    self.evaluator.compare_volumes(
                        ground_truth,
                        self.segmented_data,
                        save_path="data/processed/evaluation_comparison.png"
                    )
            except Exception as e:
                print(f"Lỗi khi đánh giá kết quả: {e}")
        else:
            # Nếu không có dữ liệu chuẩn, chỉ đánh giá lưới 3D
            if self.mesh is not None:
                results = {
                    'vertex_count': len(self.mesh['vertices']),
                    'face_count': len(self.mesh['faces']),
                }
                
                # Tính các thuộc tính lưới
                mesh_properties = self.reconstructor.calculate_mesh_properties(self.mesh)
                results.update(mesh_properties)
        
        # Tạo báo cáo đánh giá
        if results:
            self.evaluator.generate_evaluation_report(results, "data/processed/evaluation_report.md")
            
            if self.verbose:
                print("Đã tạo báo cáo đánh giá.")
        
        return results
    
    def full_pipeline(self, input_path: str, output_dir: str, 
                    data_type: str = 'auto', 
                    segmentation_method: str = 'otsu',
                    step_size: int = 1,
                    ground_truth_path: Optional[str] = None,
                    visualize: bool = False) -> bool:
        """
        Thực thi toàn bộ pipeline từ tải dữ liệu đến hiển thị kết quả.
        
        Args:
            input_path: Đường dẫn đến dữ liệu đầu vào.
            output_dir: Thư mục đầu ra.
            data_type: Loại dữ liệu đầu vào.
            segmentation_method: Phương pháp phân đoạn.
            step_size: Bước nhảy cho tái tạo 3D.
            ground_truth_path: Đường dẫn đến dữ liệu chuẩn (tùy chọn).
            visualize: Hiển thị trực quan tương tác nếu True.
            
        Returns:
            True nếu thực thi thành công, False nếu không.
        """
        # Tải dữ liệu
        if not self.load_data(input_path, data_type):
            return False
        
        # Phân đoạn dữ liệu
        if not self.segment_data(segmentation_method):
            return False
        
        # Tái tạo 3D
        if not self.reconstruct_3d(step_size):
            return False
        
        # Lưu kết quả
        if not self.save_results(output_dir):
            return False
        
        # Đánh giá kết quả
        self.evaluate_result(ground_truth_path)
        
        # Hiển thị trực quan nếu yêu cầu
        if visualize:
            self.visualize_interactive()
        
        return True


def parse_arguments():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Tái tạo 3D từ dữ liệu y tế')
    
    # Tham số bắt buộc
    parser.add_argument('input', help='Đường dẫn đến dữ liệu đầu vào (tệp hoặc thư mục)')
    parser.add_argument('output', help='Thư mục đầu ra')
    
    # Tham số tùy chọn
    parser.add_argument('--data-type', choices=['dicom', 'nifti', 'numpy', 'auto'], default='auto',
                     help='Loại dữ liệu đầu vào (mặc định: auto)')
    parser.add_argument('--segmentation', choices=['threshold', 'otsu', 'adaptive', 'deep_learning'],
                     default='otsu', help='Phương pháp phân đoạn (mặc định: otsu)')
    parser.add_argument('--threshold', type=float, help='Giá trị ngưỡng cho phương pháp phân đoạn threshold')
    parser.add_argument('--model', help='Đường dẫn đến mô hình học sâu cho phương pháp phân đoạn deep_learning')
    parser.add_argument('--step-size', type=int, default=1, 
                     help='Bước nhảy cho tái tạo 3D (giá trị lớn hơn tăng tốc độ nhưng giảm chất lượng)')
    parser.add_argument('--ground-truth', help='Đường dẫn đến dữ liệu chuẩn để đánh giá')
    parser.add_argument('--visualize', action='store_true', help='Hiển thị trực quan tương tác')
    parser.add_argument('--quiet', action='store_true', help='Không hiển thị thông tin chi tiết')
    
    return parser.parse_args()


def main():
    """Hàm chính của chương trình."""
    # Phân tích tham số dòng lệnh
    args = parse_arguments()
    
    # Khởi tạo pipeline
    pipeline = MedicalImageReconstruction(verbose=not args.quiet)
    
    # Thực thi toàn bộ pipeline
    success = pipeline.full_pipeline(
        input_path=args.input,
        output_dir=args.output,
        data_type=args.data_type,
        segmentation_method=args.segmentation,
        step_size=args.step_size,
        ground_truth_path=args.ground_truth,
        visualize=args.visualize
    )
    
    # Trả về mã thoát
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 