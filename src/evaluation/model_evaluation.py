"""
Module đánh giá và xác thực mô hình 3D.
Cung cấp các phương pháp đánh giá độ chính xác của mô hình 3D và so sánh với mô hình chuẩn.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union
from scipy import ndimage
import skimage.metrics as metrics


class ModelEvaluator:
    """Lớp đánh giá và xác thực mô hình 3D."""
    
    def __init__(self, verbose: bool = False):
        """
        Khởi tạo đối tượng đánh giá.
        
        Args:
            verbose: Hiển thị thông tin chi tiết nếu True.
        """
        self.verbose = verbose
    
    def dice_coefficient(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Tính hệ số Dice (Dice coefficient) giữa hai mặt nạ nhị phân.
        
        Args:
            y_true: Mảng nhị phân chứa phân đoạn chuẩn.
            y_pred: Mảng nhị phân chứa phân đoạn dự đoán.
            
        Returns:
            Hệ số Dice (0 đến 1, càng cao càng tốt).
        """
        # Đảm bảo các mảng đầu vào là nhị phân
        y_true_binary = y_true > 0
        y_pred_binary = y_pred > 0
        
        # Tính giao của hai mặt nạ
        intersection = np.logical_and(y_true_binary, y_pred_binary).sum()
        
        # Tính tổng của hai mặt nạ
        union = y_true_binary.sum() + y_pred_binary.sum()
        
        # Tính hệ số Dice
        if union == 0:
            return 1.0  # Cả hai mặt nạ đều trống
        
        dice = 2.0 * intersection / union
        return dice
    
    def jaccard_index(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Tính chỉ số Jaccard (IoU - Intersection over Union) giữa hai mặt nạ nhị phân.
        
        Args:
            y_true: Mảng nhị phân chứa phân đoạn chuẩn.
            y_pred: Mảng nhị phân chứa phân đoạn dự đoán.
            
        Returns:
            Chỉ số Jaccard (0 đến 1, càng cao càng tốt).
        """
        # Đảm bảo các mảng đầu vào là nhị phân
        y_true_binary = y_true > 0
        y_pred_binary = y_pred > 0
        
        # Tính giao và hợp của hai mặt nạ
        intersection = np.logical_and(y_true_binary, y_pred_binary).sum()
        union = np.logical_or(y_true_binary, y_pred_binary).sum()
        
        # Tính chỉ số Jaccard
        if union == 0:
            return 1.0  # Cả hai mặt nạ đều trống
        
        jaccard = intersection / union
        return jaccard
    
    def hausdorff_distance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Tính khoảng cách Hausdorff giữa hai bề mặt.
        
        Args:
            y_true: Mảng nhị phân chứa phân đoạn chuẩn.
            y_pred: Mảng nhị phân chứa phân đoạn dự đoán.
            
        Returns:
            Khoảng cách Hausdorff (đơn vị voxel, càng thấp càng tốt).
        """
        from scipy.ndimage import distance_transform_edt
        
        # Đảm bảo các mảng đầu vào là nhị phân
        y_true_binary = y_true > 0
        y_pred_binary = y_pred > 0
        
        # Kiểm tra xem có mặt nạ nào trống không
        if not np.any(y_true_binary) or not np.any(y_pred_binary):
            return float('inf')  # Trả về vô cùng nếu một trong hai mặt nạ trống
        
        # Tính biến đổi khoảng cách Euclidean
        dist_true = distance_transform_edt(~y_true_binary)
        dist_pred = distance_transform_edt(~y_pred_binary)
        
        # Tính khoảng cách Hausdorff
        hausdorff_dist = max(np.max(dist_true[y_pred_binary]), np.max(dist_pred[y_true_binary]))
        
        return hausdorff_dist
    
    def average_surface_distance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Tính khoảng cách bề mặt trung bình giữa hai bề mặt.
        
        Args:
            y_true: Mảng nhị phân chứa phân đoạn chuẩn.
            y_pred: Mảng nhị phân chứa phân đoạn dự đoán.
            
        Returns:
            Khoảng cách bề mặt trung bình (đơn vị voxel, càng thấp càng tốt).
        """
        from scipy.ndimage import distance_transform_edt
        
        # Đảm bảo các mảng đầu vào là nhị phân
        y_true_binary = y_true > 0
        y_pred_binary = y_pred > 0
        
        # Kiểm tra xem có mặt nạ nào trống không
        if not np.any(y_true_binary) or not np.any(y_pred_binary):
            return float('inf')  # Trả về vô cùng nếu một trong hai mặt nạ trống
        
        # Tính biến đổi khoảng cách Euclidean
        dist_true = distance_transform_edt(~y_true_binary)
        dist_pred = distance_transform_edt(~y_pred_binary)
        
        # Tìm bề mặt (biên) của các mặt nạ
        surface_true = self._get_surface(y_true_binary)
        surface_pred = self._get_surface(y_pred_binary)
        
        # Tính khoảng cách trung bình
        avg_dist_true_to_pred = np.mean(dist_pred[surface_true])
        avg_dist_pred_to_true = np.mean(dist_true[surface_pred])
        
        # Khoảng cách bề mặt trung bình hai chiều
        avg_surface_dist = (avg_dist_true_to_pred + avg_dist_pred_to_true) / 2.0
        
        return avg_surface_dist
    
    def _get_surface(self, binary_volume: np.ndarray) -> np.ndarray:
        """
        Tìm bề mặt (biên) của khối dữ liệu nhị phân.
        
        Args:
            binary_volume: Mảng nhị phân.
            
        Returns:
            Mảng boolean chỉ ra vị trí của bề mặt.
        """
        # Tạo phần tử cấu trúc
        struct = ndimage.generate_binary_structure(3, 1)
        
        # Tìm bề mặt bằng cách hiệu giữa khối ban đầu và phần xói mòn
        eroded = ndimage.binary_erosion(binary_volume, struct)
        surface = np.logical_and(binary_volume, np.logical_not(eroded))
        
        return surface
    
    def volume_similarity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Tính độ tương đồng thể tích giữa hai khối dữ liệu.
        
        Args:
            y_true: Mảng nhị phân chứa phân đoạn chuẩn.
            y_pred: Mảng nhị phân chứa phân đoạn dự đoán.
            
        Returns:
            Độ tương đồng thể tích (0 đến 1, càng cao càng tốt).
        """
        # Đảm bảo các mảng đầu vào là nhị phân
        y_true_binary = y_true > 0
        y_pred_binary = y_pred > 0
        
        # Tính thể tích (số voxel) của mỗi khối
        vol_true = np.sum(y_true_binary)
        vol_pred = np.sum(y_pred_binary)
        
        # Kiểm tra xem có thể tích nào bằng 0 không
        if vol_true == 0 and vol_pred == 0:
            return 1.0  # Cả hai đều trống -> giống nhau hoàn toàn
        if vol_true == 0 or vol_pred == 0:
            return 0.0  # Một trong hai trống -> khác nhau hoàn toàn
        
        # Tính độ tương đồng thể tích
        vol_similarity = 1.0 - abs(vol_true - vol_pred) / (vol_true + vol_pred)
        
        return vol_similarity
    
    def evaluate_segmentation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Đánh giá toàn diện phân đoạn.
        
        Args:
            y_true: Mảng nhị phân chứa phân đoạn chuẩn.
            y_pred: Mảng nhị phân chứa phân đoạn dự đoán.
            
        Returns:
            Từ điển chứa các chỉ số đánh giá.
        """
        # Tính toán các chỉ số
        dice = self.dice_coefficient(y_true, y_pred)
        jaccard = self.jaccard_index(y_true, y_pred)
        hausdorff = self.hausdorff_distance(y_true, y_pred)
        surface_dist = self.average_surface_distance(y_true, y_pred)
        vol_sim = self.volume_similarity(y_true, y_pred)
        
        # Tính độ chính xác, độ nhạy và độ đặc hiệu trên toàn bộ voxel
        y_true_flat = (y_true > 0).flatten()
        y_pred_flat = (y_pred > 0).flatten()
        
        # True Positive, True Negative, False Positive, False Negative
        tp = np.sum(np.logical_and(y_true_flat, y_pred_flat))
        tn = np.sum(np.logical_and(~y_true_flat, ~y_pred_flat))
        fp = np.sum(np.logical_and(~y_true_flat, y_pred_flat))
        fn = np.sum(np.logical_and(y_true_flat, ~y_pred_flat))
        
        # Tính các chỉ số phân loại
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Đóng gói kết quả
        results = {
            'dice': dice,
            'jaccard': jaccard,
            'hausdorff_distance': hausdorff,
            'average_surface_distance': surface_dist,
            'volume_similarity': vol_sim,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        if self.verbose:
            print("Kết quả đánh giá phân đoạn:")
            for metric, value in results.items():
                print(f"  {metric}: {value}")
        
        return results
    
    def evaluate_mesh(self, mesh_true: Dict, mesh_pred: Dict) -> Dict[str, float]:
        """
        Đánh giá độ chính xác của mô hình lưới 3D.
        
        Args:
            mesh_true: Từ điển mô tả lưới chuẩn.
            mesh_pred: Từ điển mô tả lưới dự đoán.
            
        Returns:
            Từ điển chứa các chỉ số đánh giá.
        """
        # Kiểm tra các lưới có hợp lệ không
        if len(mesh_true['vertices']) == 0 or len(mesh_pred['vertices']) == 0:
            return {'error': 'Một trong hai lưới trống'}
        
        # Tính toán các chỉ số lưới cơ bản
        results = {
            'vertex_count_true': len(mesh_true['vertices']),
            'vertex_count_pred': len(mesh_pred['vertices']),
            'face_count_true': len(mesh_true['faces']),
            'face_count_pred': len(mesh_pred['faces']),
            'vertex_count_difference': abs(len(mesh_true['vertices']) - len(mesh_pred['vertices'])),
            'face_count_difference': abs(len(mesh_true['faces']) - len(mesh_pred['faces'])),
        }
        
        # Tính tỷ lệ sai khác về số lượng đỉnh và mặt
        if len(mesh_true['vertices']) > 0:
            results['vertex_difference_ratio'] = results['vertex_count_difference'] / len(mesh_true['vertices'])
        else:
            results['vertex_difference_ratio'] = float('inf')
            
        if len(mesh_true['faces']) > 0:
            results['face_difference_ratio'] = results['face_count_difference'] / len(mesh_true['faces'])
        else:
            results['face_difference_ratio'] = float('inf')
        
        # Đo lường khoảng cách giữa các đỉnh
        # Lưu ý: Đây chỉ là ước lượng đơn giản, các phương pháp chính xác hơn đòi hỏi
        # thuật toán phức tạp như Iterative Closest Point (ICP)
        
        if self.verbose:
            print("Kết quả đánh giá lưới:")
            for metric, value in results.items():
                print(f"  {metric}: {value}")
        
        return results
    
    def compare_volumes(self, vol_true: np.ndarray, vol_pred: np.ndarray, 
                      slice_idx: Optional[int] = None, save_path: Optional[str] = None) -> None:
        """
        So sánh trực quan giữa hai khối dữ liệu qua các lát cắt.
        
        Args:
            vol_true: Khối dữ liệu chuẩn.
            vol_pred: Khối dữ liệu dự đoán.
            slice_idx: Chỉ số lát cắt. Nếu None, chọn lát cắt giữa.
            save_path: Đường dẫn để lưu hình ảnh. Nếu None, hiển thị trực tiếp.
        """
        # Đảm bảo hai khối có cùng kích thước
        if vol_true.shape != vol_pred.shape:
            print(f"Cảnh báo: Hai khối dữ liệu có kích thước khác nhau: {vol_true.shape} vs {vol_pred.shape}")
            # Cắt khối lớn hơn cho phù hợp với kích thước nhỏ hơn
            min_shape = [min(s1, s2) for s1, s2 in zip(vol_true.shape, vol_pred.shape)]
            vol_true = vol_true[:min_shape[0], :min_shape[1], :min_shape[2]]
            vol_pred = vol_pred[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # Chọn lát cắt giữa nếu không chỉ định
        if slice_idx is None:
            slice_idx = vol_true.shape[2] // 2
        
        # Lấy lát cắt từ mỗi khối
        slice_true = vol_true[:, :, slice_idx]
        slice_pred = vol_pred[:, :, slice_idx]
        
        # Tính sự khác biệt
        diff = np.abs(slice_true - slice_pred)
        
        # Tạo hình
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Hiển thị lát cắt từ khối chuẩn
        axes[0].imshow(slice_true, cmap='gray')
        axes[0].set_title('Chuẩn')
        axes[0].axis('off')
        
        # Hiển thị lát cắt từ khối dự đoán
        axes[1].imshow(slice_pred, cmap='gray')
        axes[1].set_title('Dự đoán')
        axes[1].axis('off')
        
        # Hiển thị sự khác biệt
        im = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Sự khác biệt')
        axes[2].axis('off')
        
        # Thêm thanh màu
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            if self.verbose:
                print(f"Đã lưu hình ảnh so sánh tại {save_path}")
        else:
            plt.show()
    
    def calculate_ssim(self, vol_true: np.ndarray, vol_pred: np.ndarray) -> float:
        """
        Tính chỉ số SSIM (Structural Similarity Index) giữa hai khối dữ liệu.
        
        Args:
            vol_true: Khối dữ liệu chuẩn.
            vol_pred: Khối dữ liệu dự đoán.
            
        Returns:
            Chỉ số SSIM trung bình.
        """
        # Đảm bảo hai khối có cùng kích thước
        if vol_true.shape != vol_pred.shape:
            print(f"Cảnh báo: Hai khối dữ liệu có kích thước khác nhau: {vol_true.shape} vs {vol_pred.shape}")
            # Cắt khối lớn hơn cho phù hợp với kích thước nhỏ hơn
            min_shape = [min(s1, s2) for s1, s2 in zip(vol_true.shape, vol_pred.shape)]
            vol_true = vol_true[:min_shape[0], :min_shape[1], :min_shape[2]]
            vol_pred = vol_pred[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # Tính SSIM cho từng lát cắt và lấy trung bình
        ssim_values = []
        for i in range(vol_true.shape[2]):
            slice_true = vol_true[:, :, i]
            slice_pred = vol_pred[:, :, i]
            
            # Chuẩn hóa giá trị pixel về [0, 1]
            slice_true = (slice_true - slice_true.min()) / (slice_true.max() - slice_true.min() + 1e-8)
            slice_pred = (slice_pred - slice_pred.min()) / (slice_pred.max() - slice_pred.min() + 1e-8)
            
            ssim = metrics.structural_similarity(slice_true, slice_pred, data_range=1.0)
            ssim_values.append(ssim)
        
        avg_ssim = np.mean(ssim_values)
        
        if self.verbose:
            print(f"SSIM trung bình: {avg_ssim}")
        
        return avg_ssim
    
    def generate_evaluation_report(self, results: Dict[str, float], 
                                 output_path: Optional[str] = None) -> None:
        """
        Tạo báo cáo đánh giá.
        
        Args:
            results: Từ điển chứa các chỉ số đánh giá.
            output_path: Đường dẫn đến tệp báo cáo. Nếu None, in ra màn hình.
        """
        # Tạo nội dung báo cáo
        report = "# Báo cáo Đánh giá Mô hình 3D\n\n"
        
        # Thêm mục chỉ số phân đoạn
        report += "## Chỉ số Phân đoạn\n\n"
        seg_metrics = ['dice', 'jaccard', 'hausdorff_distance', 'average_surface_distance', 
                      'volume_similarity', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
        
        for metric in seg_metrics:
            if metric in results:
                report += f"- **{metric}**: {results[metric]:.4f}\n"
        
        # Thêm mục ma trận nhầm lẫn nếu có
        if all(key in results for key in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']):
            report += "\n## Ma trận Nhầm lẫn\n\n"
            report += "| | Dự đoán Dương | Dự đoán Âm |\n"
            report += "|---|---|---|\n"
            report += f"| **Thực tế Dương** | TP: {results['true_positives']} | FN: {results['false_negatives']} |\n"
            report += f"| **Thực tế Âm** | FP: {results['false_positives']} | TN: {results['true_negatives']} |\n"
        
        # Thêm mục chỉ số lưới nếu có
        mesh_metrics = ['vertex_count_true', 'vertex_count_pred', 'face_count_true', 'face_count_pred',
                       'vertex_count_difference', 'face_count_difference', 
                       'vertex_difference_ratio', 'face_difference_ratio']
        
        if any(key in results for key in mesh_metrics):
            report += "\n## Chỉ số Lưới\n\n"
            for metric in mesh_metrics:
                if metric in results:
                    report += f"- **{metric}**: {results[metric]:.4f}\n"
        
        # Đầu ra
        if output_path:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            if self.verbose:
                print(f"Đã lưu báo cáo đánh giá tại {output_path}")
        else:
            print(report)


if __name__ == "__main__":
    # Ví dụ cách sử dụng
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    # Tạo dữ liệu mẫu
    def create_sphere(size=64, radius=25, center=None, noise=0.0):
        if center is None:
            center = [size//2, size//2, size//2]
        
        grid = np.mgrid[:size, :size, :size]
        grid = (grid.T - center).T
        dist = np.sqrt(np.sum(grid**2, axis=0))
        
        sphere = (dist <= radius).astype(np.float32)
        
        # Thêm nhiễu
        if noise > 0:
            np.random.seed(42)
            noise_data = np.random.normal(0, noise, sphere.shape)
            sphere = np.clip(sphere + noise_data, 0, 1)
        
        return sphere
    
    # Tạo khối dữ liệu chuẩn
    vol_true = create_sphere(64, 25)
    
    # Tạo khối dữ liệu dự đoán với một số sai khác
    vol_pred = create_sphere(64, 23, center=[32, 33, 31], noise=0.05)
    
    # Tạo thư mục đầu ra
    os.makedirs("../../data/processed", exist_ok=True)
    
    # Tạo đối tượng đánh giá
    evaluator = ModelEvaluator(verbose=True)
    
    # Đánh giá phân đoạn
    results = evaluator.evaluate_segmentation(vol_true, vol_pred)
    
    # Tạo báo cáo
    evaluator.generate_evaluation_report(results, "../../data/processed/evaluation_report.md")
    
    # So sánh trực quan
    evaluator.compare_volumes(vol_true, vol_pred, save_path="../../data/processed/volume_comparison.png")
    
    print("Đã hoàn thành đánh giá mô hình.") 