"""
Module tái tạo mô hình 3D từ dữ liệu phân đoạn.
Sử dụng thuật toán Marching Cubes để tạo mô hình bề mặt.
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class MarchingCubesReconstruction:
    """Lớp tái tạo mô hình 3D sử dụng thuật toán Marching Cubes."""
    
    def __init__(self, verbose: bool = False):
        """
        Khởi tạo đối tượng tái tạo.
        
        Args:
            verbose: Hiển thị thông tin chi tiết nếu True.
        """
        self.verbose = verbose
    
    def generate_mesh(self, volume: np.ndarray, threshold: float = 0.5, 
                    step_size: int = 1, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict:
        """
        Tạo lưới 3D từ dữ liệu thể tích sử dụng thuật toán Marching Cubes.
        
        Args:
            volume: Mảng 3D đầu vào (đã được phân đoạn hoặc chuẩn hóa).
            threshold: Giá trị ngưỡng để xác định bề mặt.
            step_size: Bước nhảy để giảm độ phân giải và tăng tốc độ xử lý.
            spacing: Khoảng cách voxel (x, y, z) để có tỷ lệ chính xác.
            
        Returns:
            Từ điển chứa thông tin lưới (vertices, faces, normals).
        """
        # Giảm kích thước dữ liệu nếu cần
        if step_size > 1:
            volume = volume[::step_size, ::step_size, ::step_size]
            spacing = tuple(s * step_size for s in spacing)
        
        if self.verbose:
            print(f"Tạo lưới với kích thước khối dữ liệu: {volume.shape}")
            print(f"Ngưỡng: {threshold}, Khoảng cách voxel: {spacing}")
        
        try:
            # Áp dụng thuật toán Marching Cubes
            verts, faces, normals, values = measure.marching_cubes(
                volume, 
                level=threshold, 
                spacing=spacing,
                allow_degenerate=False,
                method='lewiner'  # Sử dụng phương pháp Lewiner có thông tin pháp tuyến
            )
            
            mesh = {
                'vertices': verts,
                'faces': faces,
                'normals': normals,
                'values': values
            }
            
            if self.verbose:
                print(f"Đã tạo lưới thành công. Số đỉnh: {len(verts)}, Số mặt: {len(faces)}")
            
            return mesh
        
        except Exception as e:
            print(f"Lỗi khi tạo lưới: {e}")
            # Trả về lưới rỗng nếu có lỗi
            return {
                'vertices': np.array([]),
                'faces': np.array([]),
                'normals': np.array([]),
                'values': np.array([])
            }
    
    def simplify_mesh(self, mesh: Dict, target_reduction: float = 0.5) -> Dict:
        """
        Đơn giản hóa lưới để giảm số lượng mặt.
        
        Args:
            mesh: Từ điển chứa thông tin lưới.
            target_reduction: Tỷ lệ giảm mục tiêu (0.0 đến 1.0).
            
        Returns:
            Từ điển chứa thông tin lưới đã đơn giản hóa.
        """
        # Sử dụng thư viện PyMesh hoặc Open3D để đơn giản hóa lưới
        # Đây là một phiên bản đơn giản chỉ giữ lại một phần của mặt
        try:
            import numpy as np
            
            # Kiểm tra xem lưới có hợp lệ không
            if len(mesh['faces']) == 0:
                return mesh
            
            # Tính số lượng mặt cần giữ lại
            num_faces = len(mesh['faces'])
            keep_faces = max(1, int(num_faces * (1 - target_reduction)))
            
            # Chọn ngẫu nhiên các mặt để giữ lại
            indices = np.random.choice(num_faces, keep_faces, replace=False)
            new_faces = mesh['faces'][indices]
            
            # Tìm các đỉnh được sử dụng
            unique_vertices = np.unique(new_faces.flatten())
            
            # Tạo ánh xạ từ chỉ số đỉnh cũ sang mới
            vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
            
            # Tạo tập hợp đỉnh mới
            new_vertices = mesh['vertices'][unique_vertices]
            
            # Cập nhật chỉ số mặt
            updated_faces = np.zeros_like(new_faces)
            for i, face in enumerate(new_faces):
                for j, vertex_idx in enumerate(face):
                    updated_faces[i, j] = vertex_map[vertex_idx]
            
            # Tạo lưới mới
            simplified_mesh = {
                'vertices': new_vertices,
                'faces': updated_faces,
                'normals': None,  # Cần tính lại pháp tuyến
                'values': None
            }
            
            if self.verbose:
                print(f"Đã đơn giản hóa lưới. Số mặt ban đầu: {num_faces}, Số mặt mới: {keep_faces}")
            
            return simplified_mesh
            
        except Exception as e:
            print(f"Lỗi khi đơn giản hóa lưới: {e}")
            return mesh
    
    def calculate_mesh_properties(self, mesh: Dict) -> Dict:
        """
        Tính toán các thuộc tính của lưới: thể tích, diện tích bề mặt, v.v.
        
        Args:
            mesh: Từ điển chứa thông tin lưới.
            
        Returns:
            Từ điển các thuộc tính của lưới.
        """
        # Kiểm tra lưới hợp lệ
        if len(mesh['vertices']) == 0 or len(mesh['faces']) == 0:
            return {'volume': 0, 'surface_area': 0, 'is_watertight': False}
        
        # Tính diện tích bề mặt
        def triangle_area(triangle):
            # Tính diện tích tam giác sử dụng tích chéo
            a = triangle[1] - triangle[0]
            b = triangle[2] - triangle[0]
            return 0.5 * np.linalg.norm(np.cross(a, b))
        
        # Lấy các đỉnh của từng tam giác
        triangles = mesh['vertices'][mesh['faces']]
        
        # Tính tổng diện tích của các tam giác
        surface_area = sum(triangle_area(triangle) for triangle in triangles)
        
        # Kiểm tra xem lưới có kín nước không (đơn giản)
        # Trong một lưới kín nước, mỗi cạnh kết nối chính xác hai mặt
        is_watertight = False  # Thực tế cần kiểm tra chi tiết hơn
        
        # Tính thể tích xấp xỉ (nếu lưới kín nước)
        # Thể tích xấp xỉ = 1/6 * tổng(dot(v_i, n_i) * area_i)
        volume = 0.0  # Thực tế cần tính toán chi tiết hơn
        
        properties = {
            'volume': volume,
            'surface_area': surface_area,
            'is_watertight': is_watertight,
            'num_vertices': len(mesh['vertices']),
            'num_faces': len(mesh['faces'])
        }
        
        return properties
    
    def visualize_mesh(self, mesh: Dict, color: Tuple[float, float, float] = (0.7, 0.7, 0.7),
                     alpha: float = 1.0, figsize: Tuple[int, int] = (10, 10),
                     save_path: Optional[str] = None) -> None:
        """
        Trực quan hóa lưới 3D.
        
        Args:
            mesh: Từ điển chứa thông tin lưới.
            color: Màu sắc của lưới (R, G, B).
            alpha: Độ trong suốt (0.0 đến 1.0).
            figsize: Kích thước hình ảnh.
            save_path: Đường dẫn để lưu hình ảnh. Nếu None, hiển thị trực tiếp.
        """
        # Kiểm tra lưới hợp lệ
        if len(mesh['vertices']) == 0 or len(mesh['faces']) == 0:
            print("Không thể trực quan hóa: Lưới trống.")
            return
        
        # Tạo hình
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Tạo tập hợp tam giác từ lưới
        triangles = mesh['vertices'][mesh['faces']]
        
        # Tạo đối tượng 3D
        mesh_collection = Poly3DCollection(triangles, alpha=alpha)
        mesh_collection.set_facecolor(color)
        mesh_collection.set_edgecolor('k')
        
        # Thêm vào trục
        ax.add_collection3d(mesh_collection)
        
        # Tự động điều chỉnh tỷ lệ
        vertices = mesh['vertices']
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        
        # Đặt giới hạn trục
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        # Đặt nhãn
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Đặt tiêu đề
        ax.set_title(f'Mô hình 3D - {len(mesh["vertices"])} đỉnh, {len(mesh["faces"])} mặt')
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            if self.verbose:
                print(f"Đã lưu hình ảnh tại {save_path}")
        else:
            plt.show()
    
    def save_mesh_as_obj(self, mesh: Dict, output_path: str) -> bool:
        """
        Lưu lưới dưới dạng tệp OBJ.
        
        Args:
            mesh: Từ điển chứa thông tin lưới.
            output_path: Đường dẫn để lưu tệp OBJ.
            
        Returns:
            True nếu lưu thành công, False nếu không.
        """
        # Kiểm tra lưới hợp lệ
        if len(mesh['vertices']) == 0 or len(mesh['faces']) == 0:
            print("Không thể lưu: Lưới trống.")
            return False
        
        try:
            # Tạo thư mục nếu không tồn tại
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                # Ghi đỉnh
                for vertex in mesh['vertices']:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                
                # Ghi pháp tuyến nếu có
                if mesh['normals'] is not None and len(mesh['normals']) > 0:
                    for normal in mesh['normals']:
                        f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
                
                # Ghi mặt (OBJ sử dụng chỉ số 1-indexed)
                for face in mesh['faces']:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            if self.verbose:
                print(f"Đã lưu lưới tại {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Lỗi khi lưu lưới: {e}")
            return False
    
    def save_mesh_as_stl(self, mesh: Dict, output_path: str) -> bool:
        """
        Lưu lưới dưới dạng tệp STL.
        
        Args:
            mesh: Từ điển chứa thông tin lưới.
            output_path: Đường dẫn để lưu tệp STL.
            
        Returns:
            True nếu lưu thành công, False nếu không.
        """
        try:
            from stl import mesh as stl_mesh
            
            # Kiểm tra lưới hợp lệ
            if len(mesh['vertices']) == 0 or len(mesh['faces']) == 0:
                print("Không thể lưu: Lưới trống.")
                return False
            
            # Tạo đối tượng STL
            stl_data = stl_mesh.Mesh(np.zeros(len(mesh['faces']), dtype=stl_mesh.Mesh.dtype))
            
            # Lấy các đỉnh của từng tam giác
            triangles = mesh['vertices'][mesh['faces']]
            
            # Đặt các đỉnh
            for i, triangle in enumerate(triangles):
                for j in range(3):
                    stl_data.vectors[i][j] = triangle[j]
            
            # Tạo thư mục nếu cần
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Lưu tệp STL
            stl_data.save(output_path)
            
            if self.verbose:
                print(f"Đã lưu lưới tại {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Lỗi khi lưu lưới: {e}")
            print("Đảm bảo đã cài đặt thư viện numpy-stl: pip install numpy-stl")
            return False


if __name__ == "__main__":
    # Ví dụ cách sử dụng
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.data_processing.nifti_loader import NiftiLoader
    from src.segmentation.threshold_segmentation import ThresholdSegmentation
    
    # Tạo dữ liệu mẫu
    print("Tạo dữ liệu mẫu...")
    
    # Tạo một khối hình cầu giả để minh họa
    def create_sphere(size=64, radius=25, center=None):
        if center is None:
            center = [size//2, size//2, size//2]
        
        grid = np.mgrid[:size, :size, :size]
        grid = (grid.T - center).T
        dist = np.sqrt(np.sum(grid**2, axis=0))
        
        return (dist <= radius).astype(np.float32)
    
    # Tạo một khối dữ liệu mẫu
    volume = create_sphere(64, 25)
    
    # Thêm một số nhiễu
    np.random.seed(42)
    volume += np.random.normal(0, 0.05, volume.shape)
    volume = np.clip(volume, 0, 1)
    
    # Tạo thư mục đầu ra
    os.makedirs("../../data/processed", exist_ok=True)
    os.makedirs("../../models", exist_ok=True)
    
    # Phân đoạn khối dữ liệu
    segmenter = ThresholdSegmentation(verbose=True)
    segmented_volume = segmenter.binary_threshold(volume, threshold=0.5)
    
    # Tái tạo lưới 3D
    reconstructor = MarchingCubesReconstruction(verbose=True)
    
    # Tạo lưới
    mesh = reconstructor.generate_mesh(segmented_volume, threshold=0.5)
    
    # Trực quan hóa lưới
    reconstructor.visualize_mesh(
        mesh,
        color=(0.7, 0.3, 0.3),
        save_path="../../data/processed/marching_cubes_example.png"
    )
    
    # Lưu lưới
    reconstructor.save_mesh_as_obj(mesh, "../../models/sphere_example.obj")
    
    # In thuộc tính lưới
    properties = reconstructor.calculate_mesh_properties(mesh)
    print("Thuộc tính lưới:")
    for key, value in properties.items():
        print(f"  {key}: {value}") 