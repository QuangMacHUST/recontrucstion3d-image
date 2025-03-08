"""
Module trực quan hóa 3D sử dụng VTK.
Cung cấp giao diện để hiển thị, tương tác và phân tích mô hình 3D.
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Callable
import vtk


class VTKVisualizer:
    """Lớp trực quan hóa 3D sử dụng VTK."""
    
    def __init__(self, background_color: Tuple[float, float, float] = (0.1, 0.1, 0.2),
               window_size: Tuple[int, int] = (800, 600)):
        """
        Khởi tạo trình trực quan hóa VTK.
        
        Args:
            background_color: Màu nền (R, G, B), mỗi giá trị từ 0.0 đến 1.0.
            window_size: Kích thước cửa sổ (width, height).
        """
        # Tạo renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*background_color)
        
        # Tạo cửa sổ render
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(*window_size)
        
        # Tạo interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        # Thêm style tương tác
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # Danh sách các actor
        self.actors = []
        
        # Danh sách callback
        self.callbacks = []
    
    def add_volume_data(self, volume: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                      color_map: str = 'grayscale', opacity_map: Optional[np.ndarray] = None) -> None:
        """
        Thêm dữ liệu thể tích để trực quan hóa.
        
        Args:
            volume: Mảng 3D chứa dữ liệu thể tích.
            spacing: Khoảng cách voxel (x, y, z).
            color_map: Bảng màu ('grayscale', 'rainbow', 'hot', 'bone').
            opacity_map: Mảng ánh xạ độ trong suốt (0.0 đến 1.0).
        """
        # Tạo nguồn dữ liệu VTK từ numpy array
        data_importer = vtk.vtkImageImport()
        
        # Chuyển đổi dữ liệu
        data_string = volume.astype(np.float32).tobytes()
        data_importer.CopyImportVoidPointer(data_string, len(data_string))
        data_importer.SetDataScalarTypeToFloat()
        data_importer.SetNumberOfScalarComponents(1)
        
        # Đặt kích thước
        extent = volume.shape
        data_importer.SetDataExtent(0, extent[0] - 1, 0, extent[1] - 1, 0, extent[2] - 1)
        data_importer.SetWholeExtent(0, extent[0] - 1, 0, extent[1] - 1, 0, extent[2] - 1)
        data_importer.SetDataSpacing(*spacing)
        
        # Tạo map màu và độ trong suốt
        color_func = vtk.vtkColorTransferFunction()
        opacity_func = vtk.vtkPiecewiseFunction()
        
        # Thiết lập bảng màu
        if color_map == 'grayscale':
            color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(1.0, 1.0, 1.0, 1.0)
        elif color_map == 'rainbow':
            color_func.AddRGBPoint(0.0, 0.0, 0.0, 1.0)  # Blue
            color_func.AddRGBPoint(0.33, 0.0, 1.0, 1.0)  # Cyan
            color_func.AddRGBPoint(0.5, 0.0, 1.0, 0.0)  # Green
            color_func.AddRGBPoint(0.66, 1.0, 1.0, 0.0)  # Yellow
            color_func.AddRGBPoint(1.0, 1.0, 0.0, 0.0)  # Red
        elif color_map == 'hot':
            color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(0.33, 1.0, 0.0, 0.0)  # Red
            color_func.AddRGBPoint(0.66, 1.0, 1.0, 0.0)  # Yellow
            color_func.AddRGBPoint(1.0, 1.0, 1.0, 1.0)  # White
        elif color_map == 'bone':
            color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(0.33, 0.2, 0.2, 0.3)
            color_func.AddRGBPoint(0.66, 0.4, 0.5, 0.6)
            color_func.AddRGBPoint(1.0, 1.0, 1.0, 1.0)
        else:
            # Mặc định grayscale
            color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(1.0, 1.0, 1.0, 1.0)
        
        # Thiết lập độ trong suốt
        if opacity_map is not None:
            # Sử dụng bản đồ độ trong suốt tùy chỉnh
            for i, opacity in enumerate(opacity_map):
                val = i / (len(opacity_map) - 1)
                opacity_func.AddPoint(val, opacity)
        else:
            # Sử dụng độ trong suốt mặc định
            opacity_func.AddPoint(0.0, 0.0)
            opacity_func.AddPoint(0.2, 0.0)
            opacity_func.AddPoint(0.5, 0.1)
            opacity_func.AddPoint(0.8, 0.2)
            opacity_func.AddPoint(1.0, 0.3)
        
        # Tạo thuộc tính âm lượng
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        
        # Tạo mapper
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputConnection(data_importer.GetOutputPort())
        
        # Tạo actor âm lượng
        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(mapper)
        volume_actor.SetProperty(volume_property)
        
        # Thêm actor
        self.renderer.AddVolume(volume_actor)
        self.actors.append(volume_actor)
    
    def add_mesh(self, vertices: np.ndarray, faces: np.ndarray, 
               color: Tuple[float, float, float] = (0.8, 0.8, 0.9),
               opacity: float = 1.0, wireframe: bool = False) -> None:
        """
        Thêm lưới 3D để trực quan hóa.
        
        Args:
            vertices: Mảng đỉnh (n_vertices, 3).
            faces: Mảng mặt (n_faces, 3).
            color: Màu sắc của lưới (R, G, B).
            opacity: Độ trong suốt (0.0 đến 1.0).
            wireframe: Hiển thị dạng khung dây nếu True.
        """
        # Tạo points
        points = vtk.vtkPoints()
        for vertex in vertices:
            points.InsertNextPoint(*vertex)
        
        # Tạo triangles
        triangles = vtk.vtkCellArray()
        for face in faces:
            triangle = vtk.vtkTriangle()
            for i in range(3):
                triangle.GetPointIds().SetId(i, face[i])
            triangles.InsertNextCell(triangle)
        
        # Tạo polydata
        mesh = vtk.vtkPolyData()
        mesh.SetPoints(points)
        mesh.SetPolys(triangles)
        
        # Tính pháp tuyến
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(mesh)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.SplittingOff()
        normals.Update()
        
        # Tạo mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        
        # Tạo actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Thiết lập thuộc tính
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        
        if wireframe:
            actor.GetProperty().SetRepresentationToWireframe()
        else:
            actor.GetProperty().SetRepresentationToSurface()
        
        # Thêm actor
        self.renderer.AddActor(actor)
        self.actors.append(actor)
    
    def add_obj_file(self, obj_path: str, color: Tuple[float, float, float] = (0.8, 0.8, 0.9),
                   opacity: float = 1.0, wireframe: bool = False) -> None:
        """
        Tải và thêm tệp OBJ.
        
        Args:
            obj_path: Đường dẫn đến tệp OBJ.
            color: Màu sắc của lưới (R, G, B).
            opacity: Độ trong suốt (0.0 đến 1.0).
            wireframe: Hiển thị dạng khung dây nếu True.
        """
        if not os.path.exists(obj_path):
            print(f"Không tìm thấy tệp OBJ: {obj_path}")
            return
        
        # Tạo OBJ reader
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_path)
        reader.Update()
        
        # Tính pháp tuyến
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(reader.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.SplittingOff()
        normals.Update()
        
        # Tạo mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        
        # Tạo actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Thiết lập thuộc tính
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        
        if wireframe:
            actor.GetProperty().SetRepresentationToWireframe()
        else:
            actor.GetProperty().SetRepresentationToSurface()
        
        # Thêm actor
        self.renderer.AddActor(actor)
        self.actors.append(actor)
    
    def add_stl_file(self, stl_path: str, color: Tuple[float, float, float] = (0.8, 0.8, 0.9),
                   opacity: float = 1.0, wireframe: bool = False) -> None:
        """
        Tải và thêm tệp STL.
        
        Args:
            stl_path: Đường dẫn đến tệp STL.
            color: Màu sắc của lưới (R, G, B).
            opacity: Độ trong suốt (0.0 đến 1.0).
            wireframe: Hiển thị dạng khung dây nếu True.
        """
        if not os.path.exists(stl_path):
            print(f"Không tìm thấy tệp STL: {stl_path}")
            return
        
        # Tạo STL reader
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()
        
        # Tính pháp tuyến
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(reader.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.SplittingOff()
        normals.Update()
        
        # Tạo mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        
        # Tạo actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Thiết lập thuộc tính
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        
        if wireframe:
            actor.GetProperty().SetRepresentationToWireframe()
        else:
            actor.GetProperty().SetRepresentationToSurface()
        
        # Thêm actor
        self.renderer.AddActor(actor)
        self.actors.append(actor)
    
    def add_slice_planes(self, volume: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        """
        Thêm mặt phẳng cắt cho dữ liệu thể tích.
        
        Args:
            volume: Mảng 3D chứa dữ liệu thể tích.
            spacing: Khoảng cách voxel (x, y, z).
        """
        # Tạo nguồn dữ liệu VTK từ numpy array
        data_importer = vtk.vtkImageImport()
        
        # Chuyển đổi dữ liệu
        data_string = volume.astype(np.float32).tobytes()
        data_importer.CopyImportVoidPointer(data_string, len(data_string))
        data_importer.SetDataScalarTypeToFloat()
        data_importer.SetNumberOfScalarComponents(1)
        
        # Đặt kích thước
        extent = volume.shape
        data_importer.SetDataExtent(0, extent[0] - 1, 0, extent[1] - 1, 0, extent[2] - 1)
        data_importer.SetWholeExtent(0, extent[0] - 1, 0, extent[1] - 1, 0, extent[2] - 1)
        data_importer.SetDataSpacing(*spacing)
        data_importer.Update()
        
        # Tạo mặt phẳng cắt trục X
        plane_x = vtk.vtkImagePlaneWidget()
        plane_x.SetInteractor(self.interactor)
        plane_x.SetInputData(data_importer.GetOutput())
        plane_x.SetPlaneOrientationToXAxes()
        plane_x.SetSliceIndex(extent[0] // 2)
        plane_x.SetMarginSizeX(0)
        plane_x.SetMarginSizeY(0)
        plane_x.DisplayTextOn()
        plane_x.EnabledOn()
        
        # Tạo mặt phẳng cắt trục Y
        plane_y = vtk.vtkImagePlaneWidget()
        plane_y.SetInteractor(self.interactor)
        plane_y.SetInputData(data_importer.GetOutput())
        plane_y.SetPlaneOrientationToYAxes()
        plane_y.SetSliceIndex(extent[1] // 2)
        plane_y.SetMarginSizeX(0)
        plane_y.SetMarginSizeY(0)
        plane_y.DisplayTextOn()
        plane_y.EnabledOn()
        
        # Tạo mặt phẳng cắt trục Z
        plane_z = vtk.vtkImagePlaneWidget()
        plane_z.SetInteractor(self.interactor)
        plane_z.SetInputData(data_importer.GetOutput())
        plane_z.SetPlaneOrientationToZAxes()
        plane_z.SetSliceIndex(extent[2] // 2)
        plane_z.SetMarginSizeX(0)
        plane_z.SetMarginSizeY(0)
        plane_z.DisplayTextOn()
        plane_z.EnabledOn()
    
    def add_axes(self, length: float = 100.0) -> None:
        """
        Thêm trục tọa độ vào cảnh.
        
        Args:
            length: Độ dài của các trục.
        """
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(length, length, length)
        axes.SetShaftTypeToCylinder()
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")
        
        self.renderer.AddActor(axes)
        self.actors.append(axes)
    
    def add_text(self, text: str, position: Tuple[float, float] = (0.05, 0.95),
               color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
               font_size: int = 18) -> None:
        """
        Thêm văn bản vào cảnh.
        
        Args:
            text: Nội dung văn bản.
            position: Vị trí (x, y) trong cửa sổ (0.0 đến 1.0).
            color: Màu sắc của văn bản (R, G, B).
            font_size: Kích thước font.
        """
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(text)
        text_actor.GetTextProperty().SetFontSize(font_size)
        text_actor.GetTextProperty().SetColor(*color)
        text_actor.SetPosition(position[0] * self.render_window.GetSize()[0],
                            position[1] * self.render_window.GetSize()[1])
        
        self.renderer.AddActor2D(text_actor)
        self.actors.append(text_actor)
    
    def add_scalar_bar(self, title: str = "Giá trị", 
                     position: Tuple[float, float] = (0.85, 0.1),
                     size: Tuple[float, float] = (0.1, 0.8)) -> None:
        """
        Thêm thanh thang màu vào cảnh.
        
        Args:
            title: Tiêu đề của thanh thang màu.
            position: Vị trí (x, y) trong cửa sổ (0.0 đến 1.0).
            size: Kích thước (width, height) của thanh thang màu (0.0 đến 1.0).
        """
        # Tạo thanh thang màu
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetTitle(title)
        scalar_bar.SetNumberOfLabels(5)
        
        # Lấy mapper của actor đầu tiên (giả sử là volume hoặc mesh)
        if len(self.actors) > 0:
            actor = self.actors[0]
            if hasattr(actor, 'GetMapper'):
                mapper = actor.GetMapper()
                if hasattr(mapper, 'GetLookupTable'):
                    lut = mapper.GetLookupTable()
                    scalar_bar.SetLookupTable(lut)
        
        # Thiết lập vị trí và kích thước
        scalar_bar.SetPosition(position[0], position[1])
        scalar_bar.SetPosition2(size[0], size[1])
        
        # Thiết lập thuộc tính
        scalar_bar.GetLabelTextProperty().SetColor(1.0, 1.0, 1.0)
        scalar_bar.GetTitleTextProperty().SetColor(1.0, 1.0, 1.0)
        
        # Thêm vào renderer
        self.renderer.AddActor2D(scalar_bar)
        self.actors.append(scalar_bar)
    
    def add_callback(self, callback: Callable) -> None:
        """
        Thêm hàm callback được gọi trong mỗi vòng lặp render.
        
        Args:
            callback: Hàm callback.
        """
        observer_id = self.interactor.AddObserver('TimerEvent', callback)
        self.callbacks.append(observer_id)
    
    def start_interaction(self) -> None:
        """Bắt đầu tương tác với cảnh."""
        self.interactor.Initialize()
        self.interactor.CreateRepeatingTimer(10)  # 10ms timer
        self.render_window.Render()
        self.interactor.Start()
    
    def capture_screenshot(self, file_path: str, magnification: int = 1) -> None:
        """
        Chụp ảnh màn hình của cảnh.
        
        Args:
            file_path: Đường dẫn đến tệp đầu ra.
            magnification: Hệ số phóng đại cho độ phân giải cao hơn.
        """
        # Tạo thư mục nếu không tồn tại
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Render cảnh
        self.render_window.Render()
        
        # Chụp ảnh màn hình
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.render_window)
        window_to_image.SetScale(magnification)
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()
        window_to_image.Update()
        
        # Xác định định dạng đầu ra dựa trên phần mở rộng tệp
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.png':
            writer = vtk.vtkPNGWriter()
        elif file_extension == '.jpg' or file_extension == '.jpeg':
            writer = vtk.vtkJPEGWriter()
        elif file_extension == '.bmp':
            writer = vtk.vtkBMPWriter()
        elif file_extension == '.tif' or file_extension == '.tiff':
            writer = vtk.vtkTIFFWriter()
        else:
            print(f"Không hỗ trợ định dạng {file_extension}. Sử dụng PNG.")
            writer = vtk.vtkPNGWriter()
            file_path = os.path.splitext(file_path)[0] + '.png'
        
        # Lưu ảnh
        writer.SetFileName(file_path)
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()
        
        print(f"Đã lưu ảnh màn hình tại {file_path}")
    
    def create_animation(self, output_path: str, n_frames: int = 60) -> None:
        """
        Tạo animation bằng cách xoay mô hình và chụp từng khung hình.
        
        Args:
            output_path: Thư mục đầu ra cho các khung hình.
            n_frames: Số lượng khung hình.
        """
        # Tạo thư mục đầu ra
        os.makedirs(output_path, exist_ok=True)
        
        # Lưu camera ban đầu
        camera = self.renderer.GetActiveCamera()
        initial_position = camera.GetPosition()
        initial_focal_point = camera.GetFocalPoint()
        initial_view_up = camera.GetViewUp()
        
        # Render các khung hình
        for i in range(n_frames):
            # Tính góc xoay
            angle = i * 360.0 / n_frames
            
            # Đặt camera
            camera.SetPosition(initial_position)
            camera.SetFocalPoint(initial_focal_point)
            camera.SetViewUp(initial_view_up)
            
            # Xoay quanh trục y
            camera.Azimuth(angle)
            
            # Render và lưu khung hình
            frame_path = os.path.join(output_path, f"frame_{i:04d}.png")
            self.capture_screenshot(frame_path)
        
        print(f"Đã tạo {n_frames} khung hình tại {output_path}")
        print("Bạn có thể kết hợp chúng thành video sử dụng ffmpeg:")
        print(f"ffmpeg -framerate 30 -i {output_path}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")


if __name__ == "__main__":
    # Ví dụ cách sử dụng
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.reconstruction.marching_cubes import MarchingCubesReconstruction
    
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
    
    # Tạo hai khối dữ liệu mẫu khác nhau
    volume1 = create_sphere(64, 25)
    volume2 = create_sphere(64, 15, center=[20, 20, 20])
    
    # Kết hợp hai khối
    volume = np.maximum(volume1, volume2)
    
    # Thêm một số nhiễu
    np.random.seed(42)
    volume += np.random.normal(0, 0.05, volume.shape)
    volume = np.clip(volume, 0, 1)
    
    # Tạo thư mục đầu ra
    os.makedirs("../../data/processed", exist_ok=True)
    
    # Tái tạo lưới 3D
    reconstructor = MarchingCubesReconstruction(verbose=True)
    mesh = reconstructor.generate_mesh(volume, threshold=0.5)
    
    # Tạo trình trực quan hóa VTK
    visualizer = VTKVisualizer(background_color=(0.2, 0.3, 0.4))
    
    # Thêm lưới
    visualizer.add_mesh(mesh['vertices'], mesh['faces'], color=(1.0, 0.7, 0.7))
    
    # Thêm phần nguyên bản dưới dạng khung dây
    visualizer.add_mesh(mesh['vertices'], mesh['faces'], color=(1.0, 1.0, 1.0), 
                      opacity=0.1, wireframe=True)
    
    # Thêm trục tọa độ
    visualizer.add_axes()
    
    # Thêm văn bản
    visualizer.add_text("Mô hình 3D Mẫu", position=(0.05, 0.95))
    
    # Chụp ảnh màn hình
    visualizer.capture_screenshot("../../data/processed/vtk_example.png")
    
    # Tạo animation
    visualizer.create_animation("../../data/processed/animation", n_frames=30)
    
    # Bắt đầu tương tác
    print("Đang khởi động trình trực quan hóa VTK...")
    visualizer.start_interaction() 