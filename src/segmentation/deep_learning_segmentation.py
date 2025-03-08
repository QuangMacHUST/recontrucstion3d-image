"""
Module phân đoạn hình ảnh sử dụng deep learning.
Triển khai mô hình U-Net cho phân đoạn hình ảnh y tế 3D.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from typing import Tuple, List, Optional, Dict, Union
import matplotlib.pyplot as plt


class UNet3D:
    """Lớp triển khai mô hình U-Net 3D cho phân đoạn hình ảnh y tế."""
    
    def __init__(self, 
                input_shape: Tuple[int, int, int, int],
                n_classes: int = 1,
                base_filters: int = 16,
                depth: int = 4,
                verbose: bool = False):
        """
        Khởi tạo mô hình U-Net 3D.
        
        Args:
            input_shape: Kích thước đầu vào (height, width, depth, channels).
            n_classes: Số lớp đầu ra (1 cho phân đoạn nhị phân).
            base_filters: Số bộ lọc cơ sở cho lớp đầu tiên.
            depth: Độ sâu của mô hình U-Net.
            verbose: Hiển thị thông tin chi tiết nếu True.
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.depth = depth
        self.verbose = verbose
        self.model = None
        self._build_model()
    
    def _conv_block(self, inputs, filters, kernel_size=(3, 3, 3), activation='relu', padding='same'):
        """Khối tích chập cơ bản."""
        x = layers.Conv3D(filters, kernel_size, padding=padding)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        
        x = layers.Conv3D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        
        return x
    
    def _build_model(self):
        """Xây dựng kiến trúc mô hình U-Net 3D."""
        inputs = layers.Input(self.input_shape)
        
        # Danh sách để lưu các hoạt động tích chập cho kết nối bỏ qua
        skip_connections = []
        
        # Đường đi thu nhỏ (encoder)
        x = inputs
        for i in range(self.depth):
            x = self._conv_block(x, self.base_filters * (2 ** i))
            skip_connections.append(x)
            if i < self.depth - 1:  # Không thực hiện pooling ở đáy
                x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        
        # Đường đi tăng kích thước (decoder)
        for i in range(self.depth - 2, -1, -1):
            x = layers.Conv3DTranspose(self.base_filters * (2 ** i), (2, 2, 2), strides=(2, 2, 2), padding='same')(x)
            x = layers.concatenate([x, skip_connections[i]])
            x = self._conv_block(x, self.base_filters * (2 ** i))
        
        # Lớp đầu ra
        if self.n_classes == 1:
            outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(x)
        else:
            outputs = layers.Conv3D(self.n_classes, (1, 1, 1), activation='softmax')(x)
        
        # Tạo mô hình
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        if self.verbose:
            print(self.model.summary())
    
    def compile_model(self, learning_rate: float = 1e-4, loss: str = 'binary_crossentropy'):
        """
        Biên dịch mô hình.
        
        Args:
            learning_rate: Tốc độ học.
            loss: Hàm mất mát ('binary_crossentropy' cho phân lớp nhị phân,
                  'categorical_crossentropy' cho phân lớp nhiều lớp).
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được khởi tạo")
        
        # Định nghĩa hàm đo đạc
        metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        
        # Thêm Dice coefficient nếu phân đoạn nhị phân
        if self.n_classes == 1:
            def dice_coefficient(y_true, y_pred, smooth=1.0):
                y_true_f = tf.reshape(y_true, [-1])
                y_pred_f = tf.reshape(y_pred, [-1])
                intersection = tf.reduce_sum(y_true_f * y_pred_f)
                return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
            
            metrics.append(dice_coefficient)
        
        # Biên dịch mô hình
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            batch_size: int = 8, epochs: int = 50, 
            checkpoint_path: Optional[str] = None) -> Dict:
        """
        Huấn luyện mô hình.
        
        Args:
            X_train: Dữ liệu huấn luyện.
            y_train: Nhãn huấn luyện.
            validation_data: Dữ liệu kiểm chứng (X_val, y_val).
            batch_size: Kích thước batch.
            epochs: Số epoch huấn luyện.
            checkpoint_path: Đường dẫn để lưu mô hình tốt nhất.
            
        Returns:
            Lịch sử huấn luyện.
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được biên dịch")
        
        callbacks = []
        
        # Thêm callback checkpoint nếu đường dẫn được cung cấp
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss' if validation_data else 'loss',
                save_best_only=True,
                mode='min',
                verbose=self.verbose
            )
            callbacks.append(checkpoint)
        
        # Thêm early stopping để tránh overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=self.verbose
        )
        callbacks.append(early_stopping)
        
        # Huấn luyện mô hình
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        return history.history
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Dự đoán phân đoạn.
        
        Args:
            X: Dữ liệu đầu vào.
            threshold: Ngưỡng để chuyển đổi xác suất thành nhãn nhị phân.
            
        Returns:
            Kết quả phân đoạn.
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện")
        
        predictions = self.model.predict(X)
        
        # Áp dụng ngưỡng cho phân đoạn nhị phân
        if self.n_classes == 1:
            return (predictions > threshold).astype(np.int8)
        
        # Chọn lớp có xác suất cao nhất cho phân đoạn nhiều lớp
        return np.argmax(predictions, axis=-1).astype(np.int8)
    
    def load_weights(self, weights_path: str) -> None:
        """
        Tải trọng số đã lưu.
        
        Args:
            weights_path: Đường dẫn đến tệp trọng số.
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được khởi tạo")
            
        self.model.load_weights(weights_path)
        if self.verbose:
            print(f"Đã tải trọng số từ {weights_path}")
    
    def visualize_results(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, 
                        slice_idx: Optional[int] = None, save_path: Optional[str] = None) -> None:
        """
        Trực quan hóa kết quả phân đoạn.
        
        Args:
            X: Dữ liệu gốc.
            y_true: Nhãn thật.
            y_pred: Nhãn dự đoán.
            slice_idx: Chỉ số lát cắt. Nếu None, sử dụng lát cắt giữa.
            save_path: Đường dẫn để lưu hình. Nếu None, hiển thị trực tiếp.
        """
        if X.ndim == 5:  # (batch, height, width, depth, channels)
            X = X[0, ..., 0]  # Lấy mẫu đầu tiên và kênh đầu tiên
            y_true = y_true[0, ..., 0] if y_true.ndim == 5 else y_true[0, ...]
            y_pred = y_pred[0, ..., 0] if y_pred.ndim == 5 else y_pred[0, ...]
        elif X.ndim == 4:  # (height, width, depth, channels)
            X = X[..., 0]  # Lấy kênh đầu tiên
            y_true = y_true[..., 0] if y_true.ndim == 4 else y_true
            y_pred = y_pred[..., 0] if y_pred.ndim == 4 else y_pred
        
        # Lấy lát cắt giữa nếu không chỉ định
        if slice_idx is None:
            slice_idx = X.shape[2] // 2
        
        # Lấy dữ liệu lát cắt
        x_slice = X[:, :, slice_idx]
        y_true_slice = y_true[:, :, slice_idx]
        y_pred_slice = y_pred[:, :, slice_idx]
        
        # Tạo hình ảnh
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Hiển thị lát cắt gốc
        axes[0].imshow(x_slice, cmap='gray')
        axes[0].set_title('Đầu vào')
        axes[0].axis('off')
        
        # Hiển thị nhãn thật
        axes[1].imshow(x_slice, cmap='gray')
        axes[1].imshow(y_true_slice, cmap='hot', alpha=0.3)
        axes[1].set_title('Nhãn thật')
        axes[1].axis('off')
        
        # Hiển thị kết quả dự đoán
        axes[2].imshow(x_slice, cmap='gray')
        axes[2].imshow(y_pred_slice, cmap='hot', alpha=0.3)
        axes[2].set_title('Dự đoán')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            if self.verbose:
                print(f"Đã lưu hình ảnh tại {save_path}")
        else:
            plt.show()
    
    def visualize_training_history(self, history: Dict, save_path: Optional[str] = None) -> None:
        """
        Trực quan hóa lịch sử huấn luyện.
        
        Args:
            history: Từ điển chứa lịch sử huấn luyện.
            save_path: Đường dẫn để lưu hình. Nếu None, hiển thị trực tiếp.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Vẽ đồ thị mất mát
        axes[0].plot(history['loss'], label='Huấn luyện')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Kiểm chứng')
        axes[0].set_title('Mất mát')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Mất mát')
        axes[0].legend()
        
        # Vẽ đồ thị độ chính xác
        axes[1].plot(history['accuracy'], label='Huấn luyện')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Kiểm chứng')
        axes[1].set_title('Độ chính xác')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Độ chính xác')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            if self.verbose:
                print(f"Đã lưu hình ảnh tại {save_path}")
        else:
            plt.show()


def preprocess_data(volume: np.ndarray, mask: Optional[np.ndarray] = None, 
                   patch_size: Tuple[int, int, int] = (64, 64, 64),
                   stride: Tuple[int, int, int] = (32, 32, 32)) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Tiền xử lý dữ liệu và chia thành các patch nhỏ.
    
    Args:
        volume: Khối dữ liệu 3D đầu vào.
        mask: Mặt nạ phân đoạn (tùy chọn).
        patch_size: Kích thước patch.
        stride: Bước nhảy khi tạo patch.
        
    Returns:
        Tuple của patches đầu vào và patches mặt nạ (nếu có).
    """
    # Đảm bảo volume và mask có đúng kích thước
    if volume.ndim == 3:
        volume = volume[..., np.newaxis]  # Thêm kênh
    
    if mask is not None and mask.ndim == 3:
        mask = mask[..., np.newaxis]
    
    # Lấy kích thước
    h, w, d = volume.shape[:3]
    
    # Danh sách để lưu các patch
    patches_volume = []
    patches_mask = [] if mask is not None else None
    
    # Tạo các patch
    for z in range(0, d - patch_size[2] + 1, stride[2]):
        for y in range(0, h - patch_size[0] + 1, stride[0]):
            for x in range(0, w - patch_size[1] + 1, stride[1]):
                # Lấy patch từ volume
                patch_vol = volume[y:y+patch_size[0], x:x+patch_size[1], z:z+patch_size[2], :]
                patches_volume.append(patch_vol)
                
                # Lấy patch từ mask nếu có
                if mask is not None:
                    patch_mask = mask[y:y+patch_size[0], x:x+patch_size[1], z:z+patch_size[2], :]
                    patches_mask.append(patch_mask)
    
    # Chuyển thành mảng numpy
    patches_volume = np.array(patches_volume)
    if mask is not None:
        patches_mask = np.array(patches_mask)
    
    return patches_volume, patches_mask


def reconstruct_volume(patches: np.ndarray, original_shape: Tuple[int, int, int],
                     patch_size: Tuple[int, int, int] = (64, 64, 64),
                     stride: Tuple[int, int, int] = (32, 32, 32)) -> np.ndarray:
    """
    Tái tạo khối dữ liệu từ các patch.
    
    Args:
        patches: Mảng các patch dự đoán.
        original_shape: Kích thước gốc của khối dữ liệu.
        patch_size: Kích thước patch.
        stride: Bước nhảy khi tạo patch.
        
    Returns:
        Khối dữ liệu được tái tạo.
    """
    # Kích thước gốc
    h, w, d = original_shape
    
    # Loại bỏ kênh từ patches nếu cần
    if patches.ndim == 5 and patches.shape[4] == 1:
        patches = patches[..., 0]
    
    # Khởi tạo khối dữ liệu đầu ra và bộ đếm
    reconstructed = np.zeros((h, w, d), dtype=patches.dtype)
    counts = np.zeros((h, w, d), dtype=np.int32)
    
    # Chỉ số patch
    patch_idx = 0
    
    # Đặt giá trị từ mỗi patch vào khối đầu ra
    for z in range(0, d - patch_size[2] + 1, stride[2]):
        for y in range(0, h - patch_size[0] + 1, stride[0]):
            for x in range(0, w - patch_size[1] + 1, stride[1]):
                # Thêm giá trị patch
                reconstructed[y:y+patch_size[0], x:x+patch_size[1], z:z+patch_size[2]] += patches[patch_idx]
                
                # Tăng bộ đếm
                counts[y:y+patch_size[0], x:x+patch_size[1], z:z+patch_size[2]] += 1
                
                patch_idx += 1
    
    # Lấy trung bình cho các voxel được tính nhiều lần
    valid_indices = counts > 0
    reconstructed[valid_indices] /= counts[valid_indices]
    
    return reconstructed


if __name__ == "__main__":
    # Ví dụ cách sử dụng
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    # Tạo dữ liệu giả
    print("Tạo dữ liệu huấn luyện mẫu...")
    
    # Kích thước nhỏ để ví dụ
    vol_shape = (64, 64, 32)
    X = np.random.rand(*vol_shape)[..., np.newaxis]  # Thêm kênh
    
    # Tạo một mặt nạ ngẫu nhiên cho ví dụ
    y = np.random.randint(0, 2, size=vol_shape)[..., np.newaxis]
    
    # Chia thành tập huấn luyện và kiểm chứng
    X_train, X_val = X[:, :, :20], X[:, :, 20:]
    y_train, y_val = y[:, :, :20], y[:, :, 20:]
    
    # Tạo và huấn luyện mô hình
    model = UNet3D(
        input_shape=(64, 64, 20, 1),
        n_classes=1,
        base_filters=8,  # Giảm kích thước cho ví dụ
        depth=3,
        verbose=True
    )
    
    model.compile_model(learning_rate=1e-3)
    
    # Tạo thư mục cho mô hình
    os.makedirs("../../models", exist_ok=True)
    
    # Huấn luyện với số epoch nhỏ cho ví dụ
    history = model.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=2,
        epochs=5,
        checkpoint_path="../../models/unet3d_example.h5"
    )
    
    # Dự đoán
    y_pred = model.predict(X_val)
    
    # Trực quan hóa kết quả
    os.makedirs("../../data/processed", exist_ok=True)
    model.visualize_results(
        X_val, y_val, y_pred,
        save_path="../../data/processed/deep_learning_segmentation_example.png"
    )
    
    # Trực quan hóa lịch sử huấn luyện
    model.visualize_training_history(
        history,
        save_path="../../data/processed/training_history.png"
    )
    
    print("Đã hoàn thành ví dụ huấn luyện mô hình phân đoạn.") 