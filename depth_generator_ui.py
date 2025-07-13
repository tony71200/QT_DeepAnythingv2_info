import sys
import os
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QFileDialog, QHBoxLayout, QVBoxLayout, QSplitter, QFrame,
    QListWidget, QListWidgetItem, QTextEdit, QCheckBox, QLineEdit,
    QProgressDialog
)
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QImage, QIcon
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PIL import Image as PILImage
from PIL import PngImagePlugin
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from depth_anything_v2_func import DepthAnythingV2Detector

class ModelLoaderThread(QThread):
    model_loaded = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            detector = DepthAnythingV2Detector(device=device)
            self.model_loaded.emit(detector)
        except Exception as e:
            self.error_occurred.emit(str(e))

class DepthGenerationThread(QThread):
    generation_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, detector, image_path, colored):
        super().__init__()
        self.detector = detector
        self.image_path = image_path
        self.colored = colored

    def run(self):
        try:
            if not self.image_path or not os.path.exists(self.image_path):
                self.error_occurred.emit("Đường dẫn ảnh không hợp lệ.")
                return

            input_image = cv2.imread(self.image_path)
            if input_image is None:
                self.error_occurred.emit(f"Không thể đọc ảnh từ: {self.image_path}")
                return

            depth_image = self.detector(input_image, colored=self.colored)
            self.generation_finished.emit(depth_image)
        except Exception as e:
            self.error_occurred.emit(f"Lỗi khi tạo depth map: {e}")

class BatchGenerationThread(QThread):
    progress_updated = pyqtSignal(int, str) # value, label
    batch_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, detector, all_image_paths, colored):
        super().__init__()
        self.detector = detector
        self.all_image_paths = all_image_paths
        self.colored = colored
        self.is_cancelled = False

    def run(self):
        try:
            valid_image_paths = [
                p for p in self.all_image_paths 
                if os.path.exists(os.path.splitext(p)[0] + ".txt")
            ]

            if not valid_image_paths:
                self.batch_finished.emit("Không tìm thấy ảnh nào có file .txt tương ứng.")
                return

            total_images = len(valid_image_paths)
            for i, image_path in enumerate(valid_image_paths):
                if self.is_cancelled:
                    break
                
                progress_value = int(((i + 1) / total_images) * 100)
                progress_label = f"Đang xử lý: {os.path.basename(image_path)} ({i+1}/{total_images})"
                self.progress_updated.emit(progress_value, progress_label)

                positive_prompt, negative_prompt = "", ""
                try:
                    prompt_path = os.path.splitext(image_path)[0] + ".txt"
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        parts = content.split("###")
                        positive_prompt = parts[0].strip() if len(parts) > 0 else ""
                        negative_prompt = parts[1].strip() if len(parts) > 1 else ""
                except Exception as e:
                    print(f"Bỏ qua: Lỗi đọc file prompt {os.path.basename(prompt_path)}: {e}")
                    continue

                input_image = cv2.imread(image_path)
                if input_image is None:
                    print(f"Bỏ qua: Lỗi đọc ảnh {os.path.basename(image_path)}")
                    continue
                
                depth_image_np = self.detector(input_image, colored=self.colored)
                if depth_image_np is None:
                    print(f"Bỏ qua: Lỗi tạo depth map cho {os.path.basename(image_path)}")
                    continue

                folder_path = os.path.dirname(image_path)
                name, _ = os.path.splitext(os.path.basename(image_path))
                save_path = os.path.join(folder_path, f"{name}_depth.png")
                
                h, w = depth_image_np.shape[:2]
                image_size = f"{w}x{h}"
                metadata_str = f"{positive_prompt}###{negative_prompt}###{image_size}"

                try:
                    if len(depth_image_np.shape) == 3:
                        rgb_image = cv2.cvtColor(depth_image_np, cv2.COLOR_BGR2RGB)
                        pil_img = PILImage.fromarray(rgb_image)
                    else:
                        pil_img = PILImage.fromarray(depth_image_np)
                    pil_img = pil_img.convert("RGB")
                    
                    metadata = PngImagePlugin.PngInfo()
                    metadata.add_text("parameters", metadata_str)
                    pil_img.save(save_path, "PNG", pnginfo=metadata)
                except Exception as e:
                    print(f"Bỏ qua: Lỗi lưu ảnh {os.path.basename(save_path)}: {e}")
            
            if self.is_cancelled:
                self.batch_finished.emit("Tác vụ đã bị hủy.")
            else:
                self.batch_finished.emit(f"Hoàn thành! Đã xử lý {i+1}/{total_images} ảnh.")

        except Exception as e:
            self.error_occurred.emit(f"Lỗi nghiêm trọng trong quá trình batch: {e}")

    def cancel(self):
        self.is_cancelled = True

class ImageDropLabel(QLabel):
    """
    Một QLabel tùy chỉnh hỗ trợ kéo và thả ảnh.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.image_dropped_callback = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path) and any(path.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(path)
        
        if image_paths and self.image_dropped_callback:
            self.image_dropped_callback(image_paths)

class DepthGeneratorWindow(QMainWindow):
    """
    Cửa sổ chính của ứng dụng xử lý ảnh, kết hợp quản lý prompt và tạo Depth Map.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processor (Prompt & Depth Map)")
        self.setGeometry(100, 100, 1300, 800)
        
        self.folder_path = ""
        self.current_image_path = None
        self.output_pixmap = None 
        self.output_image_np = None # Thêm để lưu trữ ảnh numpy
        self.detector = None 
        self.progress_dialog = None

        self.init_ui()
        self.init_model_loader()
        self.statusBar().showMessage("Đang tải model...")

    def init_model_loader(self):
        self.set_ui_enabled(False)
        self.model_loader_thread = ModelLoaderThread()
        self.model_loader_thread.model_loaded.connect(self.on_model_loaded)
        self.model_loader_thread.error_occurred.connect(self.on_model_load_error)
        self.model_loader_thread.start()

    def on_model_loaded(self, detector):
        self.detector = detector
        self.statusBar().showMessage("Model đã sẵn sàng.", 5000)
        self.set_ui_enabled(True)
        self.btn_generate.setEnabled(self.current_image_path is not None)

    def on_model_load_error(self, error_message):
        self.statusBar().showMessage(f"Lỗi tải model: {error_message}")
        self.btn_generate.setText("Lỗi Model")

    def init_ui(self):
        # Widget trung tâm và layout chính
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Layout chọn thư mục và môi trường ảo
        top_bar_widget = QWidget()
        top_bar_layout = QVBoxLayout(top_bar_widget)
        top_bar_layout.setContentsMargins(0, 5, 0, 5)

        env_layout = QHBoxLayout()
        self.lbl_venv = QLabel("Môi trường ảo:")
        self.txt_venv_path = QLineEdit()
        self.txt_venv_path.setPlaceholderText("Đường dẫn đến môi trường ảo...")
        try:
            venv_path = os.environ.get('VIRTUAL_ENV', sys.executable)
            # Đi lên 2 cấp từ file thực thi python (thường là /venv/Scripts/python.exe)
            self.txt_venv_path.setText(os.path.dirname(os.path.dirname(venv_path)))
        except Exception:
            pass # Bỏ qua nếu không lấy được
        env_layout.addWidget(self.lbl_venv)
        env_layout.addWidget(self.txt_venv_path)

        folder_layout = QHBoxLayout()
        self.btn_browse = QPushButton()
        self.btn_browse.setIcon(QIcon.fromTheme("folder"))
        self.btn_browse.setFixedSize(32, 32)
        self.btn_browse.clicked.connect(self.choose_folder)
        self.lbl_folder = QLabel("Chưa chọn thư mục")
        folder_layout.addWidget(self.btn_browse)
        folder_layout.addWidget(self.lbl_folder, 1)
        
        top_bar_layout.addLayout(env_layout)
        top_bar_layout.addLayout(folder_layout)
        top_bar_widget.setFixedHeight(90)

        # Splitter chính
        self.main_splitter = QSplitter(Qt.Horizontal)

        # ----- KHU VỰC BÊN TRÁI: HIỂN THỊ ẢNH -----
        image_widget = QWidget()
        image_layout = QVBoxLayout()
        image_widget.setLayout(image_layout)

        self.lbl_image = ImageDropLabel("Kéo ảnh vào đây hoặc chọn từ danh sách")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setFrameShape(QFrame.StyledPanel)
        self.lbl_image.setMinimumHeight(400)
        self.lbl_image.image_dropped_callback = self.handle_dropped_images

        self.list_images = QListWidget()
        self.list_images.setFixedHeight(150)
        self.list_images.setViewMode(QListWidget.IconMode)
        self.list_images.setIconSize(QSize(100, 100))
        self.list_images.setResizeMode(QListWidget.Adjust)
        self.list_images.itemClicked.connect(self.display_selected_image)

        image_layout.addWidget(self.lbl_image)
        image_layout.addWidget(self.list_images)

        # ----- KHU VỰC BÊN PHẢI: ĐIỀU KHIỂN & KẾT QUẢ -----
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        # Khu vực Prompt
        self.txt_positive = QTextEdit()
        self.txt_positive.setPlaceholderText("Positive Prompt...")
        self.txt_negative = QTextEdit()
        self.txt_negative.setPlaceholderText("Negative Prompt...")
        self.chk_autosave = QCheckBox("Tự động lưu prompt khi chuyển ảnh")
        self.btn_save_prompt = QPushButton("Lưu Prompt")
        self.btn_save_prompt.clicked.connect(self.save_prompt_file)
        
        prompt_button_layout = QHBoxLayout()
        prompt_button_layout.addWidget(self.chk_autosave)
        prompt_button_layout.addWidget(self.btn_save_prompt)

        # Khu vực Depth Map
        self.lbl_output_image = QLabel("Depth map sẽ hiển thị ở đây")
        self.lbl_output_image.setAlignment(Qt.AlignCenter)
        self.lbl_output_image.setFrameShape(QFrame.StyledPanel)
        self.lbl_output_image.setMinimumHeight(250)

        self.chk_colored_output = QCheckBox("Tạo ảnh màu (Inferno)")
        self.chk_colored_output.setChecked(True)

        self.btn_generate = QPushButton("Tạo Depth Map")
        self.btn_generate.clicked.connect(self.generate_depth_map)
        self.btn_generate.setEnabled(False) 

        self.btn_save_image = QPushButton("Lưu Depth Map")
        self.btn_save_image.clicked.connect(self.save_output_image)
        self.btn_save_image.setEnabled(False)

        self.btn_batch_generate = QPushButton("Batch generate Depthmap")
        self.btn_batch_generate.clicked.connect(self.start_batch_generation)
        self.btn_batch_generate.setEnabled(False)

        depth_button_layout = QHBoxLayout()
        depth_button_layout.addWidget(self.btn_generate)
        depth_button_layout.addWidget(self.btn_save_image)
        

        options_layout = QHBoxLayout()
        options_layout.addWidget(self.chk_colored_output)
        options_layout.addStretch()

        # Sắp xếp layout bên phải
        right_layout.addWidget(QLabel("<b>Prompt Editor</b>"))
        right_layout.addWidget(self.txt_positive)
        right_layout.addWidget(self.txt_negative)
        right_layout.addLayout(prompt_button_layout)
        right_layout.addWidget(QFrame(frameShape=QFrame.HLine))
        right_layout.addWidget(QLabel("<b>Depth Map Generator</b>"))
        right_layout.addWidget(self.lbl_output_image, 1) # Cho phép label giãn ra
        right_layout.addLayout(options_layout)
        right_layout.addLayout(depth_button_layout)
        right_layout.addWidget(self.btn_batch_generate)

        # Thêm vào splitter
        self.main_splitter.addWidget(image_widget)
        self.main_splitter.addWidget(right_widget)
        self.main_splitter.setSizes([800, 500])

        # Thêm vào layout chính
        main_layout.addWidget(top_bar_widget)
        main_layout.addWidget(self.main_splitter)

    def choose_folder(self):
        initial_dir = self.folder_path if os.path.isdir(self.folder_path) else os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa ảnh", initial_dir)
        if folder:
            self.folder_path = folder
            self.lbl_folder.setText(folder)
            self.load_images_from_folder(folder)

    def load_images_from_folder(self, folder):
        self.list_images.clear()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

        try:
            image_files = [f for f in os.listdir(folder) if any(f.lower().endswith(ext) for ext in image_extensions)]
            image_files.sort()

            for file in image_files:
                full_path = os.path.join(folder, file)
                icon = QIcon(full_path)
                item = QListWidgetItem(icon, file)
                item.setData(Qt.UserRole, full_path)
                self.list_images.addItem(item)

            if self.list_images.count() > 0:
                self.list_images.setCurrentRow(0)
                self.display_selected_image(self.list_images.item(0))
        except Exception as e:
            self.statusBar().showMessage(f"Lỗi khi tải thư mục: {e}")

    def handle_dropped_images(self, image_paths):
        if not image_paths:
            return
        
        folder = os.path.dirname(image_paths[0])
        if folder != self.folder_path:
            self.folder_path = folder
            self.lbl_folder.setText(folder)
            self.load_images_from_folder(folder)
        
        # Tìm và chọn ảnh đầu tiên được thả vào
        for i in range(self.list_images.count()):
            item = self.list_images.item(i)
            if item.data(Qt.UserRole) == image_paths[0]:
                self.list_images.setCurrentItem(item)
                self.display_selected_image(item)
                break

    def display_selected_image(self, item):
        if self.chk_autosave.isChecked() and self.current_image_path:
            self.save_prompt_file()

        self.current_image_path = item.data(Qt.UserRole)
        pixmap = QPixmap(self.current_image_path)

        if not pixmap.isNull():
            self._display_image(self.lbl_image, pixmap)
            self.statusBar().showMessage(self.current_image_path)
            self.load_prompt_file(self.current_image_path)
            
            # Reset depth map view
            self.lbl_output_image.clear()
            self.lbl_output_image.setText("Depth map sẽ hiển thị ở đây")
            self.output_pixmap = None
            self.output_image_np = None # Reset ảnh numpy
            self.btn_save_image.setEnabled(False)
            self.btn_generate.setEnabled(self.detector is not None)
            self.btn_batch_generate.setEnabled(self.detector is not None)
        else:
            self.lbl_image.setText(f"Không thể hiển thị:\n{os.path.basename(self.current_image_path)}")
            self.statusBar().showMessage("Lỗi: không thể hiển thị ảnh.")
            self.btn_generate.setEnabled(False)

    def _display_image(self, label, pixmap):
        if pixmap and not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)

    def load_prompt_file(self, image_path):
        prompt_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Giả định prompt dương và âm ngăn cách bởi "###"
                    parts = content.split("###")
                    self.txt_positive.setText(parts[0].strip() if len(parts) > 0 else "")
                    self.txt_negative.setText(parts[1].strip() if len(parts) > 1 else "")
            except Exception as e:
                self.statusBar().showMessage(f"Lỗi đọc file prompt: {e}")
        else:
            self.txt_positive.clear()
            self.txt_negative.clear()

    def save_prompt_file(self):
        if not self.current_image_path:
            return

        positive_prompt = self.txt_positive.toPlainText().strip()
        negative_prompt = self.txt_negative.toPlainText().strip()
        
        content = f"{positive_prompt}###{negative_prompt}"
        prompt_path = os.path.splitext(self.current_image_path)[0] + ".txt"

        try:
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.statusBar().showMessage(f"Đã lưu: {os.path.basename(prompt_path)}", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"Lỗi khi lưu prompt: {e}", 5000)

    def generate_depth_map(self):
        if not self.current_image_path:
            self.statusBar().showMessage("Vui lòng chọn ảnh trước.")
            return
        if not self.detector:
            self.statusBar().showMessage("Model chưa được tải xong. Vui lòng chờ.")
            return
        
        self.set_ui_enabled(False)
        self.statusBar().showMessage("Đang tạo depth map...")

        colored = self.chk_colored_output.isChecked()
        self.generation_thread = DepthGenerationThread(
            detector=self.detector,
            image_path=self.current_image_path,
            colored=colored
        )
        self.generation_thread.generation_finished.connect(self.on_depth_generation_finished)
        self.generation_thread.error_occurred.connect(self.on_generation_error)
        self.generation_thread.start()

    def on_depth_generation_finished(self, depth_image_np):
        try:
            if depth_image_np is None:
                raise ValueError("Kết quả depth map là None.")
            
            self.output_image_np = depth_image_np # Lưu lại ảnh numpy gốc

            # Convert numpy array to QPixmap
            if len(depth_image_np.shape) == 3: # Color image
                h, w, ch = depth_image_np.shape
                # cv2 đọc ảnh dạng BGR, QImage cần RGB
                rgb_image = cv2.cvtColor(depth_image_np, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            else: # Grayscale image
                h, w = depth_image_np.shape
                q_image = QImage(depth_image_np.data, w, h, w, QImage.Format_Grayscale8)

            self.output_pixmap = QPixmap.fromImage(q_image)

            if self.output_pixmap and not self.output_pixmap.isNull():
                self._display_image(self.lbl_output_image, self.output_pixmap)
                self.btn_save_image.setEnabled(True)
                self.statusBar().showMessage("Tạo depth map thành công.", 5000)
            else:
                self.statusBar().showMessage("Tạo depth map thất bại (lỗi chuyển đổi ảnh).", 5000)
        except Exception as e:
            self.on_generation_error(f"Lỗi xử lý kết quả: {e}")
        finally:
            self.set_ui_enabled(True)

    def on_generation_error(self, error_message):
        self.statusBar().showMessage(error_message, 5000)
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.close()
        self.set_ui_enabled(True)

    def set_ui_enabled(self, enabled):
        """Kích hoạt hoặc vô hiệu hóa các thành phần UI chính."""
        self.main_splitter.setEnabled(enabled)
        self.btn_browse.setEnabled(enabled)
        self.list_images.setEnabled(enabled)
        
        is_ready_to_generate = self.current_image_path is not None and self.detector is not None
        self.btn_generate.setEnabled(enabled and is_ready_to_generate)
        
        is_ready_to_save = self.output_image_np is not None
        self.btn_save_image.setEnabled(enabled and is_ready_to_save)

        is_ready_for_batch = self.list_images.count() > 0 and self.detector is not None
        self.btn_batch_generate.setEnabled(enabled and is_ready_for_batch)

    def save_output_image(self):
        if self.output_image_np is None:
            self.statusBar().showMessage("Không có depth map để lưu.", 5000)
            return

        positive_prompt = self.txt_positive.toPlainText().strip()
        negative_prompt = self.txt_negative.toPlainText().strip()
        h, w = self.output_image_np.shape[:2]
        image_size = f"{w}x{h}"
        metadata_str = f"{positive_prompt}###{negative_prompt}###{image_size}"
        
        save_path = ""
        if self.chk_autosave.isChecked():
            base_name = os.path.basename(self.current_image_path)
            name, _ = os.path.splitext(base_name)
            save_path = os.path.join(self.folder_path, f"{name}_depth.png")
        else:
            base_name = os.path.basename(self.current_image_path)
            name, _ = os.path.splitext(base_name)
            suggested_path = os.path.join(self.folder_path, f"{name}_depth.png")
            save_path, _ = QFileDialog.getSaveFileName(self, "Lưu Depth Map", suggested_path, "PNG Image (*.png)")

        if save_path:
            try:
                # Chuyển đổi trực tiếp từ numpy array sang PIL Image
                if len(self.output_image_np.shape) == 3:
                    # Ảnh numpy từ OpenCV là BGR, Pillow cần RGB
                    rgb_image = cv2.cvtColor(self.output_image_np, cv2.COLOR_BGR2RGB)
                    pil_img = PILImage.fromarray(rgb_image)
                else: # Ảnh xám
                    pil_img = PILImage.fromarray(self.output_image_np)
                pil_img = pil_img.convert("RGB")
                
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("parameters", metadata_str, zip=False)
                
                pil_img.save(save_path, "PNG", pnginfo=metadata)
                self.statusBar().showMessage(f"Đã lưu: {save_path}", 5000)
            except Exception as e:
                self.statusBar().showMessage(f"Lưu ảnh thất bại: {e}", 5000)

    def start_batch_generation(self):
        if not self.detector:
            self.statusBar().showMessage("Model chưa sẵn sàng.")
            return
        
        if self.list_images.count() == 0:
            self.statusBar().showMessage("Không có ảnh trong danh sách để xử lý.")
            return

        all_image_paths = [self.list_images.item(i).data(Qt.UserRole) for i in range(self.list_images.count())]

        self.progress_dialog = QProgressDialog("Đang chuẩn bị xử lý hàng loạt...", "Hủy", 0, 100, self)
        self.progress_dialog.setWindowTitle("Đang xử lý hàng loạt")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()

        self.set_ui_enabled(False)
        
        colored = self.chk_colored_output.isChecked()
        self.batch_thread = BatchGenerationThread(
            detector=self.detector,
            all_image_paths=all_image_paths,
            colored=colored
        )
        self.progress_dialog.canceled.connect(self.batch_thread.cancel)
        self.batch_thread.progress_updated.connect(self.update_progress_dialog)
        self.batch_thread.batch_finished.connect(self.on_batch_finished)
        self.batch_thread.error_occurred.connect(self.on_generation_error)
        self.batch_thread.start()

    def update_progress_dialog(self, value, label):
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(label)

    def on_batch_finished(self, message):
        if self.progress_dialog:
            self.progress_dialog.close()
        self.statusBar().showMessage(message, 5000)
        self.set_ui_enabled(True)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Right, Qt.Key_D):
            self.navigate_image(1)
        elif key in (Qt.Key_Left, Qt.Key_A):
            self.navigate_image(-1)

    def navigate_image(self, step):
        if self.list_images.count() == 0: return
        current_row = self.list_images.currentRow()
        next_row = (current_row + step) % self.list_images.count()
        self.list_images.setCurrentRow(next_row)
        item = self.list_images.item(next_row)
        if item:
            self.display_selected_image(item)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image_path:
            self._display_image(self.lbl_image, QPixmap(self.current_image_path))
        if self.output_pixmap:
            self._display_image(self.lbl_output_image, self.output_pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        with open("MacOS.qss", "r") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        pass # Bỏ qua nếu không có file stylesheet
    window = DepthGeneratorWindow()
    window.show()
    sys.exit(app.exec_()) 