# General import
import numpy as np
import json
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Qt5 import
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

# self defined package import
np.set_printoptions(suppress=True)
class VideoPlayer(QtWidgets.QDialog):
    def __init__(self, video_path, current_frame):
        super().__init__()

        self.setWindowTitle("Vision6D - Video Player")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint) # Disable the question mark
        self.setFixedSize(1000, 600)

        self.layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel(self)
        self.layout.addWidget(self.label, 0, QtCore.Qt.AlignCenter)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        self.layout.addWidget(self.slider)

        self.button_layout = QtWidgets.QHBoxLayout()

        self.prev_button = QtWidgets.QPushButton('Previous Frame', self)
        self.prev_button.clicked.connect(self.prev_frame)
        self.button_layout.addWidget(self.prev_button)

        self.play = False

        # Load the video
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setMaximum(self.frame_count - 1)

        video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if video_width > 960 and video_height > 540: self.video_size = int(video_width // 2), int(video_height // 2)
        else: self.video_size = video_width, video_height
        
        self.current_frame = current_frame

        self.play_pause_button = QtWidgets.QPushButton(f'Play/Pause ({self.current_frame}/{self.frame_count})', self)
        self.play_pause_button.clicked.connect(self.play_pause_video)
        self.button_layout.addWidget(self.play_pause_button)

        self.next_button = QtWidgets.QPushButton('Next Frame', self)
        self.next_button.clicked.connect(self.next_frame)
        self.button_layout.addWidget(self.next_button)

        self.layout.addLayout(self.button_layout)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.isPlaying = False

        # Display frame
        self.update_frame()

    def slider_moved(self, value):
        self.current_frame = value
        self.update_frame()

    def play_pause_video(self):
        self.play = not self.play

        if self.play:
            if not self.isPlaying:
                self.timer.start(self.fps)  # play 30 frames per second
                self.isPlaying = True
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
        else:
            self.timer.stop()
            self.isPlaying = False
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)

    def next_frame(self):
        self.current_frame = min(self.current_frame + 1, self.frame_count)
        self.slider.setValue(self.current_frame)

    def prev_frame(self):
        self.current_frame = max(0, self.current_frame - 1)
        self.slider.setValue(self.current_frame)

    def update_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.play_pause_button.setText(f'Play/Pause ({self.current_frame}/{self.frame_count})')
        ret, frame = self.cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QtGui.QImage(rgb_image.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            p = convert_to_qt_format.scaled(*self.video_size, QtCore.Qt.KeepAspectRatio)
            self.label.setPixmap(QtGui.QPixmap.fromImage(p))

class YesNoBox(QtWidgets.QMessageBox):
    def __init__(self, *args, **kwargs):
        super(YesNoBox, self).__init__(*args, **kwargs)
        self.canceled = False

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.canceled = True
        super(YesNoBox, self).closeEvent(event)

class PopUpDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, on_button_click=None):
        super().__init__(parent)

        self.setWindowTitle("Vision6D - Colors")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint) # Disable the question mark

        button_grid = QtWidgets.QGridLayout()
        colors = ["nocs", "cyan", "magenta", 
                "yellow", "lime", "deepskyblue", "latlon", "salmon", 
                "silver", "aquamarine", "plum", "blueviolet"]

        button_count = 0
        for i in range(2):
            for j in range(6):
                name = f"{colors[button_count]}"
                button = QtWidgets.QPushButton(name)
                button.clicked.connect(lambda _, idx=name: on_button_click(str(idx)))
                button_grid.addWidget(button, j, i)
                button_count += 1

        self.setLayout(button_grid)

class CameraPropsInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, 
                    line1=(None, None), 
                    line2=(None, None), 
                    line3=(None, None), 
                    line4=(None, None), 
                    line5=(None, None),
                    line6=(None, None)):
        super().__init__(parent)

        self.args1 = QtWidgets.QLineEdit(self, text=str(line1[1]))
        self.args2 = QtWidgets.QLineEdit(self, text=str(line2[1]))
        self.args3 = QtWidgets.QLineEdit(self, text=str(line3[1]))
        self.args4 = QtWidgets.QLineEdit(self, text=str(line4[1]))
        self.args5 = QtWidgets.QLineEdit(self, text=str(line5[1]))
        self.args6 = QtWidgets.QLineEdit(self, text=str(line6[1]))

        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow(f"{line1[0]}", self.args1)
        layout.addRow(f"{line2[0]}", self.args2)
        layout.addRow(f"{line3[0]}", self.args3)
        layout.addRow(f"{line4[0]}", self.args4)
        layout.addRow(f"{line5[0]}", self.args5)
        layout.addRow(f"{line6[0]}", self.args6)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.args1.text(), 
                self.args2.text(), 
                self.args3.text(),
                self.args4.text(),
                self.args5.text(),
                self.args6.text())
    
class CalibrationPopWindow(QtWidgets.QDialog):
    def __init__(self, calibrated_image, original_image, parent=None):
        super(CalibrationPopWindow, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.calibrated_image = calibrated_image
        self.original_image = original_image

        self.setWindowTitle("Vision6D")

        # Set the size for display
        if self.original_image.shape[1] > 960 and self.original_image.shape[1] > 540:
            size = int(self.original_image.shape[1] // 2), int(self.original_image.shape[0] // 2)
        else: size = int(self.original_image.shape[1]), int(self.original_image.shape[0])
        
        overall = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()

        vbox1layout = QtWidgets.QVBoxLayout()
        label1 = QtWidgets.QLabel("Calibrated image", self)
        label1.setAlignment(Qt.AlignCenter)
        pixmap_label1 = QtWidgets.QLabel(self)

        qimage1 = self.numpy_to_qimage(self.calibrated_image)
        pixmap1 = QtGui.QPixmap.fromImage(qimage1).scaled(*size, Qt.KeepAspectRatio)
        
        pixmap_label1.setPixmap(pixmap1)
        pixmap_label1.setAlignment(Qt.AlignCenter)
        pixmap_label1.setFixedSize(*size)
        
        vbox1layout.addWidget(label1)
        vbox1layout.addWidget(pixmap_label1)

        vbox2layout = QtWidgets.QVBoxLayout()
        label2 = QtWidgets.QLabel("Original image", self)
        label2.setAlignment(Qt.AlignCenter)
        pixmap_label2 = QtWidgets.QLabel(self)

        qimage2 = self.numpy_to_qimage(self.original_image)
        pixmap2 = QtGui.QPixmap.fromImage(qimage2).scaled(*size, Qt.KeepAspectRatio)
        
        pixmap_label2.setPixmap(pixmap2)
        pixmap_label2.setAlignment(Qt.AlignCenter)
        pixmap_label2.setFixedSize(*size)
        
        vbox2layout.addWidget(label2)
        vbox2layout.addWidget(pixmap_label2)

        layout.addLayout(vbox1layout)
        layout.addLayout(vbox2layout)

        psnr, ssim = self.calculate_similarity()
        similarity_label = QtWidgets.QLabel(f"PSNR is {psnr} and SSIM is {ssim}", self)
        similarity_label.setAlignment(Qt.AlignCenter)

        overall.addLayout(layout)
        overall.addWidget(similarity_label)

        self.setLayout(overall)

    def numpy_to_qimage(self, array):
        height, width, channel = array.shape
        #* bytesPerline is very important
        bytesPerLine = channel * width
        return QtGui.QImage(array.tobytes(), width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

    def calculate_similarity(self):
        psnr = peak_signal_noise_ratio(self.calibrated_image, self.original_image, data_range=255)
        ssim = structural_similarity(self.calibrated_image, self.original_image, data_range=255, channel_axis=2)
        return psnr, ssim

class GetTextDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(GetTextDialog, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        
        self.setWindowTitle("Vision6D")
        self.introLabel = QtWidgets.QLabel("Input the Ground Truth Pose:")
        self.btnloadfromfile = QtWidgets.QPushButton("Load from file", self)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.introLabel)
        hbox.addWidget(self.btnloadfromfile)
        hbox.setContentsMargins(0, 0, 0, 0)
        self.hboxWidget = QtWidgets.QWidget()
        self.hboxWidget.setLayout(hbox)

        self.btnloadfromfile.clicked.connect(self.load_from_file)
        self.textEdit = QtWidgets.QTextEdit(self)
        self.textEdit.setPlainText(f"[[1, 0, 0, 0], \n[0, 1, 0, 0], \n[0, 0, 1, 0], \n[0, 0, 0, 1]]")
        self.btnSubmit = QtWidgets.QPushButton("Submit", self)
        self.btnSubmit.clicked.connect(self.submit_text)

        layout.addWidget(self.hboxWidget)
        layout.addWidget(self.textEdit)
        layout.addWidget(self.btnSubmit)

    def submit_text(self):
        self.user_text = self.textEdit.toPlainText()
        self.accept()

    def load_from_file(self):
        file_dialog = QtWidgets.QFileDialog()
        pose_path, _ = file_dialog.getOpenFileName(None, "Open file", "", "Files (*.npy)")
        if pose_path != '':
            gt_pose = np.load(pose_path)
            self.textEdit.setPlainText(f"[[{np.around(gt_pose[0, 0], 8)}, {np.around(gt_pose[0, 1], 8)}, {np.around(gt_pose[0, 2], 8)}, {np.around(gt_pose[0, 3], 8)}],\n"
                                    f"[{np.around(gt_pose[1, 0], 8)}, {np.around(gt_pose[1, 1], 8)}, {np.around(gt_pose[1, 2], 8)}, {np.around(gt_pose[1, 3], 8)}],\n"
                                    f"[{np.around(gt_pose[2, 0], 8)}, {np.around(gt_pose[2, 1], 8)}, {np.around(gt_pose[2, 2], 8)}, {np.around(gt_pose[2, 3], 8)}],\n"
                                    f"[{np.around(gt_pose[3, 0], 8)}, {np.around(gt_pose[3, 1], 8)}, {np.around(gt_pose[3, 2], 8)}, {np.around(gt_pose[3, 3], 8)}]]")

    def get_text(self):
        return self.user_text

