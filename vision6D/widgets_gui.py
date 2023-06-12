# General import
import numpy as np
import cv2
import pathlib
import PIL.Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Qt5 import
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

# self defined package import
np.set_printoptions(suppress=True)

class VideoSampler(QtWidgets.QDialog):
    def __init__(self, video_player, fps, parent=None):
        super(VideoSampler, self).__init__(parent)
        self.setWindowTitle("Vision6D - Video Sampler")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint) # Disable the question mark
        self.setFixedSize(800, 500)
        
        self.video_player = video_player
        self.fps = fps

        layout = QtWidgets.QVBoxLayout(self)

        # Create QLabel for the top
        label1 = QtWidgets.QLabel("How often should we sample this video?", self)
        font = QtGui.QFont('Times', 14)
        font.setBold(True)
        label1.setFont(font)

        label1.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label1)

        line = QtWidgets.QFrame(self)
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("color: grey")
        layout.addWidget(line)

        # Load the video
        self.video_path = self.video_player.video_path
        self.cap = self.video_player.cap
        self.frame_count = self.video_player.frame_count
        video_width = self.video_player.video_width
        video_height = self.video_player.video_height

        if video_width > 600 and video_height > 400: self.video_size = int(video_width // 4), int(video_height // 4)
        else: self.video_size = video_width, video_height
        
        # Create a QLabel to hold the thumbnail
        self.thumbnail_label = QtWidgets.QLabel(self)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret: thumbnail_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Load the image using QPixmap
        img = QtGui.QImage(thumbnail_frame.tobytes(), thumbnail_frame.shape[1], thumbnail_frame.shape[0], thumbnail_frame.shape[2]*thumbnail_frame.shape[1], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(img)
        
        # Resize the QPixmap to the desired thumbnail size
        thumbnail = pixmap.scaled(*self.video_size)  # Change the size to fit your needs
        
        # Set the QPixmap as the image displayed by the QLabel
        self.thumbnail_label.setPixmap(thumbnail)
        
        layout.addWidget(self.thumbnail_label, alignment=Qt.AlignCenter)

        # Calculate and print the video duration in seconds
        total_seconds = self.frame_count / self.fps
        minutes, remainder_seconds = divmod(total_seconds, 60)

        # Create QLabel for the bottom
        label2 = QtWidgets.QLabel(f"{pathlib.Path(self.video_path).stem}{pathlib.Path(self.video_path).suffix} ({int(minutes)}m{int(remainder_seconds)}s)", self)
        font = QtGui.QFont('Times', 10)
        font.setBold(True)
        label2.setFont(font)
        label2.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label2, alignment=Qt.AlignCenter)

        hlayout = QtWidgets.QGridLayout()
        self.label_frame_rate = QtWidgets.QLabel(f"Frame per step: ")
        self.label_frame_rate.setContentsMargins(80, 0, 0, 0)
        self.label_frame_rate.setFont(font)
        hlayout.addWidget(self.label_frame_rate, 0, 0)
        self.step_spinbox = QtWidgets.QSpinBox()
        self.step_spinbox.setMinimum(1)
        self.step_spinbox.setMaximum(self.frame_count)
        self.step_spinbox.setValue(self.fps)
        self.step_spinbox.valueChanged.connect(self.step_spinbox_value_changed)
        hlayout.addWidget(self.step_spinbox, 0, 1)
        self.output_size_label = QtWidgets.QLabel(f"Total output images: ")
        self.output_size_label.setContentsMargins(80, 0, 0, 0)
        self.output_size_label.setFont(font)
        hlayout.addWidget(self.output_size_label, 0, 2)
        self.output_spinbox = QtWidgets.QSpinBox()
        self.output_spinbox.setMinimum(1)
        self.output_spinbox.setMaximum(self.frame_count)
        self.output_spinbox.setValue(round(self.frame_count // self.fps))
        self.output_spinbox.valueChanged.connect(self.output_spinbox_value_changed)
        hlayout.addWidget(self.output_spinbox, 0, 3)
        layout.addLayout(hlayout)

        line = QtWidgets.QFrame(self)
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("color: grey")
        layout.addWidget(line)

        accept_button = QtWidgets.QPushButton('Choose Frame Rate', self)
        font = QtGui.QFont('Times', 12); font.setBold(True)
        accept_button.setFont(font)
        accept_button.setFixedSize(300, 40)
        accept_button.clicked.connect(self.accept)
        layout.addWidget(accept_button, alignment=Qt.AlignRight)

        self.setLayout(layout)

    def step_spinbox_value_changed(self, value):
        self.fps = value
        self.output_spinbox.setValue(round(self.frame_count // self.fps))
        
    def output_spinbox_value_changed(self, value):
        self.fps = round(self.frame_count // value)
        self.step_spinbox.setValue(self.fps)

    def closeEvent(self, event):
        event.ignore()
        super().closeEvent(event)

    def accept(self):
        super().accept()

class VideoPlayer(QtWidgets.QDialog):
    def __init__(self, video_path, current_frame):
        super().__init__()

        self.setWindowTitle("Vision6D - Video Player")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint) # Disable the question mark
        self.setFixedSize(1000, 600)

        # Add shortcut to the right, left, space buttons
        QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self).activated.connect(self.next_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self).activated.connect(self.prev_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self).activated.connect(self.play_pause_video)

        self.video_path = video_path

        self.layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel(self)
        self.layout.addWidget(self.label, 0, QtCore.Qt.AlignCenter)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        self.layout.addWidget(self.slider)

        self.button_layout = QtWidgets.QHBoxLayout()

        self.prev_button = QtWidgets.QPushButton('Previous Frame', self)
        self.prev_button.clicked.connect(self.prev_frame)
        self.prev_button.setFixedSize(300, 30)
        self.button_layout.addWidget(self.prev_button)

        self.play = False

        # Load the video
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setMaximum(self.frame_count - 1)

        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.video_width > 960 and self.video_height > 540: self.video_size = int(self.video_width // 2), int(self.video_height // 2)
        else: self.video_size = self.video_width, self.video_height
        
        self.current_frame = current_frame

        self.playback_speeds = [0.1, 0.2, 0.5, 1.0, 4.0, 16.0]  # different speeds
        self.current_playback_speed = 1  # Default speed is 1

        self.play_pause_button = QtWidgets.QToolButton(self)
        self.play_pause_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.play_pause_button.setText(f'Play/Pause ({self.current_frame}/{self.frame_count})')
        self.play_pause_button.clicked.connect(self.play_pause_video)

        self.play_pause_menu = QtWidgets.QMenu(self)
        self.play_action = QtWidgets.QAction('Play', self, triggered=self.play_video)
        self.pause_action = QtWidgets.QAction('Pause', self, triggered=self.pause_video)
        self.speed_menu = QtWidgets.QMenu('Playback Speed', self)

        self.speed_action_group = QtWidgets.QActionGroup(self.speed_menu)
        self.speed_action_group.setExclusive(True)
        for speed in self.playback_speeds:
            speed_action = QtWidgets.QAction(f'{speed}x', self.speed_menu, checkable=True)
            speed_action.triggered.connect(lambda _, s=speed: self.change_speed(speed=s))
            if speed == self.current_playback_speed: speed_action.setChecked(True)
            self.speed_menu.addAction(speed_action)
            self.speed_action_group.addAction(speed_action)

        self.play_pause_menu.addActions([self.play_action, self.pause_action])
        self.play_pause_menu.addMenu(self.speed_menu)
        self.play_pause_button.setMenu(self.play_pause_menu)
        self.play_pause_button.setFixedSize(300, 30)
        self.button_layout.addWidget(self.play_pause_button)

        self.next_button = QtWidgets.QPushButton('Next Frame', self)
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setFixedSize(300, 30)
        self.button_layout.addWidget(self.next_button)

        self.layout.addLayout(self.button_layout)

        line = QtWidgets.QFrame(self)
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("color: grey")
        self.layout.addWidget(line)

        accept_button = QtWidgets.QPushButton('Set selected frame', self)
        font = QtGui.QFont('Times', 12); font.setBold(True)
        accept_button.setFont(font)
        accept_button.setFixedSize(400, 40)
        accept_button.clicked.connect(self.accept)
        self.layout.addWidget(accept_button, alignment=Qt.AlignRight)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.isPlaying = False

        # Display frame
        self.update_frame()

    def slider_moved(self, value):
        self.current_frame = value
        self.update_frame()

    def change_speed(self, speed):
        self.current_playback_speed = speed
        self.play_video()

    def play_video(self):
        self.isPlaying = True
        self.timer.start(self.fps / self.current_playback_speed)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def pause_video(self):
        self.isPlaying = False
        self.timer.stop()
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)

    def play_pause_video(self):
        if self.isPlaying: self.pause_video()
        else: self.play_video()

    def next_frame(self):
        current_frame = self.current_frame + 1
        if current_frame <= self.frame_count: self.current_frame = current_frame
        self.slider.setValue(self.current_frame)

    def prev_frame(self):
        current_frame = self.current_frame - 1
        if current_frame >= 0: self.current_frame = current_frame
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

    def closeEvent(self, event):
        event.ignore()
        super().closeEvent(event)

    def accept(self):
        super().accept()

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
        colors = ["nocs", "cyan", "magenta", "yellow", "lime", "latlon", "dodgerblue", "darkviolet", "darkorange", "forestgreen"]

        button_count = 0
        for i in range(2):
            for j in range(5):
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

class LabelImage(QtWidgets.QLabel):
    output_path_changed = QtCore.pyqtSignal(str)
    def __init__(self, pixmap):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setPixmap(pixmap)
        self.setContentsMargins(0, 0, 0, 0)
        
        self.height = pixmap.height()
        self.width = pixmap.width()
        self.points = QtGui.QPolygon()
        self.output_path = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S: self.save_mask()

    def save_mask(self):
        points = []
        for point in self.points:
            coord = [point.x(), point.y()]
            points.append(coord)

        points = np.array(points).astype('int32')
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        image = cv2.fillPoly(mask, [points], 255)
        self.output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Mask Files (*.png)")
        if self.output_path:
            if pathlib.Path(self.output_path).suffix == '': self.output_path = self.output_path.parent / (self.output_path.stem + '.png')
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(self.output_path)
            self.output_path_changed.emit(self.output_path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.points.append(event.pos())
        elif event.button() == Qt.RightButton and not self.points.isEmpty():
            self.points.remove(self.points.size()-1)
        self.update()
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor(0, 0, 255))
        
        if not self.points.isEmpty():
            complete_points = QtGui.QPolygon(self.points)
            if (QtCore.QLineF(self.points.first(), self.points.last()).length() < 10) and (self.points.first() != self.points.last()):
                complete_points.append(self.points.first())
            # Draw the first point with a larger radius
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255)))
            painter.drawEllipse(self.points.point(0), 2, 2)
            painter.drawPolyline(complete_points)

class LabelWindow(QtWidgets.QWidget):
    def __init__(self, image_source):
        super().__init__()
        image_source = np.array(PIL.Image.open(image_source), dtype='uint8')
        image = QtGui.QImage(image_source.tobytes(), image_source.shape[1], image_source.shape[0], image_source.shape[2]*image_source.shape[1], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.setFixedSize(pixmap.size())

        layout = QtWidgets.QVBoxLayout()

        #* setContentsMargins sets the width of the outside border around the layout
        layout.setContentsMargins(0, 0, 0, 0)
        #* setSpacing sets the width of the inside border between widgets in the layout.
        layout.setSpacing(0)
        #* Both are set to zero to eliminate any space between the widgets and the layout border.

        self.image_label = LabelImage(pixmap)
        layout.addWidget(self.image_label)
        self.setLayout(layout)