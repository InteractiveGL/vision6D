from .file_menu import FileMenu
from .export_menu import ExportMenu
from .video_menu import VideoMenu
from .camera_menu import CameraMenu
from .mirror_menu  import MirrorMenu
from .register_menu import RegisterMenu
from .pnp_menu import PnPMenu

class Menu():

    def __init__(self, mainMenu):
        self.mainMenu = mainMenu
        self.file_menu = FileMenu(self.mainMenu.addMenu('File'))
        # self.export_menu = ExportMenu(self.mainMenu.addMenu('Export'))
        # self.video_menu = VideoMenu(self.mainMenu.addMenu('Video/Folder'))
        # self.camera_menu = CameraMenu(self.mainMenu.addMenu('Camera'))
        # self.mirror_menu = MirrorMenu(self.mainMenu.addMenu('Mirror'))
        # self.register_menu = RegisterMenu(self.mainMenu.addMenu('Register'))
        # self.pnp_menu = PnPMenu(self.mainMenu.addMenu('PnP'))
        