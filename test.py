import sys
from PyQt5 import QtWidgets
from vision6D.widgets.custom_image_button_widget import CustomImageButtonWidget

# Replace this import with the actual import path of your CustomImageButtonWidget
# from your_module import CustomImageButtonWidget

class ScrollAreaTest(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        # Create a scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        # Create a container widget and layout
        container_widget = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container_widget)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Add custom widgets to the container layout
        for i in range(2000):
            button_widget = CustomImageButtonWidget(f"Button {i}", image_path='path/to/image')
            container_layout.addWidget(button_widget)

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(container_widget)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ScrollAreaTest()
    window.resize(400, 600)
    window.show()
    sys.exit(app.exec_())
