# Qt5 import
from PyQt5 import QtWidgets

from .display_panel import DisplayPanel
from .output_panel import OutputPanel
from ...stores import QtStore, PlotStore

class Panel():

    def __init__(self, panel_widget):

        # Save reference
        self.panel_widget = panel_widget

        # Create the store
        self.plot_store = PlotStore()
        self.qt_store = QtStore()

        # Create a left panel layout
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel_widget)

        # Create a top panel bar with a toggle button
        self.panel_bar = QtWidgets.QMenuBar()
        self.toggle_action = QtWidgets.QAction("Panel", self.qt_store.main_window)
        self.toggle_action.triggered.connect(self.toggle_panel)
        self.panel_bar.addAction(self.toggle_action)

        # Create the display
        self.display = QtWidgets.QGroupBox("Console")
        self.panel_display = DisplayPanel(self.display)

        # Create the output
        self.output = QtWidgets.QGroupBox("Output")
        self.panel_output = OutputPanel(self.output)

        # Set the layout
        self.panel_layout.addWidget(self.display)
        self.panel_layout.addWidget(self.output)
        
        # Set the stretch factor for each section to be equal
        self.panel_layout.setStretchFactor(self.display, 1)
        self.panel_layout.setStretchFactor(self.output, 1)

    def toggle_panel(self):

        if self.panel_widget.isVisible():
            # self.panel_widget width changes when the panel is visiable or hiden
            self.panel_widget_width = self.panel_widget.width()
            self.panel_widget.hide()
            x = (self.plot_store.plotter.size().width() + self.panel_widget_width - self.qt_store.hintLabel.width()) // 2
            y = (self.plot_store.plotter.size().height() - self.qt_store.hintLabel.height()) // 2
            self.qt_store.hintLabel.move(x, y)
        else:
            self.panel_widget.show()
            x = (self.plot_store.plotter.size().width() - self.panel_widget_width - self.qt_store.hintLabel.width()) // 2
            y = (self.plot_store.plotter.size().height() - self.qt_store.hintLabel.height()) // 2
            self.qt_store.hintLabel.move(x, y)
