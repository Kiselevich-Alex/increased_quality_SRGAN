import os
import sys
from PyQt5 import QtWidgets
import ui

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DesignApplication(QtWidgets.QMainWindow, ui.Ui_mainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window_application = DesignApplication()
    window_application.show()
    app.exec_()
