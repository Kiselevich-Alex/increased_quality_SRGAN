from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QAction, QFileDialog
from PyQt5.QtCore import Qt
import evaluate
from evaluate import evaluate
import scipy
from PIL import Image
from numpy import asarray


class Ui_mainWindow(object):
    filename = ""

    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(800, 600)
        mainWindow.setMinimumSize(QtCore.QSize(800, 600))
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 801, 581))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

        open = QAction("Open File", self)
        open.setShortcut("Ctrl+O")
        open.triggered.connect(lambda: self.open())

        upscale = QAction("Upscale Image", self)
        upscale.setShortcut("Ctrl+U")
        upscale.triggered.connect(lambda: self.upscale())

        save = QAction("Save Image", self)
        save.setShortcut("Ctrl+S")
        save.triggered.connect(lambda: self.save())

        self.menubar.addAction(open)
        self.menubar.addAction(upscale)
        self.menubar.addAction(save)

    def open(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "PNG Files (*.png)", options=options)
        if filename:
            self.filename = filename

            # load image
            image = Image.open(self.filename)

            # convert image to numpy array
            lr_image = asarray(image)

            # convert numpy array to Qimage
            height, width, channel = lr_image.shape
            bytesPerLine = 3 * width
            qImg = QImage(lr_image.data, width, height, bytesPerLine, QImage.Format_RGB888)

            # convert Qimage to pixmap
            self.pixmap = QPixmap(QPixmap.fromImage(qImg))
            self.pixmap = self.pixmap.scaled(self.label.size(), Qt.KeepAspectRatio)

            # output image on screen
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setPixmap(self.pixmap)

    def upscale(self):
        if len(self.filename) != 0:
            self.upscale_image = evaluate(self.filename)

            # convert numpy array to Qimage
            height, width, channel = self.upscale_image.shape
            bytesPerLine = 3 * width
            qImg = QImage(self.upscale_image.data, width, height, bytesPerLine, QImage.Format_RGB888)

            # convert Qimage to pixmap
            self.pixmap = QPixmap(QPixmap.fromImage(qImg))
            self.pixmap = self.pixmap.scaled(self.label.size(), Qt.KeepAspectRatio)

            # output image on screen
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setPixmap(self.pixmap)

    def save(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "upscale image.png", "PNG Files (*.png)", options=options)
        if filename:
            scipy.misc.imsave(filename, self.upscale_image)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "SRGAN"))
