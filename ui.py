
import os
import numpy as np
import tensorlayer as tl
from PIL import Image
from model import get_G
from numpy import asarray
from scipy.misc import imsave
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import QFileDialog, QMessageBox

checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)


class Ui_mainWindow(QtWidgets.QMainWindow):
    filename = ""
    has_lr_image = False
    has_upscale_image = False
    model_name = ""

    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(800, 600)
        mainWindow.setMinimumSize(QtCore.QSize(400, 300))

        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("background-color: rgb(245, 245, 245);")

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setStyleSheet("color: rgb(188, 188, 188);"
                                 "background-color: rgb(230, 230, 230);")

        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setEnabled(True)
        self.progressBar.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.progressBar.setAutoFillBackground(False)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setTextVisible(False)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setStyleSheet("QProgressBar{ "
                                       + "border-radius: 7px;"
                                       + "height: 10px;"
                                       + "border: 1px solid rgb(188, 188, 188);"
                                       + "}"
                                       + "QProgressBar::chunk {"
                                       + "border-radius: 6px;"
                                       + "}")
        self.progressBar.valueChanged.connect(self.progressBar_update)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)

        mainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")

        mainWindow.setMenuBar(self.menubar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

        self.menu_open = self.menubar.addAction("Open File")
        self.menu_open.setShortcut("Ctrl+O")
        self.menu_open.triggered.connect(lambda: self.open())

        self.menuUpscale = self.menubar.addMenu("Upscale Image")

        self.menu_upscale_clear = self.menuUpscale.addAction("Upscale Image without defects")
        self.menu_upscale_clear.setShortcut("Ctrl+W")
        self.menu_upscale_clear.triggered.connect(lambda: self.upscale("g_clear.h5"))

        self.menu_upscale_doppler = self.menuUpscale.addAction("Upscale Image with Doppler effect")
        self.menu_upscale_doppler.setShortcut("Ctrl+D")
        self.menu_upscale_doppler.triggered.connect(lambda: self.upscale("g_doppler.h5"))

        self.menu_upscale_noise = self.menuUpscale.addAction("Upscale Image with Noise")
        self.menu_upscale_noise.setShortcut("Ctrl+E")
        self.menu_upscale_noise.triggered.connect(lambda: self.upscale("g_noise.h5"))

        self.menu_save = self.menubar.addAction("Save Image")
        self.menu_save.setShortcut("Ctrl+S")
        self.menu_save.triggered.connect(lambda: self.save())

    def progressBar_update(self):
        self.progressBar.setStyleSheet("QProgressBar{ "
                                       + "border-radius: 7px;"
                                       + "height: 10px;"
                                       + "border: 1px solid rgb(188, 188, 188);"
                                       + "}"
                                       + "QProgressBar::chunk {"
                                       + "background-color: hsv({},{},{});".format((self.progressBar.value() * 115 / 100), 155, 255)
                                       + "border-radius: 6px;"
                                       + "}")

    def resizeEvent(self, *args):
        if self.has_lr_image:
            if self.has_upscale_image:
                # convert numpy array to Qimage
                height, width, channel = self.upscale_image.shape
                bytesPerLine = 3 * width
                qImg = QImage(self.upscale_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            else:
                # convert numpy array to Qimage
                height, width, channel = self.lr_image.shape
                bytesPerLine = 3 * width
                qImg = QImage(self.lr_image.data, width, height, bytesPerLine, QImage.Format_RGB888)

            # convert Qimage to pixmap
            self.pixmap = QPixmap(QPixmap.fromImage(qImg))
            self.pixmap = self.pixmap.scaled(self.label.size(), Qt.KeepAspectRatio)

            # output image on screen
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setPixmap(self.pixmap)

    def on_progress_changed(self, value):
        self.progressBar.setValue(value)

    def open(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "PNG Files (*.png)", options=options)
        if filename:
            self.filename = filename

            # load image
            image = Image.open(self.filename)

            # convert image to numpy array
            self.lr_image = asarray(image)
            self.has_lr_image = True

            # convert numpy array to Qimage
            height, width, channel = self.lr_image.shape
            bytesPerLine = 3 * width
            qImg = QImage(self.lr_image.data, width, height, bytesPerLine, QImage.Format_RGB888)

            # convert Qimage to pixmap
            self.pixmap = QPixmap(QPixmap.fromImage(qImg))
            self.pixmap = self.pixmap.scaled(self.label.size(), Qt.KeepAspectRatio)

            # output image on screen
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setPixmap(self.pixmap)

            QMessageBox.about(self, 'Info', "Image successfully loaded")

    def upscale(self, model_name):
        self.model_name = model_name
        if self.filename:
            # start process in new Thread
            self.progressBar.setValue(0)
            self.progressBar.setTextVisible(True)
            self.menuUpscale.setDisabled(True)
            self.worker = UpscaleWorker(self.evaluate, self.filename)
            self.worker.value_changed.connect(self.on_progress_changed)
            self.worker.start()
        else:
            QMessageBox.about(self, 'Info', "You should Open Image to upscale it")

    def save(self):
        if self.has_lr_image or self.has_upscale_image:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "upscale image.png", "PNG Files (*.png)",
                                                      options=options)
            if filename:
                if self.has_upscale_image:
                    imsave(filename, self.upscale_image)
                elif self.has_lr_image:
                    imsave(filename, self.lr_image)
                QMessageBox.about(self, 'Info', "Image successfully saved")
        else:
            QMessageBox.about(self, 'Info', "You should Open Image to save it")

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "SRGAN"))

    def evaluate(self, lr_img_path):
        # update progressBar value
        self.worker.value_changed.emit(11)

        # load image
        image = Image.open(lr_img_path)

        # convert image to numpy array
        lr_image = asarray(image)

        # update progressBar value
        self.worker.value_changed.emit(15)

        # define model
        G = get_G([1, None, None, 3])
        G.load_weights(os.path.join(checkpoint_dir, self.model_name))
        G.eval()

        # rescale to ［－1, 1]
        lr_image = (lr_image / 127.5) - 1
        lr_image = np.asarray(lr_image, dtype=np.float32)
        lr_image = lr_image[np.newaxis, :, :, :]

        # update progressBar value
        self.worker.value_changed.emit(33)

        # get upscale image
        out = G(lr_image).numpy()

        # release model's memory
        G.release_memory()

        # update progressBar value
        self.worker.value_changed.emit(57)

        # save SRGAN upscale image and rescale from [-1,  1] to [0, 255]
        tl.vis.save_image(out[0], './gen.png')

        image = Image.open('./gen.png')
        self.upscale_image = asarray(image)
        os.remove('./gen.png')

        self.has_upscale_image = True

        # update progressBar value
        self.worker.value_changed.emit(77)

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

        # update progressBar value
        self.worker.value_changed.emit(100)
        self.menuUpscale.setDisabled(False)
        self.worker.exit()


class UpscaleWorker(QThread):
    value_changed = pyqtSignal(int)

    def __init__(self, fn, filename):
        super().__init__()
        self.fn = fn
        self.args = filename

    def run(self):
        self.fn(self.args)
