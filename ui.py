
import os
import scipy
import numpy as np
import tensorlayer as tl
from PIL import Image
from model import get_G
from numpy import asarray
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QAction, QFileDialog
from PyQt5.QtCore import Qt, QRunnable, pyqtSlot, QThreadPool

checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)


class Ui_mainWindow(object):
    filename = ""
    has_lr_image = False
    has_upscale_image = False

    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(800, 600)
        mainWindow.setMinimumSize(QtCore.QSize(800, 600))
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
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
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setEnabled(True)
        self.progressBar.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.progressBar.setAutoFillBackground(False)
        self.progressBar.setStyleSheet("")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setTextVisible(False)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
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

        self.threadpool = QThreadPool()

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

    def upscale(self):
        if len(self.filename) != 0:
            # start process in new Thread
            worker = UpscaleWorker(self.evaluate, self.filename)
            self.threadpool.start(worker)

    def save(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "upscale image.png", "PNG Files (*.png)",
                                                  options=options)
        if filename:
            scipy.misc.imsave(filename, self.upscale_image)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "SRGAN"))

    def evaluate(self, lr_img_path):
        # update progressBar value
        self.progressBar.setValue(11)

        # load image
        image = Image.open(lr_img_path)

        # convert image to numpy array
        lr_image = asarray(image)

        # update progressBar value
        self.progressBar.setValue(15)

        # define model
        G = get_G([1, None, None, 3])
        G.load_weights(os.path.join(checkpoint_dir, 'g.h5'))
        G.eval()

        # rescale to ［－1, 1]
        lr_image = (lr_image / 127.5) - 1
        lr_image = np.asarray(lr_image, dtype=np.float32)
        lr_image = lr_image[np.newaxis, :, :, :]

        # update progressBar value
        self.progressBar.setValue(33)

        # get upscale image
        out = G(lr_image).numpy()

        # update progressBar value
        self.progressBar.setValue(57)

        # save SRGAN upscale image and rescale from [-1,  1] to [0, 255]
        tl.vis.save_image(out[0], './gen.png')

        image = Image.open('./gen.png')
        self.upscale_image = asarray(image)
        os.remove('./gen.png')

        self.has_upscale_image = True

        # update progressBar value
        self.progressBar.setValue(77)

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
        self.progressBar.setValue(100)


class UpscaleWorker(QRunnable):

    def __init__(self, fn, filename):
        super(UpscaleWorker, self).__init__()
        self.fn = fn
        self.args = filename

    @pyqtSlot()
    def run(self):
        self.fn(self.args)
