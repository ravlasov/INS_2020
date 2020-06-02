import sys
from model import RecognitionModel
from utils import *
import numpy as np
from PySide2.QtCore import Qt
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QMessageBox
from sys import platform

class MainWindow(QDialog):
    drop_image_text = "Drag & Drop\nimage here"
    drop_model_text = "Drag & Drop\nmodel here"
    predict_button_text = "Predict"
    usual_stylesheet = "border: 3px solid black; color: black"
    has_model_stylesheet = "border: 3px solid black; color: black; font-size: 200px"
    dragged_stylesheet = "border: 3px solid blue; color: blue"
    dragged_has_model_stylesheet = "border: 3px solid blue; color: blue; font-size: 200px"
    error_stylesheet = "border: 3px solid red; color: red"
    image_box_size = 400
    model_box_size = 400
    predict_button_height = 50
    indents_size = 20
    image_path = None
    model_path = None

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.model = RecognitionModel()
        self.setFixedSize(self.image_box_size + self.model_box_size + 3 * self.indents_size,
                          self.image_box_size + self.predict_button_height + 3 * self.indents_size)
        self.setWindowTitle("idz")
        self.setAcceptDrops(True)

        self.image_label = self.get_drop_image_label()
        self.model_label = self.get_drop_model_label()
        self.predict_button = self.get_predict_button()

        layout = QHBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.model_label)

        window_layout = QVBoxLayout()
        window_layout.addLayout(layout)
        window_layout.addWidget(self.predict_button)

        self.setLayout(window_layout)

    def show_prediction(self, prediction):
        message = QMessageBox(QMessageBox.NoIcon, "Prediction", prediction)
        message.layout().setMargin(0)
        message.layout().setSpacing(0)
        message.setStyleSheet("QLabel{max-width: 95px; min-height: 100px; font-size:100px; font-weight: bold;}"
                              "QPushButton{min-width: 120px;min-height: 30px; font-size:25px;}")
        message.exec_()

    def predict(self):
        if self.image_path is not None and self.model_path is not None:
            img = load_image(self.image_path)
            img = np.reshape(img, (1, *(img.shape)))
            prediction = self.model.predict(img)
            self.show_prediction(prediction)
            return
        if self.image_path is None:
            self.image_label.setStyleSheet(self.error_stylesheet)
        if self.model_path is None:
            self.model_label.setStyleSheet(self.error_stylesheet)

    def get_drop_model_label(self):
        label = QLabel(self.drop_model_text)
        label.setAlignment(Qt.AlignCenter)
        label.setAutoFillBackground(True)
        font = label.font()
        font.setPointSize(30)
        font.setBold(True)
        label.setFont(font)
        label.setFixedSize(self.model_box_size, self.model_box_size)
        label.setStyleSheet(self.usual_stylesheet)
        return label

    def get_drop_image_label(self):
        label = QLabel(self.drop_image_text)
        label.setAlignment(Qt.AlignCenter)
        label.setAutoFillBackground(True)
        font = label.font()
        font.setPointSize(30)
        font.setBold(True)
        label.setFont(font)
        label.setFixedSize(self.image_box_size, self.image_box_size)
        label.setStyleSheet(self.usual_stylesheet)
        return label

    def get_predict_button(self):
        btn = QPushButton(self.predict_button_text)
        btn.setFixedHeight(self.predict_button_height)
        font = btn.font()
        font.setPointSize(25)
        btn.setFont(font)
        btn.clicked.connect(self.predict)
        return btn

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and len(e.mimeData().urls()) == 1:
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        c = self.childAt(e.pos())
        if c is not None:
            if c is self.image_label:
                self.image_label.setStyleSheet(self.dragged_stylesheet)
                self.model_label.setStyleSheet(
                    self.usual_stylesheet if self.model_path is None else self.has_model_stylesheet)

            elif c is self.model_label:
                self.image_label.setStyleSheet(self.usual_stylesheet)
                self.model_label.setStyleSheet(self.dragged_stylesheet if self.model_path is None else self.dragged_has_model_stylesheet)
            else:
                self.image_label.setStyleSheet(self.usual_stylesheet)
                self.model_label.setStyleSheet(
                    self.usual_stylesheet if self.model_path is None else self.has_model_stylesheet)


        else:
            self.image_label.setStyleSheet(self.usual_stylesheet)
            self.model_label.setStyleSheet(
                self.usual_stylesheet if self.model_path is None else self.has_model_stylesheet)

    def dragLeaveEvent(self, e):
        self.image_label.setStyleSheet(self.usual_stylesheet)
        self.model_label.setStyleSheet(self.usual_stylesheet if self.model_path is None else self.has_model_stylesheet)

    def dropEvent(self, e):
        self.image_label.setStyleSheet(self.usual_stylesheet)
        self.model_label.setStyleSheet(self.usual_stylesheet if self.model_path is None else self.has_model_stylesheet)
        c = self.childAt(e.pos())
        url = e.mimeData().urls()[0].path()
        if platform == "win32":
            url = '//'.join(url.rsplit('/', 1))
            if url.startswith("/"):
                url = url[1:]
        if c is not None:
            if c is self.image_label and url.endswith(".png"):
                self.image_path = url
                self.image_label.setPixmap(QPixmap(self.image_path).scaled(400, 400, Qt.KeepAspectRatio))
            elif c is self.model_label and url.endswith(".hdf5"):
                self.model_path = url
                self.model_label.setText('✓')
                self.model_label.setStyleSheet(self.has_model_stylesheet)
                self.model.load_model(url)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
