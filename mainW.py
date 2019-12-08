import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from MtcnnDetector import MtcnnDetector
from detector import Detector
from fcn_detector import FcnDetector
from model import P_Net, R_Net, O_Net
import config
import os
import time
import copy


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.width(), self.height())
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.icons = []
        self.gen_iconlist()
        self.coverflag = False
        self.embedingList = []
        self.mtcnn_detector = self.load_align()
        self.images_placeholder = None
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        self.keep_probability_placeholder = None
        self.sess = None
        self.count = 0
        self.iconPos = []
        self.iconclass = []
        self.sess_init()

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()
        self.button_open_camera = QtWidgets.QPushButton('打开相机')
        self.button_close = QtWidgets.QPushButton('退出')
        self.button_cover = QtWidgets.QPushButton('遮盖')
        self.button_select = QtWidgets.QPushButton('已选 0 人')
        self.button_open_camera.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)
        self.button_cover.setMinimumHeight(50)
        self.button_select.setMinimumHeight(50)

        self.label_show_camera = QtWidgets.QLabel()
        self.label_show_camera.setFixedSize(
            641, 481)
        self.__layout_fun_button.addWidget(
            self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_select)
        self.__layout_fun_button.addWidget(self.button_cover)
        self.__layout_fun_button.addWidget(
            self.button_close)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(
            self.label_show_camera)
        self.setLayout(self.__layout_main)

    def slot_init(self):
        self.button_open_camera.clicked.connect(
            self.button_open_camera_clicked)
        self.timer_camera.timeout.connect(
            self.show_camera)
        self.button_close.clicked.connect(self.close)
        self.button_cover.clicked.connect(self.startCover)
        self.button_select.clicked.connect(self.genEmbedings)

    def startCover(self):
        if self.coverflag:
            self.coverflag = False
            self.button_cover.setText('遮盖')
        else:
            self.coverflag = True
            self.button_cover.setText('原图')

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() is False:
            flag = self.cap.open(self.CAM_NUM)
            if flag is False:
                msg = QtWidgets.QMessageBox.warning(
                    self, 'warning', "未检测到相机！", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(10)
                self.button_open_camera.setText('关闭相机')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText('打开相机')

    def gen_iconlist(self):
        for i in range(10):
            icon = Image.open("../icons/" + str(i) + ".png")
            self.icons.append(icon)

    def addIcon(self, img0, classList, posList):
        if not len(classList):
            return img0
        img1 = Image.fromarray(img0)
        for i, pos in enumerate(posList):
            icon = self.icons[classList[i] % 10]
            icon = icon.resize(
                (pos[2] - pos[0], pos[3] - pos[1]), Image.ANTIALIAS)
            layer = Image.new('RGBA', img1.size, (0, 0, 0, 0))
            layer.paste(icon, (pos[0], pos[1]))
            img1 = Image.composite(layer, img1, layer)
        return np.asarray(img1)

    def show_camera(self):
        self.count += 1
        self.count %= 2
        flag, self.image = self.cap.read()
        if not flag:
            return
        show = cv2.resize(self.image, (640, 480))
        show = cv2.flip(show, 1)
        if len(self.embedingList) and not self.count and self.coverflag:
            self.genIDPos(copy.deepcopy(show))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        if self.coverflag:
            show = self.addIcon(show, self.iconclass, self.iconPos)
        showImage = QtGui.QImage(
            show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(
            QtGui.QPixmap.fromImage(showImage))

    def load_align(self):
        thresh = config.thresh
        min_face_size = config.min_face
        stride = config.stride
        test_mode = "ONet"
        detectors = [None, None, None]
        model_path = ['./model/PNet/',
                      './model/RNet/', './model/ONet']
        batch_size = config.batches
        PNet = FcnDetector(P_Net, model_path[0])
        detectors[0] = PNet

        if test_mode in ["RNet", "ONet"]:
            RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
            detectors[1] = RNet

        if test_mode == "ONet":
            ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
            detectors[2] = ONet

        mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                       stride=stride, threshold=thresh)
        return mtcnn_detector

    def align_face_init(self, img):
        scaled_arr = []
        class_names_arr = []
        try:
            boxes_c, _ = self.mtcnn_detector.detect(img)
        except:
            print('未检测到人脸！\n')
            return None, None
        num_box = boxes_c.shape[0]
        if num_box > 0:
            det = boxes_c[:, :4]
            det_arr = []
            img_size = np.asarray(img.shape)[:2]
            if num_box > 1:
                score = boxes_c[:, 4]
                index = np.argmax(score)
                det_arr.append(det[index, :])
            else:
                det_arr.append(np.squeeze(det))
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = [
                    int(max(det[0], 0)),
                    int(max(det[1], 0)),
                    int(min(det[2], img_size[1])),
                    int(min(det[3], img_size[0]))
                ]
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

                scaled = cv2.resize(
                    cropped,
                    (160, 160), interpolation=cv2.INTER_LINEAR)
                scaled = cv2.cvtColor(
                    scaled, cv2.COLOR_BGR2RGB) - 127.5 / 128.0
                scaled_arr.append(scaled)
                class_names_arr.append(i)

        else:
            print('无法对齐人脸')
            return None, None
        scaled_arr = np.asarray(scaled_arr)
        class_names_arr = np.asarray(class_names_arr)
        return scaled_arr, class_names_arr

    def align_face(self, img):
        try:
            boxes_c, _ = self.mtcnn_detector.detect(img)
        except:
            print('未检测到人脸')
            return None, None, None
        num_box = boxes_c.shape[0]
        scaled_arr = []
        recList = []
        if num_box > 0:
            det = boxes_c[:, :4]
            det_arr = []
            img_size = np.asarray(img.shape)[:2]
            for i in range(num_box):
                det_arr.append(np.squeeze(det[i]))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = [int(max(det[0], 0)), int(max(det[1], 0)), int(
                    min(det[2], img_size[1])), int(min(det[3], img_size[0]))]
                recList.append(bb)
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = cv2.resize(cropped, (160, 160),
                                    interpolation=cv2.INTER_LINEAR)
                scaled = cv2.cvtColor(
                    scaled, cv2.COLOR_BGR2RGB) - 127.5 / 128.0
                scaled_arr.append(scaled)
            scaled_arr = np.array(scaled_arr)
            return img, scaled_arr, recList
        else:
            print('未检测到人脸')
            return None, None, None

    def sess_init(self):
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state('../model/')
        saver = tf.train.import_meta_graph(
            ckpt.model_checkpoint_path + '.meta')
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name(
            "input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name(
            "embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph(
        ).get_tensor_by_name("phase_train:0")
        self.keep_probability_placeholder = tf.get_default_graph(
        ).get_tensor_by_name('keep_probability:0')
        feed_dict = {
            self.images_placeholder: np.zeros([1, 160, 160, 3]),
            self.phase_train_placeholder: False,
            self.keep_probability_placeholder: 1.0
        }
        print("正在加载模型，请稍等！")
        tmp = self.sess.run(self.embeddings, feed_dict=feed_dict)
        print("启动完成！")

    def genEmbedings(self):
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "警告", "请先打开相机")
            return
        else:
            time.sleep(0.5)
            flag, img = self.cap.read()
        if flag:
            self.timer_camera.stop()
            QtWidgets.QMessageBox.information(self, "提示", "选取成功")
        else:
            QtWidgets.QMessageBox.warning(self, "警告", "选取失败")
            return
        scaled_arr, class_arr = self.align_face_init(img)
        feed_dict = {
            self.images_placeholder: scaled_arr,
            self.phase_train_placeholder: False,
            self.keep_probability_placeholder: 1.0
        }
        embs = self.sess.run(self.embeddings, feed_dict=feed_dict)
        self.embedingList.append(embs)
        self.button_select.setText('已选 ' + str(len(self.embedingList)) + ' 人')
        self.timer_camera.start(10)

    def genIDPos(self, img):
        img, scaled_arr, recList = self.align_face(img)
        if scaled_arr is not None:
            feed_dict = {self.images_placeholder: scaled_arr,
                         self.phase_train_placeholder: False, self.keep_probability_placeholder: 1.0}
            embs = self.sess.run(self.embeddings, feed_dict=feed_dict)
            face_num = embs.shape[0]
            face_class = []
            icons_pos_scale = []
            for i in range(face_num):
                diff = []
                for man in self.embedingList:
                    diff.append(np.mean(np.square(embs[i] - man), axis=1))
                min_diff = min(diff)
                if min_diff < 0.002:
                    face_class.append(np.argmin(diff))
                    icons_pos_scale.append(recList[i])
                else:
                    break
            self.iconclass = face_class
            self.iconPos = icons_pos_scale
        else:
            self.iconclass = []
            self.iconPos = []


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
