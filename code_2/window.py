# -*- coding: utf-8 -*-

import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import os.path as osp
import numpy as np
import os
import argparse
import datetime
import sys
from collections import defaultdict
import cv2
import numpy as np
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.yolo import YOLOV3DarkNet53
from src.logger import get_logger
from src.config import ConfigYOLOV3DarkNet53
from src.transforms import _reshape_data
from predict import DetectionEngine
import warnings

warnings.filterwarnings('ignore')
envpath = '/home/whitewings/miniconda3/envs/mind15/lib/python3.7/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

label_list = ['ignored region', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle',
              'bus', 'motor', 'others']


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser('mindspore coco testing')
    # device related
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: GPU)')
    # dataset related
    parser.add_argument('--output_dir', type=str, default='./', help='image file output folder')
    parser.add_argument('--per_batch_size', default=1, type=int, help='batch size for per gpu')

    # logging related
    parser.add_argument('--log_path', type=str, default='outputs/', help='checkpoint save location')
    # detect_related
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='threshold for NMS')
    parser.add_argument('--ignore_threshold', type=float, default=0.1,
                        help='threshold to throw low quality boxes')

    args, _ = parser.parse_known_args()
    return args


def data_preprocess(img_path, config):
    img = cv2.imread(img_path, 1)
    img, ori_image_shape = _reshape_data(img, config.test_img_shape)
    img = img.transpose(2, 0, 1)

    return img, ori_image_shape


def network_load(pretrained_path='./models/yolov3-320_517440.ckpt'):
    """The function of predict."""
    args = parse_args()
    devid = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        save_graphs=False, device_id=devid)

    # logger
    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    args.logger = get_logger(args.outputs_dir, rank_id)

    args.logger.info('Creating Network....')
    network = YOLOV3DarkNet53(is_training=False)

    if os.path.isfile(pretrained_path):
        param_dict = load_checkpoint(pretrained_path)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(pretrained_path))
    else:
        args.logger.info('{} not exists or not a pre-trained file'.format(args.pretrained))
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(args.pretrained))
        exit(1)
    return network


def img_predict(network, image_path=""):
    args = parse_args()
    config = ConfigYOLOV3DarkNet53()
    # data preprocess operation
    image, image_shape = data_preprocess(image_path, config)

    # init detection engine
    detection = DetectionEngine(args)

    input_shape = Tensor(tuple(config.test_img_shape), ms.float32)
    network.set_train(False)
    prediction = network(Tensor(image.reshape(1, 3, 416, 416), ms.float32), input_shape)
    output_big, output_me, output_small = prediction
    output_big = output_big.asnumpy()
    output_me = output_me.asnumpy()
    output_small = output_small.asnumpy()

    detection.detect([output_small, output_me, output_big], args.per_batch_size,
                     image_shape, config)
    detection.do_nms_for_results()
    img = detection.draw_boxes_in_image(image_path)

    cv2.imwrite(os.path.join(args.output_dir, 'imgs/output.jpg'), img)


class MainWindow(QTabWidget):

    def __init__(self):

        super().__init__()
        self.setWindowTitle('基于MindSpore框架的YOLOv3目标检测程序')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/xiaoban.jpg"))

        self.output_size = 480
        self.img2predict = ""

        self.origin_shape = ()

        self.model = network_load()
        self.initUI()


    def initUI(self):

        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)

        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()

        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()

        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(34,139,34)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(34,139,34)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")

        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图像检测')

        self.setTabIcon(0, QIcon('images/UI/xiaoban.jpg'))


    def upload_img(self):

        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
 
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)

            self.img2predict = fileName
            self.origin_shape = (im0.shape[1], im0.shape[0])
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))

            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))



    def detect_img(self):

        source = self.img2predict

        img_predict(self.model, source)
        img_src = cv2.imread("imgs/output.jpg")
        im0 = cv2.resize(img_src, self.origin_shape)
        cv2.imwrite("images/tmp/single_result.jpg", im0)
        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))


    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
