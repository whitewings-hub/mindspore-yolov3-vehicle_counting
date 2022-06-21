# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# /scm/data/xianyu/det/visdrone/models/mindspore-21-days-tutorials-main/chapter4/basketball-dataset/test/00086.jpg
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""YoloV3 eval."""
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


def network_load(pretrained_path='./models/2022-02-28_time_15_17_05/ckpt_0/yolov3-320_517440.ckpt'):
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
    # args.logger.info('Start inference....')
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


if __name__ == "__main__":
    model = network_load(pretrained_path='./models/2022-02-28_time_15_17_05/ckpt_0/yolov3-320_517440.ckpt')
    img_predict(model, "./imgs/0000002_00005_d_0000014.jpg")
