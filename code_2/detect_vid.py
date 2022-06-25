# -*- coding: utf-8 -*-

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


class DetectionEngine:
    """Detection engine."""

    def __init__(self, args):
        self.ignore_threshold = args.ignore_threshold
        self.labels = label_list
        self.num_classes = len(self.labels)
        self.results = defaultdict(list)
        self.det_boxes = []
        self.nms_thresh = args.nms_thresh

    def do_nms_for_results(self):
        """Get result boxes."""
        for clsi in self.results:
            dets = self.results[clsi]
            dets = np.array(dets)
            # keep_index = self._nms(dets, self.nms_thresh)
            keep_index = self._diou_nms(dets, self.nms_thresh)
            keep_box = [{'category_id': self.labels[int(clsi)],
                         'bbox': list(dets[i][:4].astype(float)),
                         'score': dets[i][4].astype(float)}
                        for i in keep_index]
            self.det_boxes.extend(keep_box)

    def _diou_nms(self, dets, thresh=0.6):

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            center_x1 = (x1[i] + x2[i]) / 2
            center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
            center_y1 = (y1[i] + y2[i]) / 2
            center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
            inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
            out_max_x = np.maximum(x2[i], x2[order[1:]])
            out_max_y = np.maximum(y2[i], y2[order[1:]])
            out_min_x = np.minimum(x1[i], x1[order[1:]])
            out_min_y = np.minimum(y1[i], y1[order[1:]])
            outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
            diou = ovr - inter_diag / outer_diag
            diou = np.clip(diou, -1, 1)
            inds = np.where(diou <= thresh)[0]
            order = order[inds + 1]
        return keep

    def _nms(self, predicts, threshold):
        """Calculate NMS."""
        # convert xywh -> xmin ymin xmax ymax
        x1 = predicts[:, 0]
        y1 = predicts[:, 1]
        x2 = x1 + predicts[:, 2]
        y2 = y1 + predicts[:, 3]
        scores = predicts[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)

            indexs = np.where(ovr <= threshold)[0]
            order = order[indexs + 1]
        return reserved_boxes

    def detect(self, outputs, batch, image_shape, config=None):
        """Detect boxes."""
        outputs_num = len(outputs)

        for batch_id in range(batch):
            for out_id in range(outputs_num):

                out_item = outputs[out_id]

                out_item_single = out_item[batch_id, :]

                dimensions = out_item_single.shape[:-1]
                out_num = 1
                for d in dimensions:
                    out_num *= d
                ori_w, ori_h = image_shape
                x = out_item_single[..., 0] * ori_w
                y = out_item_single[..., 1] * ori_h
                w = out_item_single[..., 2] * ori_w
                h = out_item_single[..., 3] * ori_h

                conf = out_item_single[..., 4:5]
                cls_emb = out_item_single[..., 5:]

                cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
                x = x.reshape(-1)
                y = y.reshape(-1)
                w = w.reshape(-1)
                h = h.reshape(-1)
                cls_emb = cls_emb.reshape(-1, config.num_classes)
                conf = conf.reshape(-1)
                cls_argmax = cls_argmax.reshape(-1)

                x_top_left = x - w / 2.
                y_top_left = y - h / 2.
                # create all False
                flag = np.random.random(cls_emb.shape) > sys.maxsize
                for i in range(flag.shape[0]):
                    c = cls_argmax[i]
                    flag[i, c] = True
                confidence = cls_emb[flag] * conf
                print(self.ignore_threshold)
                self.ignore_threshold = 0.1
                for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left, w, h, confidence, cls_argmax):
                    if confi < self.ignore_threshold:
                        continue
                    x_lefti = max(0, x_lefti)
                    y_lefti = max(0, y_lefti)
                    wi = min(wi, ori_w)
                    hi = min(hi, ori_h)
                    # transform catId to match coco
                    coco_clsi = str(clsi)
                    self.results[coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])

    def draw_boxes_in_image(self, img_path):
        num_record = [0 for i in range(12)]
        img = cv2.imread(img_path, 1)
        for i in range(len(self.det_boxes)):
            x = int(self.det_boxes[i]['bbox'][0])
            y = int(self.det_boxes[i]['bbox'][1])
            w = int(self.det_boxes[i]['bbox'][2])
            h = int(self.det_boxes[i]['bbox'][3])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 225, 0), 1)
            score = round(self.det_boxes[i]['score'], 3)
            classname = self.det_boxes[i]['category_id']
            text = self.det_boxes[i]['category_id'] + ', ' + str(score)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

            num_record[label_list.index(classname)] = num_record[label_list.index(classname)] + 1
        result_str = ""
        for ii in range(12):
            current_name = label_list[ii]
            current_num = num_record[ii]
            if current_num != 0:
                result_str = result_str + "{}:{} ".format(current_name, current_num)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, result_str, (20, 20), font, 0.5, (255, 0, 0), 2)

        return img


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
    return img


if __name__ == "__main__":
    model = network_load(pretrained_path='./models/yolov3-320_517440.ckpt')
    cap = cv2.VideoCapture("test.mp4")
    # cap = cv2.VideoCapture(0)
    while (1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imwrite("imgs/input.jpg", frame)
        result_img = img_predict(model, "imgs/input.jpg")
        cv2.imshow("capture", result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
