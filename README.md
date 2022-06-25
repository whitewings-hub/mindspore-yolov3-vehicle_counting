# mindspore-yolov3-vehicle_counting
training mindspore yolov3 model and counting vehicle

# 简介

## code_1

在MindSpore ModelZoo中YOLOv3-DarkNet53 [https://gitee.com/mindspore/models/tree/r1.5/official/cv/yolov3_darknet53] 的基础上增加了albumentations数据增强以及将原本的NMS修改为DIoU-NMS

## code_2

在 MindSpore YOLOv3-DarkNet53的开源项目 [https://github.com/leonwanghui/ms-yolov3-basketball] 的基础上增加了视频检测以及对检测到的目标进行计数
