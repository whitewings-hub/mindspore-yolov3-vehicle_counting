# 基于MindSpore YOLOv3-Darknet53的车辆检测与计数实现

## 数据集下载

VisDrone数据集下载 [http://aiskyeye.com/download/object-detection-2/]

## 数据集处理

将原始VisDrone数据集转换为coco格式，然后存放在本地目录

```
python VisDrone2coco.py
```

## 环境配置

Ubuntu 18.04  
python 3.7.5  
mindspore-gpu 1.5.2

## 预训练权重

预训练权重获取参考[https://gitee.com/mindspore/models/blob/r1.5/official/cv/yolov3_darknet53/README_CN.md]

## 模型训练

```
python train.py --data_dir ../train/  --pretrained_backbone ./ckpt_files/backbone_darknet53.ckpt --lr=0.1 --T_max=320 --max_epoch=320 --warmup_epochs=4 --training_shape=416 --lr_scheduler=cosine_annealing
```

## 模型验证

```
python eval.py --data_dir ../val/ --pretrained ./models/yolov3-320_517440.ckpt
```

## 模型使用

```
python predict.py --image_path ./imgs/0000002_00448_d_0000015.jpg --pretrained ./models/yolov3-320_517440.ckpt
```

## 基于图像的车辆检测与计数

```
python detect_img.py
```
需要修改detect_img.py文件中的pretrained_path为ckpt文件路径  
需要修改detect_img.py文件中最后一行对应检测图片的路径
```
python window.py
```
需要修改window.py文件中的pretrained_path为ckpt文件路径

## 基于视频的车辆检测与计数

```
python detect_vid.py
```
需要修改detect_vid.py文件中的pretrained_path为ckpt文件路径  
需要在cap = cv2.VideoCapture()中添加视频路径  
cap = cv2.VideoCapture(0)则为检测电脑摄像头的实时视频
