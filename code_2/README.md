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
python predict.py --image_path ./imgs/0000077_00922_d_0000003.jpg --pretrained ./ckpt_files/608_32batch_new/0-320_64640.ckpt
```

## 基于图像的车辆检测与计数

```
python detect_img.py
```
```
python window.py
```

## 基于视频的车辆检测与计数

```
python detect_vid.py
```
