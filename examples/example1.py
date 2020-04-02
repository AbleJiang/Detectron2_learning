import matplotlib; matplotlib.use('TkAgg')

import numpy as np
import cv2
#from matplotlib import pyplot
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# 下载测试图片:
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
im = cv2.imread("d:/tmp/input.jpg")
#plt.figure()
#plt.imshow(im[:, :, ::-1])
#plt.show()
cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('input_image', im)
#cv2.waitKey(0)


cfg = get_cfg()
cfg.merge_from_file("D:/detectron2-master/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  #模型阈值
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

pred_classes = outputs["instances"].pred_classes
pred_boxes = outputs["instances"].pred_boxes

#在原图上画出检测结果
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#plt.figure(2)
#plt.imshow(v.get_image()[:, :, ::-1])
#plt.show()

cv2.namedWindow('output_image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('output_image', v.get_image())
cv2.waitKey(0)

