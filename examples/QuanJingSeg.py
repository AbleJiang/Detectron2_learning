
# https://www.aiuai.cn/aifarm1288.html


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
cfg.merge_from_file("D:/detectron2-master/detectron2/model_zoo/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
cfg.MODEL.WEIGHTS = "detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl"
predictor = DefaultPredictor(cfg)
panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
#plt.imshow(v.get_image()[:, :, ::-1])
#plt.show()



cv2.namedWindow('output_image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('output_image', v.get_image())
cv2.waitKey(0)

