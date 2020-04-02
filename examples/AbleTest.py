
import numpy as np
import cv2
#from matplotlib import pyplot
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import nn


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog



cfg = get_cfg()
cfg.merge_from_file("D:/detectron2-master/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

#
myDevice = "cpu"
num_channels = len(cfg.MODEL.PIXEL_MEAN)
pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(myDevice).view(num_channels, 1, 1)
pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(myDevice).view(num_channels, 1, 1)
print(cfg.MODEL.PIXEL_MEAN)
print(cfg.MODEL.PIXEL_STD)
print("-----------------------")
print("num_channels is "+str(num_channels))
print("pixel_mean is "+str(pixel_mean))
print("pixel_std is "+str(pixel_std))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  #模型阈值
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"



