import numpy as np
import cv2
from matplotlib import pyplot

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# 下载测试图片:
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
im = cv2.imread("/home/sxm/pic/2020-03-31-173345.jpg")
pyplot.figure()
pyplot.imshow(im[:, :, ::-1])
pyplot.show()

#
cfg = get_cfg()
cfg.merge_from_file("/home/sxm/detectron2/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
cfg.MODEL.WEIGHTS = "/home/sxm/detectron2/model/model_final_cafdb1.pkl"
predictor = DefaultPredictor(cfg)
panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
pyplot.imshow(v.get_image()[:, :, ::-1])
pyplot.show()