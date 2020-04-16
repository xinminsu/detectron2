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
#im = cv2.imread("/home/sxm/cocodataset/data/images/8.jpg")
im = cv2.imread("/home/sxm/cocodataset/ftb/images/train/1A3.jpg")
pyplot.figure()
pyplot.imshow(im[:, :, ::-1])
pyplot.show()

#
cfg = get_cfg()
cfg.merge_from_file("/home/sxm/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  #模型阈值
cfg.MODEL.WEIGHTS = "/home/sxm/detectron2/model/model_final_f10217.pkl"
#cfg.MODEL.WEIGHTS = "/home/sxm/detectron2/detectron2/tools/output/model_final.pth"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

#
pred_classes = outputs["instances"].pred_classes
pred_boxes = outputs["instances"].pred_boxes

#在原图上画出检测结果
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
pyplot.figure(2)
pyplot.imshow(v.get_image()[:, :, ::-1])
pyplot.show()
