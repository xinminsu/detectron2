# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserv
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os
# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from detectron2.data.datasets import load_coco_json

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

WINDOW_NAME = "detections"

# inference
INPUT_IMG_PATH = '/home/sxm/cocodataset/ftb/images/train'
OUTPUT_IMG_PATH = '/home/sxm/cocodataset/ftb/out'

# 数据集路径
DATASET_ROOT = '/home/sxm/cocodataset/ftb'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'images', 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'images', 'val')
TRAIN_JSON = os.path.join(ANN_ROOT, 'train', 'trainval.json')
VAL_JSON = os.path.join(ANN_ROOT, 'val', 'trainval.json')

# 数据集类别元数据
DATASET_CATEGORIES = [
    {"name": "ball", "id": 1, "isthing": 1, "color": [220, 20, 60]},
    {"name": "man", "id": 2, "isthing": 1, "color": [219, 142, 185]},
]

# 数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "football": (TRAIN_PATH, TRAIN_JSON),
    "footballval": (VAL_PATH, VAL_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   metadate=get_dataset_instances_meta(),
                                   json_file=json_file,
                                   image_root=image_root)


def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_dataset_id_to_contiguous_id = {0: 0, 1: 1}
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco",
                                  **metadate)


# 注册数据集和元数据
def plain_register_dataset():
    DatasetCatalog.register("football", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "football"))
    MetadataCatalog.get("football").set(thing_classes=["ball", "man"],
                                          json_file=TRAIN_JSON,
                                          image_root=TRAIN_PATH)
    DatasetCatalog.register("footballval", lambda: load_coco_json(VAL_JSON, VAL_PATH, "footballval"))
    MetadataCatalog.get("footballval").set(thing_classes=["ball", "man"],
                                        json_file=VAL_JSON,
                                        image_root=VAL_PATH)

from detectron2.data.datasets import register_coco_instances
#register_coco_instances("fruits_nuts", {}, "/home/sxm/cocodataset/data/trainval.json", "/home/sxm/cocodataset/data/images")

register_dataset()

fruits_nuts_metadata = MetadataCatalog.get("football")
dataset_dicts = DatasetCatalog.get("football")

import random

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    #cv2_imshow(vis.get_image()[:, :, ::-1])
    plt.figure()
    plt.imshow(vis.get_image()[:, :, ::-1])
    plt.show()