import numpy as np
import argparse
import cv2
from matplotlib import pyplot
import random
import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import multiprocessing as mp
from google.colab.patches import cv2_imshow

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import load_coco_json
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.utils.visualizer import ColorMode
# constants
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


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    args.config_file = "/home/sxm/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    #cfg.merge_from_file("/home/sxm/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = os.path.join("/home/sxm/cocodataset/ftb/output", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TRAIN = ("football",)
    cfg.DATASETS.TEST = ("footballval",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.INPUT.MAX_SIZE_TRAIN = 400
    cfg.INPUT.MAX_SIZE_TEST = 400
    cfg.INPUT.MIN_SIZE_TRAIN = (160,)
    cfg.INPUT.MIN_SIZE_TEST = 160
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 类别数

    cfg.SOLVER.IMS_PER_BATCH = 2  # batch_size=2; iteration = 1434/batch_size = 717 iters in one epoch
    ITERS_IN_ONE_EPOCH = int(1434 / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1  # 12 epochs
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (30000,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="/home/sxm/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

#cfg = get_cfg()
#cfg.merge_from_file("/home/sxm/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#cfg.MODEL.WEIGHTS = os.path.join("/home/sxm/cocodataset/ftb/output", "model_final.pth")
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
#cfg.DATASETS.TEST = ("footballval", )
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    predictor = DefaultPredictor(cfg)

    register_dataset()

    fruits_nuts_metadata = MetadataCatalog.get("footballval")
    dataset_dicts = DatasetCatalog.get("footballval")



    pyplot.figure()

    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                   metadata=fruits_nuts_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2_imshow(v.get_image()[:, :, ::-1])
        pyplot.imshow(v.get_image()[:, :, ::-1])
        pyplot.show()