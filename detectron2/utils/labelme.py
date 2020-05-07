import base64
import io
import json
from labelme.label_file import LabelFile
import math
import os
import os.path as osp
import PIL.Image
import PIL.ImageDraw

import sys
from qtpy import QT_VERSION

QT4 = QT_VERSION[0] == '4'
PY2 = sys.version[0] == '2'

class LabelMe(object):
    def __init__(self, file_name, save_json_path="./coco.json"):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
        self.file_name = file_name
        self.save_json_path = save_json_path
        self.shapes = []

    def add_shape(self, shape):
        self.shapes.append(shape)

    def shape(self, polygons, label):
        length = len(polygons[0])
        points = polygons[0].reshape(int(length / 2), 2)
        shape = {}
        shape["label"] = label
        #shape["points"] = self.shortPoints(points.tolist())
        shape["points"] = points.tolist()
        shape["group_id"] = None
        shape["shape_type"] = "polygon"
        shape["flags"] = {}

        return shape

    def shortPoints(self, points):
        newPoints = []
        newPoints.append(points[0])
        pointslen = len(points)

        keepPointIndex = 0;
        for i in range(1, pointslen):
            dx = math.fabs(float(points[i][0]) - float(points[keepPointIndex][0]))
            dy = math.fabs(float(points[i][1]) - float(points[keepPointIndex][1]))

            if dx > 2.0 or dy > 2.0:
                keepPointIndex = i
                newPoints.append(points[keepPointIndex])

        return newPoints

    def data2labelme(self):
        data_labelme = {}
        data_labelme["version"] = "4.2.10"
        data_labelme["flags"] = {}
        data_labelme["shapes"] = self.shapes
        data_labelme["imagePath"] = os.path.basename(self.file_name)
        #data_labelme["imageData"] = None
        data_labelme["imageData"] = base64.b64encode(LabelFile.load_image_file(self.file_name)).decode('utf-8')
        return data_labelme

    def save_json(self):
        self.data_labelme = self.data2labelme()

        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_labelme, open(self.save_json_path, "w"), indent=4)
        #json.dump(self.data_coco, open(self.save_json_path, "w"), separators=(',', ':'))

    def load_image_file(self,filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            print('Failed opening image file: {}'.format(filename))
            return

        # apply orientation to image according to exif
        image_pil = self.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = 'PNG'
            elif ext in ['.jpg', '.jpeg']:
                format = 'JPEG'
            else:
                format = 'PNG'
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def apply_exif_orientation(self, image):
        try:
            exif = image._getexif()
        except AttributeError:
            exif = None

        if exif is None:
            return image

        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in exif.items()
            if k in PIL.ExifTags.TAGS
        }

        orientation = exif.get('Orientation', None)

        if orientation == 1:
            # do nothing
            return image
        elif orientation == 2:
            # left-to-right mirror
            return PIL.ImageOps.mirror(image)
        elif orientation == 3:
            # rotate 180
            return image.transpose(PIL.Image.ROTATE_180)
        elif orientation == 4:
            # top-to-bottom mirror
            return PIL.ImageOps.flip(image)
        elif orientation == 5:
            # top-to-left mirror
            return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
        elif orientation == 6:
            # rotate 270
            return image.transpose(PIL.Image.ROTATE_270)
        elif orientation == 7:
            # top-to-right mirror
            return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
        elif orientation == 8:
            # rotate 90
            return image.transpose(PIL.Image.ROTATE_90)
        else:
            return image
