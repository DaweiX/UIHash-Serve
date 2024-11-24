from os.path import abspath, join, dirname, basename
from os import listdir
import sys
sys.path.append(dirname(abspath(__file__)))

import cv2
import numpy as np
import yaml

from .xml2nodes import XMLReader
from .util import init_logger
from .nodes2hash import Nodes2Hash
from .reidentify import ImgClassifier

# load configs
home_path = dirname(dirname(abspath(__file__)))
with open(join(home_path, "config.yml"), 'r') as f:
    configs = yaml.safe_load(f)

# some constant
DATA_PATH = configs["path"]["DATA"]
SIMPLE_UI = configs["uihash"]["SIMPLE_UI"]
TYPE_COUNT = configs["uihash"]["TYPE_COUNT"]
grid_size = configs["uihash"]["GRID_SIZE"].split(',')
img_size = configs["uihash"]["IMG_SIZE"].split(',')
GRID_SIZE = (int(grid_size[0]), int(grid_size[1]))
IMG_SIZE = (int(img_size[0]), int(img_size[1]))
LOG_PATH = configs["log"]["UIHASH"]
LOG_LEVEL = configs["log"]["LEVEL"]

logger = init_logger(LOG_LEVEL, LOG_PATH, "UIHash")
hasher = Nodes2Hash(GRID_SIZE, TYPE_COUNT)
classifier = ImgClassifier(class_num=TYPE_COUNT, logger=logger)
classifier.load_model()
logger.debug(f"Device: {classifier.device}")

def get_view_imgs_from_xml(xml_file: str, view_list: list) -> np.array:
    """ use opencv to split a ui screenshot to extract its view images

    Args:
        xml_file (str): a hierarchy file
        view_list (list): view nodes
    """
    logger.debug("Extract view images from the screenshot")
    jpg_path = f"{xml_file[:-4]}.jpg"
    img_full = cv2.imread(jpg_path, 1)
    logger.debug(f"Image size: {img_full.shape}")
    if img_full is None:
        return None
    
    images = []

    for n in view_list:
        lt, rb = n["lt"], n["rb"]
        (h1, v1), (h2, v2) = lt, rb
        h1, v1, h2, v2 = int(h1), int(v1), int(h2), int(v2)
        
        img = img_full[v1:v2, h1:h2, :]
        # logger.debug(f"View size: {img.shape} ({v1}:{v2}, {h1}:{h2})")
        if img.shape[0] == 0 or img.shape[1] == 0:
            # the xml is not align with the jpg
            # (e.g., one is landscape and the other is not)
            images.append(np.zeros(shape=IMG_SIZE))
            continue
        try:
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)

        except cv2.error as e:
            logger.error(f"CV2 error: {e.msg}")
            images.append(np.zeros(shape=IMG_SIZE))
    
    images = np.array(images)

    logger.debug(f"Get views: {images.shape}")
    return images


def apk2data(apk_path: str):
    data_list = []
    xmls = [i for i in listdir(apk_path) if i.endswith(".xml")]

    for xml in xmls:
        xml_path = join(apk_path, xml)
        data = ui2data(xml_path)
        if data is not None:
            data_list.append(data)

    logger.info(f"Get UI#: {len(data_list)} UI(s) parsed")
    return data_list


def ui2data(xml_path: str):
    logger.info(f"Parsing xml: {xml_path}")
    nodes = XMLReader(xml_path, logger).node_dicts
    if SIMPLE_UI > 0:
        if len(nodes) < SIMPLE_UI:
            logger.debug(f"Too simple UI (less than {SIMPLE_UI} elements)")
            return None
    view_images = get_view_imgs_from_xml(xml_path, nodes)
    types = classifier.predict(view_images)
    ui_hash = hasher.gen_uihash(xml_path=xml_path, nodes=nodes, types=types, logger=logger)
    if ui_hash is not None:
        xml = basename(xml_path)[:-4]
        apk = basename(dirname(xml_path))
        types = [str(i) for i in types]
        return apk, xml, "|".join(types), ui_hash


if __name__ == "__main__":
    print(apk2data("/home/jiawei/uihash/collect/demo/4B21112A49D177FD8DD3DB27369A51B18B12C2D4F291E0DF9B3F376594147825"))