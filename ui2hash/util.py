import logging
import os
from xml.dom.minidom import Document, parseString, Element
from typing import Tuple
import re
import math

from colorlog import ColoredFormatter


def init_logger(level: int, log_path: str, logger_name: str):
    """
    Initialize the logger.
    :param level: The level of the logger.
    :param log_path: The path to the log file.
    :param logger_name: The name of the logger.
    :return: The logger.
    """
    logger = logging.getLogger(logger_name)
    formatter = ColoredFormatter(
        "%(white)s%(asctime)10s | %(log_color)s%(name)6s | %(log_color)s%(levelname)4s | %(log_color)s%(message)6s",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'yellow',
            'WARNING':  'light_yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
    )

    file_formatter = logging.Formatter(
        "%(asctime)10s | %(name)6s | %(levelname)4s | %(message)6s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if len(log_path):
        log_file = os.path.abspath(path=log_path)
        output_file_handler = logging.FileHandler(log_file, mode='a+')
        output_file_handler.setFormatter(file_formatter)
        logger.addHandler(output_file_handler)

    logger.setLevel(level)
    return logger


def is_removal(node: Element, keywords: str = "") -> bool:
    """Help to remove android sys nodes (like the top banner)
    and views added by other sys apps (i.e., float items)

    Args:
        node (Element): A xml node
        keywords (str): Split by comma, indicating all the keywords that
          are not related to UI analysis (e.g., global float area, or
          some controls drawed by certain simulator/OS)

    Returns:
        whether a n is of no sense
    """
    package = node.getAttribute('package')
    if package == 'android':
        return True
    if package.count('com.android.systemui'):
        return True
    counts = sum([package.count(a) for a in keywords.split(',') if len(a)])
    if counts > 0:
        return True
    return False


def xml_init(node: Document):
    """Initialize a xml doc by removing its
    comment nodes or text nodes

    Args:
        node (Document): The xml doc object
    """
    if node.childNodes:
        for child in node.childNodes[:]:
            if child.nodeType == child.TEXT_NODE or \
                    child.nodeType == child.COMMENT_NODE:
                node.removeChild(child)
                continue
            if is_removal(child):
                node.removeChild(child)
                continue
            xml_init(child)


def remove_sysnode(xml: str) -> Tuple[bool, Element]:
    """Initialize a hierarchy and remove its root node

    Args:
        xml (str): Input hierarchy string

    Returns:
        The first return value is a bool, indicating whether
          the hierarchy is empty. The second return value is
          the output hierarchy dom
    """
    dom = parseString(xml)
    xml_init(dom)
    has_contents = False
    hier = dom.getElementsByTagName('hierarchy') 
    if len(hier) > 0:
        dom = hier[0]
        has_contents = dom.hasChildNodes()
    return has_contents, dom


def read_all_nodes(node_list: list, node: Element, isroot: bool = True):
    """Read xml nodes from a root node
    
    Args:
        node_list (list): An input list for xml nodes, usually an empty one.
          The results will be saved in the list.
        node (Element): The root xml node
        isroot (bool): Just make it True when calling the
          function somewhere else
    """
    if not isroot:
        node_list.append(node)
    if node.hasChildNodes():
        for child in node.childNodes:
            read_all_nodes(node_list, child, False)


def valid_xml(xml_path: str):
    """Make the xml in the valid format

    Args:
        xml_path (str): Path to the xml file

    Returns:
        A xml string or None
    """
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml = f.read()
    if not xml:
        return None
    if xml.count('<?xml') <= 1:
        return xml
    else:
        a = re.findall(r'(?<=\?>)[\s\S]+?(?=<\?)|(?<=\?>)[\s\S]+?$', xml)
        xml_tag = re.search(r'<\?.+?\?>', xml).group()
        lengths = [len(i) for i in a]
        new_xml = xml_tag + a[lengths.index(max(lengths))]
        return new_xml
    

def get_iou(boundary_view: Tuple[int, int, int, int],
            boundary_grid: Tuple[float, float, float, float]) -> float:
    """Calculate IoU (Intersection over Union)

    Args:
        boundary_view: A quadri-tuple (h_left, v_top, h_right, v_bottom),
          indicating the left-top and right-bottom corner of a view
        boundary_grid: (h_left, v_top, h_right, v_bottom) for a hash grid

    Returns:
        IoU value (float)
    """
    v_h1, v_v1, v_h2, v_v2 = boundary_view
    g_h1, g_v1, g_h2, g_v2 = boundary_grid
    garea = (g_h2 - g_h1) * (g_v2 - g_v1)

    h1, v1 = max(v_h1, g_h1), max(v_v1, g_v1)  # left-top
    h2, v2 = min(v_h2, g_h2), min(v_v2, g_v2)  # right-bottom
    w = max(0, (h2 - h1))
    h = max(0, (v2 - v1))

    common_area = w * h
    return common_area / garea


def amp_small_scaler(x: float) -> float:
    """Amplify small signals in UI#

    Args:
        x (float): Input value

    Returns:
        An amplified value (float)
    """
    def ex(_x):
        return math.log(_x + 0.01, 2)
    return (ex(x) - ex(0)) / (ex(1) - ex(0))
