"""Given XML nodes, generate UI#"""

from typing import Tuple
from logging import Logger

import numpy as np
from util import get_iou, amp_small_scaler
from PIL import Image, UnidentifiedImageError


class Nodes2Hash:
    """ Takes view dicts as input and output UIHash

    Args:
        channels (int): the expected channel number in UIHash
        h_v_ticks ((int, int)): the grid size for each channel

    Attributes:
        channels (int): UIHash channel number
    """
    def __init__(self, h_v_ticks: Tuple[int, int], channels: int):
        self._screen_h, self._screen_v = 0, 0
        self._h_tick = h_v_ticks[0]
        self._v_tick = h_v_ticks[1]
        self.channels = channels

    @staticmethod
    def fine_tune_grid_lt(num: float, tick: int, size: float,
                          t: float = 0.25) -> int:
        """ Adjust view's start grid by its left/top coords, like:
          * 0.1 -> 0, 0.9 -> 1
          * 1.1 -> 1, 1.9 -> 2
          * 2.1 -> 2, 2.9 -> 2

        Args:
            num (float): input value
            tick (int): `self._h_tick` or `self._v_tick`
            size (float): view width or height (from 0 to self._?_tick)
            t (float): threshold to determine when a view occupies a grid
                when a part of it is in the grid
        """
        diff = num - int(num)
        if diff < 0.5:
            return int(num)
        try:
            if (1 - diff) / tick / size < t:
                # if only a tiny part located in the left/top corner
                # then just go right/down
                return min(int(num) + 1, tick - 1)
        except ZeroDivisionError:
            return int(num)
        else:
            # hold in this grid
            return int(num)

    @staticmethod
    def fine_tune_grid_rb(num: float, tick: int, size: float,
                          t: float = 0.25) -> int:
        """ Adjust view's start grid by its right/bottom coords

        Args:
            num (float): input value
            tick (int): `self._h_tick` or `self._v_tick`
            size (float): view width or height (from 0 to self._?_tick)
            t (float): threshold to determine when a view occupies a grid
                when a part of it is in the grid
        """
        diff = num - int(num)
        if diff >= 0.5:
            return int(num)
        try:
            if diff / tick / size < t:
                # go left/up
                return max(int(num) - 1, 0)
        except ZeroDivisionError:
            return int(num)
        else:
            # keep in this grid
            return int(num)

    def assign_hash_grid(self, nodes: list):
        """ Put views into the grids where they should be.
        For each view (dict), keys `area` and `grids`
        will be appended. Besides, if the view is already
        out-of-screen, then this function will skip the view:
        the key `grids` will not be added, and the `area` will
        be zero. As a parallel way for UIHash generationg, this
        function is not used by default in the release.

        Args:
            nodes (list): input node dict list
        """
        for n in nodes:
            n['area'] = 0
            h, v = self._screen_h, self._screen_v

            # if the left top corner out of the
            # right/bottom bounds of screen, continue
            h_1 = int(n['lt'][0]) / (h / self._h_tick)
            v_1 = int(n['lt'][1]) / (v / self._v_tick)
            if h_1 >= self._h_tick or v_1 >= self._v_tick:
                continue

            h_2 = int(n['rb'][0]) / (h / self._h_tick)
            v_2 = int(n['rb'][1]) / (v / self._v_tick)
            if h_2 < 0 or v_2 < 0:
                continue

            width = int(n['rb'][0]) - int(n['lt'][0])
            height = int(n['rb'][1]) - int(n['lt'][1])
            width = float(width) / h
            height = float(height) / v

            h_1 = self.fine_tune_grid_lt(h_1, self._h_tick, width)
            v_1 = self.fine_tune_grid_lt(v_1, self._v_tick, height)
            h_2 = self.fine_tune_grid_rb(h_2, self._h_tick, width)
            v_2 = self.fine_tune_grid_rb(v_2, self._v_tick, height)

            n['grids'] = (h_1, v_1, h_2, v_2)
            n['area'] = width * height

    def gen_uihash(self, xml_path: str, nodes: list, 
                   types: list, logger: Logger) -> np.array:
        """ Given a hierarchy xml, generate uihash. If the view is
        already out-of-screen, then the view will be skipped.
        The value in a grid will be originally given by the IoU
        between the view and the grid, then the value will be
        amplified by a log function. We determine which channel to
        put the view firstly by its reidentified view type. If the
        reidentification model is not very sure of its decision,
        then we compromise to the claimed view type.

        Args:
            nodes (list): view nodes in UI
            xml_path (str): the input hierarchy file
            types (list): reidentified control types
        """
        try:
            img = Image.open(f"{xml_path[:-4]}.jpg")
        except UnidentifiedImageError:
            logger.error(f"Unable to load image {xml_path[:-4]}.jpg")
            return None
        except FileNotFoundError:
            logger.error(f"image {xml_path[:-4]}.jpg not exists")
            return None
        self._screen_h = img.size[0]
        self._screen_v = img.size[1]
        for n in nodes:
            h, v = self._screen_h, self._screen_v
            # if the left top corner out of the
            # right/bottom bounds of screen, continue
            h_min, v_min = int(n["lt"][0]), int(n["lt"][1])
            if h_min >= h or v_min >= v:
                continue
            h_max, v_max = int(n["rb"][0]), int(n["rb"][1])
            if h_max < 0 or v_max < 0:
                continue
            n["area4grids"] = [0] * (self._v_tick * self._h_tick)

            for i in range(self._v_tick * self._h_tick):
                # calculate IoU (Intersection over Union) for each grid
                h_start_idx, v_start_idx = i % self._h_tick, \
                                           int(i / self._h_tick)
                size_unit_h = float(h) / self._h_tick
                size_unit_v = float(v) / self._v_tick
                h1 = h_start_idx * size_unit_h
                v1 = v_start_idx * size_unit_v
                h2, v2 = h1 + size_unit_h, v1 + size_unit_v
                iou = get_iou((h_min, v_min, h_max, v_max),
                              (h1, v1, h2, v2))
                width = int(n["rb"][0]) - int(n["lt"][0])
                height = int(n["rb"][1]) - int(n["lt"][1])
                width = width / (float(h) / self._h_tick)
                height = height / (float(v) / self._v_tick)
                area = width * height  # n area (compared to one grid)
                if area == 0:
                    continue
                # when the iou is only a tiny part of the view
                # we dont consider it
                t = 0.07
                if iou / area > t:
                    n['area4grids'][i] = iou

        mat = np.zeros((self.channels, self._h_tick * self._v_tick))

        for i, n in enumerate(nodes):
            if "area4grids" not in n:
                continue
            # get type of the node
            ori_type = n["name"]
            _c = types[i]
            if _c < 0:
                # 0 bar
                # 1 tab
                # 2 list
                # 3 spinner
                # 4 text
                # 5 button
                # 6 toggle
                # 7 edit
                # 8 check
                # 9 others
                # use the declared class name instead
                if "Radio" in ori_type:
                    _c = 8
                elif "Toggle" in ori_type:
                    _c = 6
                elif "Button" in ori_type:
                    _c = 5
                elif "Check" in ori_type:
                    _c = 8
                elif "ListView" in ori_type:
                    _c = 2
                elif "TextView" in ori_type:
                    _c = 4
                elif "EditT" in ori_type:
                    _c = 7
                elif "Switch" in ori_type:
                    _c = 6
                elif "CompoundButton" in ori_type:
                    _c = 5
                elif "Tab" in ori_type:
                    _c = 1
                else:
                    _c = 9

            for j, k in enumerate(n['area4grids']):
                mat[_c][j] += k
            # update the values
            mat[_c] = [amp_small_scaler(i) for i in mat[_c]]
        
        mat = np.array(mat, dtype=np.float32)
        assert not np.isnan(mat).any()
        return mat