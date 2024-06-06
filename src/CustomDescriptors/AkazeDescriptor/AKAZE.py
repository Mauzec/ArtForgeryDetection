from CustomDescriptors.abstract.abstract import ABSDescriptor
import cv2
import numpy as np
from numpy.typing import NDArray

class AKAZE(ABSDescriptor):
    def compute(self, image_path: str, index_process:int = -1) -> tuple[NDArray, NDArray]:
        AKAZE = cv2.AKAZE.create()
        img = cv2.imread(image_path, 0)
        kp, des = AKAZE.detectAndCompute(img, None)
        print(f"found {len(kp)} keypoints")
        return np.array(kp), np.array(des)