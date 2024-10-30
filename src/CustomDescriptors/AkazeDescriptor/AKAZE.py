from CustomDescriptors.abstract.abstract import ABSDescriptor
import cv2
import numpy as np
from numpy.typing import NDArray

class AKAZE(ABSDescriptor):
    def __init__(self) -> None:
        self.AKAZE = cv2.AKAZE.create()
    def compute(self, image_path: str, index_process:int = -1) -> tuple[NDArray, NDArray]:
        img = cv2.imread(image_path, 0)
        kp, des = self.AKAZE.detectAndCompute(img, None)
        print(f"found {len(kp)} keypoints")
        return np.array(kp), np.array(des)
    
    def __repr__(self) -> str:
        return "AKAZE"