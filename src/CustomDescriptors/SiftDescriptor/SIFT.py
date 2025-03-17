from CustomDescriptors.abstract.abstract import ABSDescriptor
import cv2
import numpy as np
from numpy.typing import NDArray

class SIFT(ABSDescriptor):
    def compute(self, image_path: str, index_process:int = -1) -> tuple[NDArray, NDArray]:
        SIFT = cv2.SIFT.create()
        img = cv2.imread(image_path, 0)
        kp, des = SIFT.detectAndCompute(img, None)
        print(f"found {len(kp)} keypoints")
        return np.array(kp), np.array(des)
    
    def __repr__(self) -> str:
        return "SIFT"