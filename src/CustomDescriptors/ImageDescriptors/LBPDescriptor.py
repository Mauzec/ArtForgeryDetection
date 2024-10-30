from numpy.typing import NDArray
import cv2
from skimage.feature import local_binary_pattern
from CustomDescriptors.abstract.abstract import ABSImageDescriptor

class LBP(ABSImageDescriptor):
    def __init__(self, P=8, R=1, method="uniform") -> None:
        self.P = P
        self.R = R
        self.method = method
        
    def compute(self, image: str, index_process: int = -1) -> tuple:
        return None, local_binary_pattern(image,
                                    self.P,
                                    self.R,
                                    self.method)
        