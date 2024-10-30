from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray 
import cv2

class ABSDescriptor(metaclass=ABCMeta):
    @abstractmethod 
    def compute(self, image_path: str, index_process:int = -1) -> tuple[NDArray, NDArray]:
        return None, None
    
class ABSImageDescriptor(metaclass=ABCMeta):
    @abstractmethod 
    def compute(self, image: str, index_process:int = -1) -> NDArray:
        return None, None