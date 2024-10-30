from CustomDescriptors.abstract.abstract import ABSDescriptor, ABSImageDescriptor
import cv2
import numpy as np

class Combine(ABSDescriptor):
    def __init__(self,
                 ImageDescriptors: list[ABSImageDescriptor],
                 descriptor: ABSDescriptor
                 ) -> None:
        self.ImageDescriptors = ImageDescriptors
        self.descriptor = descriptor
            
    def compute(self, image_path: str, index_process = -1, drawkps: int = 0) -> tuple:
        img = cv2.imread(image_path, 0)
        for ImageDescriptor in self.ImageDescriptors:
            _, img = ImageDescriptor.compute(img)
        cv2.imwrite(image_path, img)
        kp, des = self.descriptor.compute(image_path)
        
        return np.array(kp), np.array(des)
    
    def __repr__(self) -> str:
        names = [None] * (len(self.ImageDescriptors) + 1)
        for idx, descriptor in enumerate(self.ImageDescriptors):
            names[idx] = descriptor.__repr__()
        names[-1] = self.descriptor.__repr__()    
            
        return str(names)