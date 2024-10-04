from CustomDescriptors.abstract.abstract import ABSDescriptor
import numpy as np
from numpy.typing import NDArray
import os
import json

class SIFT(ABSDescriptor):
    def __init__(self, entry_path: str = None) -> None:
        if not entry_path:
            ValueError("enter the entry path")
            
        self.entry_path = f"./{entry_path}"
        
    def compute(self, image_path: str, index_process: int = -1, drawkps: int = 0) -> tuple[NDArray, NDArray]:
        ''' Search keypoints and descriptors. 
            Usage example:
                from sift_cpp.compute import DescriptorSift

                kps, des = DescriptorSift.compute('book.png', drawkps=0)
                kps, des = DescriptorSift.compute('book.png', drawkps=1) for create result.jpg with keypoints marked
        '''
        assert drawkps == 0 or drawkps == 1
        if not os.path.isfile(image_path):
            print(image_path)
        os.system(f'{self.entry_path} {image_path} -drawkps={drawkps} {index_process}')

        data = None
        with open(f"kps{index_process}.json", 'r') as file: data = json.load(file)
        os.remove(f"kps{index_process}.json")
        kps = []
        des = []
        for item in data['kpsdes']:
            kps.append((item["x"], item["y"]))
            des.append(item['des'])
        return np.array(kps), np.array(des)
    
    def __repr__(self) -> str:
        return "SIFT"

        

