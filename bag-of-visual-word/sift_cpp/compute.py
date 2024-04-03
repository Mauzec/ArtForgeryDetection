import os
import json
import numpy

class DescriptorSift:
    @staticmethod
    def compute(image_path: str, drawkps: int = 0):
        ''' Search keypoints and descriptors. WARNING: this method must be run from bag-of-visual-word dir. 
            Usage example:
                from sift_cpp.compute import DescriptorSift

                kps, des = DescriptorSift.compute('book.png', drawkps=0)
                #kps, des = DescriptorSift.compute('book.png', drawkps=1) for create result.jpg with keypoints marked
        '''
        assert drawkps == 0 or drawkps == 1
        os.system(f'./sift_cpp/main {image_path} -drawkps={drawkps}')

        data = None
        with open('kps.json', 'r') as file: data = json.load(file)
        kps = []
        des = []
        for item in data['kpsdes']:
            kps.append((item["x"], item["y"]))
            des.append(item['des'])
        return kps, des


        

