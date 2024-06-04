from BoVW.BoVW import BoVW
from dataset.get_pictures import Dataset_operations
from CustomDescriptors.SiftDescriptor.SIFT import SIFT
import numpy as np


import yaml
with open('config.yaml', 'r') as config:
    cfg = yaml.safe_load(config)
    
PATH = cfg['Victor']['Dataset']

def main(percentage: int, descriptor) -> int:
    Dataset_operations.clear()
    Dataset_operations.get_mona_original(PATH=PATH)
    Dataset_operations.get_work_train_dataset(PATH=PATH, percentage_train=percentage)
    bovw = BoVW(scale=True, descriptor=descriptor, number_words=500)
    
    bovw.add_train_dataset("dataset/train")
    
    bovw.model_training()
    bovw.save_model('test')
    
    Dataset_operations.get_mona_original(PATH=PATH)
    Dataset_operations.get_mona_younger(PATH=PATH)
    
    result = np.array([
        bovw.classification_image("dataset/train/artist/mona_original.png"),
        bovw.classification_image("dataset/test/artist/mona_younger_1.jpg"),
        bovw.classification_image("dataset/test/artist/mona_younger_2.jpg")
    ])
    print(result)
    
if __name__ == "__main__":
    main(percentage=5, descriptor=SIFT(cfg['Victor']['SIFT']))
        