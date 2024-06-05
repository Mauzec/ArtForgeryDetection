from BoVW.BoVW import BoVW
from dataset.get_pictures import DatasetOperations
from CustomDescriptors.SiftDescriptor.SIFT import SIFT
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from numpy.typing import NDArray
import numpy as np


import yaml
with open('config.yaml', 'r') as config:
    cfg = yaml.safe_load(config)
    
PATH = cfg['Victor']['Dataset']



def main(percentage: int, descriptor) -> int:
    DatasetOperations.clear()
    DatasetOperations.get_mona_original(PATH=PATH)
    DatasetOperations.get_work_train_dataset(PATH=PATH, percentage_train=percentage)
    DatasetOperations.get_mona_test(PATH=PATH)
    DatasetOperations.scale_all()
    
    bovw = BoVW(descriptor=descriptor, number_words=500)
    
    bovw.add_train_dataset("dataset/train")
    
    bovw.model_training()
    
    
    result = np.array([
        bovw.classification_image("dataset/train/artist/mona_original.png"),
        bovw.classification_image("dataset/test/artist/mona_younger_1.jpg"),
        bovw.classification_image("dataset/test/artist/mona_younger_2.jpg"),
        bovw.classification_image("dataset/test/artist/mona_younger_3.jpg"),
        bovw.classification_image("dataset/test/artist/mona_younger_4.jpg"),
    ])
    print(result)
    DatasetOperations.clear()
    
def add_train_dataset(percentage: int = 100):
    DatasetOperations.clear()
    DatasetOperations.get_mona_original(PATH=PATH)
    DatasetOperations.get_work_train_dataset(PATH=PATH, percentage_train=percentage)
    DatasetOperations.get_mona_test(PATH=PATH)
    DatasetOperations.scale_all()
    
def train(
    descriptor = SIFT(entry_path=cfg['Victor']['SIFT']),
    cluster = KMeans,
    number_words = 200,
    clf = LinearSVC(max_iter=80000)
    ) -> BoVW:
    bovw = BoVW(
        descriptor=descriptor,
        cluster=cluster(n_clusters=number_words),
        clf=clf,
        number_words=number_words,
        scale=False
    )
    bovw.add_train_dataset("dataset/train")
    
    bovw.model_training()
    
    return bovw
    
def test(bovw: BoVW) -> tuple[list[str], str]:
    propotion_correctly_definded = bovw.testing("dataset/test")
    result = [
            bovw.classification_image(image)
            for image in cfg['Victor']['Test']
            ]
    
    return result, propotion_correctly_definded
    
    
if __name__ == "__main__":
    add_train_dataset(5)
    result = test(
        train()
    )
    
    print(result[0])
    print(result[1])
        