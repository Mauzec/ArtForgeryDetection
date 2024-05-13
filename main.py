from BoVW.BoVW import BoVW
from BoVW.dataset.get_pictures import Dataset_operations
from ResnetDescriptor.Resnet import ResnetDescruptor as Resnet
from BoVW.sift_cpp.compute import DescriptorSift
import matplotlib.pyplot as plt
import numpy as np

def main(percentage: int, descriptor) -> int:
    Dataset_operations.clear()
    Dataset_operations.get_mona_original()
    Dataset_operations.get_work_train_dataset(percentage_train=percentage)
    bovw = BoVW(scale=True, descriptor=descriptor, number_words=500)
    
    bovw.add_train_dataset("BoVW/dataset/train")
    
    bovw.model_training()
    bovw.save_model()
    
    Dataset_operations.get_mona_original()
    Dataset_operations.get_mona_younger()
    
    result = np.array([
        bovw.classification_image("BoVW/dataset/train/artist/mona_original.png"),
        bovw.classification_image("BoVW/dataset/test/artist/mona_younger_1.jpg"),
        bovw.classification_image("BoVW/dataset/test/artist/mona_younger_2.jpg")
    ])
    print(result)
    # right_answer = np.array([0, 0, 1])
    # return np.count_nonzero(result == right_answer)
    
if __name__ == "__main__":
    # Данные для столбцов
    labels = ["Resnet", "SIFT"]
    values = [2.6, 2.4]

    # Цвета для столбцов
    colors = ['blue', 'orange']

    # Создание столбцов
    plt.bar(labels, values, color=colors)

    # Показать числа на столбцах
    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha='center', va='bottom')

    # Показать график
    plt.savefig("result.png")
        