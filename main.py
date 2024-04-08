from BoVW.BoVW import BoVW
from BoVW.dataset.get_pictures import Dataset_operations

def main():
    Dataset_operations.clear()
    # Dataset_operations.get_mona()
    # Dataset_operations.get_images(0, 1, "train")
    # Dataset_operations.get_images(1, 2, "test")
    # bovw = BoVW()
    # bovw.add_train_dataset("BoVW/dataset/train")
    # bovw.model_training()
    # print(bovw.testing("BoVW/dataset/test"))
    # print(bovw.classification_image("BoVW/mona/mona_original.png"))
    # print(bovw.classification_image("BoVW/mona/mona_younger_1.jpg"))
    # print(bovw.classification_image("BoVW/mona/mona_younger_2.jpg"))
    # bovw.save_model()
    
if __name__ == "__main__":
    main()