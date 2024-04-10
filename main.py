from BoVW.BoVW import BoVW
from BoVW.dataset.get_pictures import Dataset_operations
from BoVW.svm_cpp.svm import SVM
from BoVW.sift_cpp.compute import DescriptorSift

def main():
    Dataset_operations.clear()
    Dataset_operations.get_mona()
    Dataset_operations.get_images(0, 3, "train")
    Dataset_operations.get_images(3, 6, "test")
    
    bovw = BoVW(descriptor=DescriptorSift, clf=SVM())
    bovw.add_train_dataset("BoVW/dataset/train")
    
    bovw.model_training()
    print(bovw.testing("BoVW/dataset/test"))
    
    Dataset_operations.get_mona()
    
    print(bovw.classification_image("BoVW/dataset/train/artist/mona_original.png"))
    print(bovw.classification_image("BoVW/dataset/test/artist/mona_younger_1.jpg"))
    print(bovw.classification_image("BoVW/dataset/test/artist/mona_younger_2.jpg"))
    
    bovw.save_model()
    
    Dataset_operations.clear()
    
if __name__ == "__main__":
    main()