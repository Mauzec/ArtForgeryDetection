from BoVW.BoVW import BoVW
from BoVW.dataset.get_pictures import Dataset_operations
from BoVW.svm_cpp.svm import SVM
from BoVW.sift_cpp.compute import DescriptorSift

def main(percentage: int):
    Dataset_operations.clear()
    Dataset_operations.get_mona_original()
    Dataset_operations.get_mona_younger()
    Dataset_operations.get_work_dataset(percentage_train=percentage)
    
    bovw = BoVW(descriptor=DescriptorSift, clf=SVM(), number_words=1000)
    
    print("start add dataset")
    bovw.add_train_dataset("BoVW/dataset/train")
    print("end add dataset")
    
    print("start training model")
    bovw.model_training()
    print("end training model")
    
    print("save model")
    bovw.save_model()
    bovw.download_model()
    # print("download model")
    # bovw.download_model()
    
    # print("start testing")
    # print(bovw.testing("BoVW/dataset/test"))
    # print("end testing")
    
    print(f"percentage={percentage}")
    Dataset_operations.get_mona_original()
    # Dataset_operations.get_mona_younger()
    
    print(bovw.classification_image("BoVW/dataset/train/artist/mona_original.png"))
    # print(bovw.classification_image("BoVW/dataset/test/artist/mona_younger_1.jpg"))
    # print(bovw.classification_image("BoVW/dataset/test/artist/mona_younger_2.jpg"))
    
    # Dataset_operations.clear()
    
    print("end program")
    
if __name__ == "__main__":
    main(10)