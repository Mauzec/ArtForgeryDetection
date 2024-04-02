import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab as pl
import sift.sift as SIFT
import ABS_classes.adstract_classes as ABC
from sklearn.metrics import confusion_matrix,accuracy_score
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

class Descriptor_CV2_SIFT(ABC.Descriptor):
    def __init__(self) -> None:
        super().__init__()
        self._sift = cv2.SIFT_create()
        
    def compute(self, image: cv2.typing.MatLike) -> tuple:
        kp = self._sift.detect(image, None)
        keypoints, descriptors = self._sift.compute(image, kp)
        return keypoints, descriptors

class BoVW(ABC.Descriptor):
    def __init__(self) -> None:
        self._descriptor = Descriptor_CV2_SIFT()
        self._image_paths = []
        self._dataset = []
        self._image_classes = []
        self._code_book = np.ndarray(shape=0)
        self._number_words = 200
        self._stdslr = np.ndarray(shape=0)
        self._clf = LinearSVC(max_iter=80000)
        self._class_names = []
        
    def add_train_dataset(self, path: str) -> None:
        self._clear_dataset()
        
        if not os.path.isdir(path):
            raise NameError("No such directory " + path)
        
        self._class_names = os.listdir(path)
        
        for k, name in enumerate(self._class_names):
            directory = os.path.join(path, name)
            class_path = (os.path.join(directory,f) for f in os.listdir(directory))
            was_len = len(self._image_paths)
            self._image_paths += class_path
            self._image_classes += [k] * (len(self._image_paths) - was_len)
            
            
        for i in range(len(self._image_paths)):
            self._dataset.append((self._image_paths[i], self._image_classes[i]))
            
    def model_training(self) -> None:
        descriptor_list = []
        
        for image_path in self._image_paths:
            image=cv2.imread(image_path)
            _, descriptor= self._descriptor.compute(image)
            descriptor_list.append((image_path, descriptor))
        
        descriptors = descriptor_list[0][1]
        
        for image_path, descriptor in descriptor_list[1:]:
            descriptors=np.vstack((descriptors,descriptor))

        descriptors = descriptors.astype(float)
        self._code_book, _ = kmeans(descriptors, self._number_words, 1)
        
        image_features=np.zeros((len(self._image_paths), self._number_words),"float32")
        for i in range(len(self._image_paths)):
            words, _ = vq(descriptor_list[i][1], self._code_book)
            for w in words:
                image_features[i][w]+=1
        
        self._stdslr  = StandardScaler().fit(image_features)
        image_features=self._stdslr.transform(image_features)
        
        self._clf.fit(image_features,np.array(self._image_classes))
        
      
    def testing(self, path_tests: str) -> float:
        self.add_train_dataset(path_tests)
          
        descriptor_list_test = []
        for image_path in self._image_paths:
            image = cv2.imread(image_path)
            _, descriptor_test = self._descriptor.compute(image)
            descriptor_list_test.append((image_path, descriptor_test))
        test_features=np.zeros((len(self._image_paths), self._number_words),"float32")
        
        for i in range(len(self._image_paths)):
            words, _ =vq(descriptor_list_test[i][1], self._code_book)
            for w in words:
                test_features[i][w] += 1

        true_classes=[]
        for k in self._image_classes:
            true_classes.append(k)
            
            
        predict_classes=[]
        for k in self._clf.predict(test_features):
            predict_classes.append(k)
        
        accuracy=accuracy_score(true_classes, predict_classes)
        
        return accuracy
          
    def update(self, descriptor = Descriptor_CV2_SIFT(), code_book = np.ndarray(shape=0), \
        number_words = 200, clf = LinearSVC(max_iter=80000)) -> None:
        self._descriptor = descriptor
        self._code_book = code_book
        self._number_words = number_words
        self._clf = clf
        
    def _clear_dataset(self) -> None:
        self._image_paths = []
        self._dataset = []
        self._image_classes = []
        self._image_classes_name = []
        
    def classification_image(self, image_path: str) -> tuple[str, int]:
        if not os.path.isfile(image_path):
            return ("no file", -1)

        image = cv2.imread(image_path)
        _, descriptor = self._descriptor.compute(image)
        
        test_features = np.zeros((1, self._number_words), "float32")

        words, _ = vq(descriptor, self._code_book)
        for w in words:
            test_features[0][w] += 1

        predicted_class = self._clf.predict(test_features)[0]

        return (self._class_names[predicted_class], predicted_class)
        
        
        
if __name__ == "__main__":
    bovw = BoVW()
    bovw.add_train_dataset("dataset/train")
    print("start modeling")
    bovw.model_training()
    print("Result:")
    name = f"dataset/test/artist/mona_younger.jpeg"
    while True:
        print(bovw.classification_image(name))
        name = input()