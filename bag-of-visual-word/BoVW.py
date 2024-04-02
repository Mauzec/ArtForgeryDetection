import cv2
import numpy as np
import os
from sift.sift import *
import matplotlib.pyplot as plt
from random import choice
from sklearn.metrics import accuracy_score
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class BoVW():
    def __init__(self) -> None:
        self._descriptor = DescriptorSift
        self._image_paths = []
        self._dataset = []
        self._image_classes = []
        self._code_book = np.ndarray(shape=0)
        self._number_words = 200
        self._stdslr = np.ndarray(shape=0)
        self._clf = LinearSVC(max_iter=80000)
        self._class_names = []
        
    def add_train_dataset(self, path: str) -> None:
        
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
            print(self._image_paths[i])
            
    def model_training(self) -> None:
        descriptor_list = []
        
        for image_path in self._image_paths:
            image=cv2.imread(image_path, 0)
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
        
        self._stdslr  = StandardScaler().partial_fit(image_features)
        image_features=self._stdslr.transform(image_features)
        
        self._clf.fit(image_features,np.array(self._image_classes))
        
      
    def testing(self, path_tests: str) -> float:
        self._clear_dataset()
        self.add_train_dataset(path_tests)
          
        descriptor_list_test = []
        for image_path in self._image_paths:
            image = cv2.imread(image_path, 0)
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
          
    def update(self, descriptor = DescriptorSift, code_book = np.ndarray(shape=0), \
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

        image = cv2.imread(image_path, 0)
        _, descriptor = self._descriptor.compute(image)
        
        test_features = np.zeros((1, self._number_words), "float32")

        words, _ = vq(descriptor, self._code_book)
        for w in words:
            test_features[0][w] += 1

        predicted_class = self._clf.predict(test_features)[0]

        return (self._class_names[predicted_class], predicted_class)
    
    @property
    def classes(self) -> dict:
        classes = dict()
        for k, name in enumerate(self._class_names):
            classes[name] = k
            
        return classes
    
    @property
    def example(self) -> None:
        image_path = choice(self._image_paths)
        image = cv2.imread(image_path, 0)
        keypoints, _ = self._descriptor.compute(image)
        for keypoint in keypoints[::]:
            x, y = keypoint.pt
            plt.imshow(cv2.circle(image, (int(x), int(y)), 5, (255, 255, 255)))
            
        plt.savefig("example")
        
    def _scale(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        image = cv2.GaussianBlur(image, (5,5), sigmaX=36, sigmaY=36)
        height, width = image.shape
        new_width = min(500, width // 2)
        new_height = int(new_width * (height / width))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image
        
        
if __name__ == "__main__":
    bovw = BoVW()
    bovw.add_train_dataset("dataset/test")
    bovw.example
    # print("start modeling")
    # bovw.model_training()
    # print(bovw.classes)
    # print("Result:")
    
    # while True:
    #     name = input()
    #     print(bovw.classification_image(name))