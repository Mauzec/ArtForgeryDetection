import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import json
from sift_cpp.compute import DescriptorSift
from random import choice
from sklearn.metrics import accuracy_score
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from joblib import dump, load

NUM_PROCESS = 4

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
            
    def model_training(self) -> None:
        descriptor_list = self._get_descriptor_list()
        
        descriptors = descriptor_list[0][1]
        
        for _, descriptor in descriptor_list[1:]:
            descriptors=np.vstack((descriptors,descriptor))

        descriptors = descriptors.astype(float)
        self._code_book, _ = kmeans(descriptors, self._number_words, 1)
        
        image_features = self._get_image_features(descriptor_list)
        
        self._stdslr  = StandardScaler().partial_fit(image_features)
        image_features=self._stdslr.transform(image_features)
        
        self._clf.fit(image_features, np.array(self._image_classes))
        
      
    def testing(self, path_tests: str) -> float:
        self.clear_dataset()
        self.add_train_dataset(path_tests)
          
        descriptor_list_test = self._get_descriptor_list()
            
        test_features = self._get_image_features(descriptor_list_test)

        true_classes=[]
        for k in self._image_classes:
            true_classes.append(k)
            
        predict_classes=[]
        for k in self._clf.predict(test_features):
            predict_classes.append(k)
        
        accuracy=accuracy_score(true_classes, predict_classes)
        
        return accuracy
          
    def update(self, descriptor = DescriptorSift, code_book = np.ndarray(shape=0),
               number_words = 200, clf = LinearSVC(max_iter=80000)) -> None:
        self._descriptor = descriptor
        self._code_book = code_book
        self._number_words = number_words
        self._clf = clf
        
    def clear_dataset(self) -> None:
        self._image_paths = []
        self._dataset = []
        self._image_classes = []
        self._image_classes_name = []
        
    def classification_image(self, image_path: str) -> tuple[str, int]:
        if not os.path.isfile(image_path):
            return ("no file", -1)

        image = self._image(image_path)
        _, descriptor = self._descriptor.compute(image)
        
        test_features = np.zeros((1, self._number_words), "float32")

        words, _ = vq(descriptor, self._code_book)
        for w in words:
            test_features[0][w] += 1

        predicted_class = self._clf.predict(test_features)[0]

        return (self._class_names[predicted_class], predicted_class)
    
    def _get_descriptor_list(self) -> list:
        descriptor_list = []
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        processes = [
            mp.Process(target=self._daemon_function, 
                       args=(input_queue, output_queue, self._get_descriptor, i + 1), daemon=True)
            for i in range(NUM_PROCESS)
            ]
        
        for process in processes:
            process.start()
        
        for image_path in self._image_paths:
            input_queue.put(image_path)
        k = 0    
        while k < len(self._image_paths):
            if not output_queue.empty():
                descriptor_list.append(output_queue.get())
                k += 1
            
        for process in processes:
            process.terminate()
            
        return descriptor_list
    
    def _get_descriptor(self, image_path: str, index_process: int) -> tuple[str, np.ndarray]:
        image = self._image(image_path)
        _, descriptor= self._descriptor.compute(image, index_process=index_process)
        return (image_path, descriptor)
    
    
    def _get_image_features(self, descriptor_list: list) -> np.ndarray:
        image_features=np.zeros((len(self._image_paths), self._number_words),"float32")
        
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        processes = [
            mp.Process(target=self._daemon_function, 
                       args=(input_queue, output_queue, self._get_image_feature, i + 1), daemon=True)
            for i in range(NUM_PROCESS)
            ]
        
        for process in processes:
            process.start()
        for i in range(len(self._image_paths)):
            input_queue.put({ "descriptor": descriptor_list[i][1], "number": i})
            
        k = 0
        while k < len(self._image_paths):
            if not output_queue.empty():
                image_feature, index = output_queue.get()
                image_features[index] = image_feature
                k += 1
                
        return image_features
    
    def _get_image_feature(self, data: dict, index_process: int) -> tuple[np.ndarray, int]:
        descriptor = data["descriptor"]
        image_feature = np.zeros(self._number_words,"float32")
        words, _ = vq(descriptor, self._code_book)
        for w in words:
            image_feature[w] += 1 

        return image_feature, data["number"]
    
    @property
    def classes(self) -> dict:
        classes = dict()
        for k, name in enumerate(self._class_names):
            classes[name] = k
            
        return classes
    
    @property
    def dataset(self) -> dict:
        size_classes = dict.fromkeys(self._class_names, 0)
        for k in self._image_classes:
            size_classes[self._class_names[k]] += 1
        
        dataset = {
            "size": len(self._image_paths),
            "size classes": size_classes,
            "words": self._number_words, 
        }
        
        return dataset
    
    @property
    def example(self) -> None:
        image_path = choice(self._image_paths)
        image = self._image(image_path)
        keypoints, _ = self._descriptor.compute(image)
        for keypoint in keypoints:
            x, y = keypoint
            plt.imshow(cv2.circle(image, (int(x), int(y)), 5, (255, 255, 255)))
            
        plt.savefig("example")
        
    def _image(self, image_path: cv2.typing.MatLike) -> cv2.typing.MatLike:
        print(image_path)
        # image = cv2.imread(image_path, 0)
        # image = cv2.GaussianBlur(image, (5,5), sigmaX=36, sigmaY=36)
        # height, width = image.shape
        # new_width = min(200, width // 2)
        # new_height = int(new_width * (height / width))
        # image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image_path
    
    def save_model(self, name_model = 'modelSVM.joblib', name_classes = "name_classes.json",
                   name_scaler = 'std_scaler.joblib', name_code_book = 'code_book_file_name.npy') -> None:
        dump(self._clf, name_model, compress=True)
        dump(self._stdslr, name_scaler, compress=True)
        np.save(name_code_book, self._code_book)
        with open(name_classes, "w") as json_file:
            data = {"names": self._class_names}
            json.dump(data, json_file, ensure_ascii=False)
        
    def download_model(self, name_model = 'modelSVM.joblib', name_classes = "name_classes.json",
                       name_scaler = 'std_scaler.joblib', name_code_book = 'code_book_file_name.npy') -> None:
        self._clf = load(name_model)
        self._stdslr = load(name_scaler)
        self._code_book = np.load(name_code_book)
        with open(name_classes, 'r') as json_file: self._class_names = json.load(json_file)
        
    def _daemon_function(self, input_queue: mp.Queue, output_queue: mp.Queue,
                         function, index_process: int) -> None:
        while True:
            if not input_queue.empty():
                input_data = input_queue.get()
                output_data = function(input_data, index_process)
                output_queue.put(output_data)
                
    
if __name__ == "__main__":
    bovw = BoVW()
    bovw.add_train_dataset("dataset/debug/train")
    print("start training")
    start = time.time()
    bovw.model_training()
    end = time.time()
    
    bovw.save_model()
    print("Result:")
    print(bovw.classification_image("dataset/test/artist/mona_younger.jpg"))