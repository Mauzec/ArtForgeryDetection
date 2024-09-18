import numpy as np
import json
import cv2
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from joblib import dump, load
from CustomDescriptors.abstract.abstract import ABSDescriptor
from Helper.Multiprocessor import Multiprocessor
from typing import FunctionType


class BoVW(ClassifierMixin, BaseEstimator):
    def __init__(self,
                 descriptor: ABSDescriptor,
                 number_words: int,
                 clf: ClassifierMixin,
                 cluster: TransformerMixin,
                 stdslr: TransformerMixin = StandardScaler(),
                 ) -> None:
        
        self._descriptor = descriptor
        self._dataset = []
        self._number_words = number_words
        self._stdslr = stdslr
        self._clf = clf
        self._cluster = cluster
        self.labels_ = None
        
        self._mp = Multiprocessor(NUM_PROCESS=2)
        
    def fit(self, X: list, y: list) -> None:
        X = self._get_list(X, self._get_gray_image_path)
        self._image_classes = y
        descriptor_list = self._get_list(X, self._get_descriptor)
        
        k = 0
        while k < len(descriptor_list):
            if descriptor_list[k].shape[0] == 0:
                descriptor_list.pop(k)
                k -= 1
                
            k += 1
        
        descriptors = descriptor_list[0]
        for descriptor in descriptor_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))  
        descriptors = descriptors.astype(np.float64)
        
        self._cluster.fit(descriptors)
        
        image_features = self._get_list(descriptor_list, self._get_image_feature)
        
        self._stdslr.fit(image_features)
        image_features=self._stdslr.transform(image_features)
        
        self._clf.fit(image_features, y)
        self.labels_ = self._clf.predict(image_features)
        
      
    def predict(self, X: list) -> NDArray:
          
        descriptor_list_test = self._get_list(X, self._get_descriptor)
            
        test_features = self._get_list(descriptor_list_test, self._get_image_feature)
        test_features = self._stdslr.transform(test_features)
            
        return self._clf.predict(test_features)
    
    def score(self,
              X: list,
              y: list,
              score = accuracy_score
              ) -> None:
        
        return score(y, self.predict(X))
        
    
    def _get_list(self, image_paths: list, func) -> list:
        return self._mp.run(image_paths, func)
    
    def _get_descriptor(self, image_paths: NDArray, index_process: int) -> tuple[str, np.ndarray]:
        _, descriptor = self._descriptor.compute(image_paths, index_process=index_process)
        return np.array(descriptor, dtype=np.float64)     
    
    def _get_image_feature(self, descriptor: np.ndarray, index_process: int) -> tuple[np.ndarray, int]:
        image_feature = np.zeros(self._number_words, dtype=np.float64)
        words = self._cluster.predict(descriptor)
        for w in words:
            image_feature[w] += 1 

        return image_feature
    
    def _get_gray_image_path(self, image_path: str, index_process: int) -> str:
        image = cv2.imread(image_path, 0)
        cv2.imwrite(image_path, image)
        return image_path
    
    def get_params(self) -> str:
        return {
            "descriptor": f"{self._descriptor}",
            "number words": f"{self._number_words}",
            "clf": f"{self._clf}",
            "cluster": f"{self._cluster}",
            "stdslr": f"{self._stdslr}",
            }
    
    def save_model(self,
                   directory: str,
                   name_model = 'modelSVM.jolib',
                   name_classes = "name_classes.json",
                   name_scaler = 'std_scaler.joblib',
                   name_cluster  ="cluster.jolib") -> None:
         
        dump(self._clf, f"{directory}/{name_model}", compress=True)
        dump(self._stdslr, f"{directory}/{name_scaler}", compress=True)
        dump(self._cluster, f"{directory}/{name_cluster}", compress=True)
        with open(f"{directory}/{name_classes}", "w") as json_file:
            data = {"names": self._class_names}
            json.dump(data, json_file, ensure_ascii=False)
            
        print("model is saved")
        
    def download_model(self,
                   directory: str,
                   name_model = 'modelSVM.jolib',
                   name_classes = "name_classes.json",
                   name_scaler = 'std_scaler.joblib',
                   name_cluster  ="cluster.jolib") -> None:
        
        self._clf = load(f"{directory}/{name_model}")
        self._stdslr = load(f"{directory}/{name_scaler}")
        self._cluster = load(f"{directory}/{name_cluster}")
        with open(f"{directory}/{name_classes}", 'r') as json_file: 
            self._class_names = json.load(json_file)["names"]
            
        print("model is downloaded")