import numpy as np
import multiprocessing as mp
import os

class SVM:
    def fit(self, image_features: np.ndarray, name_classes: np.ndarray) -> None:
        first_class, second_class = self._split_class(image_features, name_classes)
        self._write_class("dataclass0", first_class)
        self._write_class("dataclass1", second_class)
        
        os.system(".\BoVW\svm_cpp\svm_entry.exe -train")
        
        self._remove("dataclass0")
        self._remove("dataclass1")
        
        
        
    def predict(self, image_features: np.ndarray) -> list:
        classification = []
        for image_feature in image_features:
            self._write_vector("feature", image_feature)
            os.system(".\BoVW\svm_cpp\svm_entry.exe -predict")
            classification.append(self._read_result("predict"))
            
        self._remove("feature")
        self._remove("predict")
        
        return classification
    
    def _split_class(self, image_features: np.ndarray, name_classes: np.ndarray) -> tuple[list, list]:
        first_class = list()
        second_class = list()
        for k, image_feature in enumerate(image_features):
            if name_classes[k] == 0:
                first_class.append(image_feature)
            elif name_classes[k] == 1:
                second_class.append(image_feature)
            else:
                raise NameError("invallid class")
            
        return first_class, second_class
    
    def _write_class(self, filename: str, recorded_class: list) -> None:
        with open(f"{filename}.log", "w") as writer:
            writer.write(f"{len(recorded_class)} {len(recorded_class[0])}\n")
            for vector in recorded_class:
                write_string = f"{vector[0]}"
                for v in vector[1:]:
                    write_string = write_string + f" {v}"
                write_string = write_string + "\n"
                writer.write(write_string)
                
    def _write_vector(self, filename: str, recorded_vector: np.ndarray) -> None:
        with open(f"{filename}.log", "w") as writer:
            writer.write(f"{len(recorded_vector)}\n")
            write_string = f"{recorded_vector[0]}"
            for v in recorded_vector:
                write_string = write_string + f" {v}"
            write_string = write_string + "\n"
            writer.write(write_string)  
            
    def _read_result(self, filename: str) -> int:
        with open(f"{filename}.log") as reader:
            result = int(reader.read())
            if result == -1: result = 0
            return result
        
    def _remove(self, filename: str) -> None:
        os.remove(f"{filename}.log")
                
if __name__ == "__main__":
    test = SVM()
    features = np.array([
        [1, 0, 1, 0, 0.5],
        [1, 0, 1, 0, 0.5],
        [1, 0, 1, 0, 0.5],
        [1, 0, 1, 0, 0.5],
        [1, 0, 1, 0, 0.5]
    ])
    classes = np.array([1, 0, 1, 0, 1])
    test.fit(features, classes)
    print(test.predict(features))
                
            