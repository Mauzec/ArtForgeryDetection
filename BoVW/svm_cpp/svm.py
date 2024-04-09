import numpy as np
import multiprocessing as mp
import os

class SVM:
    def fit(self, image_features: np.ndarray, name_classes: np.ndarray) -> None:
        first_class, second_class = self._split_class(image_features, name_classes)
        self._write_class("first_class", first_class)
        self._write_class("second_class", second_class)
        
         # написать как вызывается C++
        
    def predict(self, image_features: np.ndarray) -> list:
        for image_feature in image_features:
            os.system()
        
        # написать как вызывается C++
    
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
    
    def _write_class(self, file_name: str, recorded_class: list) -> None:
        with open(f"BoVW/svm_cpp/{file_name}.loc", "w") as writer:
            writer.write(f"{len(recorded_class)} {len(recorded_class[0])}\n")
            for vector in recorded_class:
                write_string = f"{vector[0]}"
                for v in vector[1:]:
                    write_string = write_string + f" {v}"
                write_string = write_string + "\n"
                writer.write(write_string)
                
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
                
            