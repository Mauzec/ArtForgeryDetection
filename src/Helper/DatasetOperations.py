import yaml
import shutil
import os
import cv2
from sklearn.model_selection import train_test_split
    
class DataOperations:
    
    def __init__(self, user: str) -> None:
        
        self.Dirs = None
        with open('config.yaml', 'r') as config:
            cfg = yaml.safe_load(config)[user]
            self.Dirs: dict = {
                "Data": cfg["Dataset"]["Input"],
                "Train": cfg["Dataset"]["Output"]["Train"],
                "Test": cfg["Dataset"]["Output"]["Test"],
                "Inference": cfg["Dataset"]["Output"]["Inference"]}
        self.__isDir()
        
        self.encoder = {"Artist": 1, "Other-Artist": 0}
    
    def __isDir(self,
               isClear: bool = False,
               ) -> bool:

        if not os.path.isdir(self.Dirs[f"Data"]):
                ValueError(f"dataset directory is not found")
                
        for typeData in ["Train", "Test", "Inference"]:
            if not os.path.isdir(self.Dirs[typeData]):
                if not isClear: Warning(f"{typeData} directory was not created earlier")
                os.makedirs(self.Dirs[typeData])
    
    def __makeDirs(self,
                  nameDirs: list,
                  typeData: str = ["Train", "Test"]) -> None:
        for nameDir in nameDirs:
            if not os.path.isdir(f"{self.Dirs[typeData]}/{nameDir}"):
                print(f"{self.Dirs[typeData]}/{nameDir}")
                os.makedirs(f"{self.Dirs[typeData]}/{nameDir}")
    
    def getData(self, type_data: str = ["TrainTest", "Inference"]):
        functions = {"TrainTest": self.__getTrainTestData,
                     "Inference": self.__getInferenceData}
        
        return functions[type_data]
    
    def __getInferenceData(self, artists: list[str]) -> dict:
        self.__makeDirs(["Artist", "Other-Artist"], "Inference")
        data = {
            "Artist": [],
            "Other-Artist": [],
        }
        for artist in artists:
            for type_bin in ["Artist", "Other-Artist"]:
                data_artist_picture = os.listdir(f"{self.Dirs['Data']}/Inference/{type_bin}/{artist}")  
                data[type_bin] += data_artist_picture

                for picture in data_artist_picture:
                    shutil.copy(
                        f"{self.Dirs['Data']}/Inference/{type_bin}/{artist}/{picture}",
                        f"{self.Dirs['Inference']}/{type_bin}"
                    )
                    
        return data
    
    def __getTrainTestData(self, artist_train: list[str],
                artist_test: list[str],
                train_size: float,
                random_state: int = 42
                ) -> tuple[list, list]:
        
        self.__makeDirs(["Artist", "Other-Artist"], "Train")
        self.__makeDirs(["Artist", "Other-Artist"], "Test")
        
        data = {
            "Train": {
                "Artist": [],
                "Other-Artist": [],
            },
            "Test": {
                "Artist": [],
                "Other-Artist": []
            }
        }
        
        # Train
        
        for artist in artist_train:
            data_artist_picture = os.listdir(f"{self.Dirs['Data']}/Other-Artist/{artist}")
            if train_size == 1.0:
                data_artist_picture_train = data_artist_picture
                data_artist_picture_test = []
            else:
                data_artist_picture_train, data_artist_picture_test = train_test_split(data_artist_picture,
                                                                                  train_size=train_size,
                                                                                  random_state=random_state)
            
            data["Train"]["Other-Artist"] += data_artist_picture_train
            data["Test"]["Other-Artist"] += data_artist_picture_test
            
            for picture in data_artist_picture_train:
                shutil.copy(
                    f"{self.Dirs['Data']}/Other-Artist/{artist}/{picture}",
                    f"{self.Dirs['Train']}/Other-Artist"
                )
                
            for picture in data_artist_picture_test:
                shutil.copy(
                    f"{self.Dirs['Data']}/Other-Artist/{artist}/{picture}",
                    f"{self.Dirs['Test']}/Other-Artist"
                )
            
         
         # Test   
        for artist in artist_test:
            data_artist_picture = os.listdir(f"{self.Dirs['Data']}/Artist/{artist}")
            
            data["Train"]["Artist"] += data_artist_picture
            data["Test"]["Artist"] += data_artist_picture
            
            for picture in data_artist_picture:
                shutil.copy(
                    f"{self.Dirs['Data']}/Artist/{artist}/{picture}",
                    f"{self.Dirs['Train']}/Artist"
                )
                
                shutil.copy(
                    f"{self.Dirs['Data']}/Artist/{artist}/{picture}",
                    f"{self.Dirs['Test']}/Artist"
                )
                
        return data
            
        
                    
    def clearDirs(self,
                   isTrain: bool = True,
                   isTest: bool = True,
                   isInference: bool = True
                   ) -> None:
        if isTrain: shutil.rmtree(self.Dirs["Train"])
        if isTest: shutil.rmtree(self.Dirs["Test"])
        if isInference: shutil.rmtree(self.Dirs["Inference"])
        self.__isDir(isClear=True)
        
    def scaleImage(self,
                   imagePath: str,
                   scaleFunctions: list = [cv2.GaussianBlur, cv2.GaussianBlur],
                   args: list = [[(9,9)], [(9,9)]],
                   kwargs: list = [{"sigmaX": 50, "sigmaY": 50}, {"sigmaX": 50, "sigmaY": 50}],
                   crop: int = 10
                   ) -> None:
    
        image = cv2.imread(imagePath, 0)
        
        for idx, scaleFunction in enumerate(scaleFunctions):
            image = scaleFunction(image, *args[idx], **kwargs[idx])
        
        # Обрезка краев
        if crop > 0:
            image = image[crop:-crop, crop:-crop]
        
        isWritten = cv2.imwrite(imagePath, image)
        
    def scaleImages(self,
                   imagePaths: list[str],
                   scaleFunctions: list = [cv2.GaussianBlur],
                   args: list = [[(9,9)]],
                   kwargs: list = [{"sigmaX": 50, "sigmaY": 50}],
                   crop: int = 10
                   ) -> None:
        
        for imagePath in imagePaths:
            self.scaleImage(imagePath, scaleFunctions, args, kwargs, crop)
          
        
        
    def encode(self, data: dict, type_data: str = ["Inference", "Train", "Test"]) -> tuple[list, list]:
        
        X = []
        y = []
        for type_bin_artist in ['Artist', 'Other-Artist']:
            y += [self.encoder[type_bin_artist]] * len(data[type_bin_artist])
            X += [f"{self.Dirs[type_data]}/{type_bin_artist}/{picture}" for picture in data[type_bin_artist]]
        
        return X, y

                    
    