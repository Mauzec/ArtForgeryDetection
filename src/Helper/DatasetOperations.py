import yaml
import shutil
import os
import cv2
from sklearn.model_selection import train_test_split
from typing import FunctionType
    
class DataOperations:
    
    def __init__(self, user: str) -> None:
        
        self.Dirs = None
        with open('config.yaml', 'r') as config:
            cfg = yaml.safe_load(config)[user]
            self.Dirs: dict = {
                "Data": cfg["Dataset"]["Input"],
                "Train": cfg["Dataset"]["Output"]["Train"],
                "Test": cfg["Dataset"]["Output"]["Test"]}
        self._isDir()
    
    def _isDir(self,
               isClear: bool = False,
               ) -> bool:
        if not os.path.isdir(self.Dirs["Data"]):
            ValueError("dataset directory is not found")
        for typeData in ["Train", "Test"]:
            if not os.path.isdir(self.Dirs[typeData]):
                if not isClear: Warning(f"{typeData} directory was not created earlier")
                os.makedirs(self.Dirs[typeData])
    
    def _makeDirs(self,
                  nameDirs: list,
                  typeData: str = ["Train", "Test"]) -> None:
        for nameDir in nameDirs:
            if not os.path.isdir(self.Dirs[typeData]):
                os.makedirs(f"{self.Dirs[typeData]}/{nameDir}")
    
    def getData(self, contextKeys: str = ["TrainTest", "Split"]) -> FunctionType:
        self.clearDirs()
        def getTrainTestData(artists: dict[list] = {"train": [], "test": []},
                            all: bool = False,
                            ) -> None:
            if all: 
                artists["train"] = os.listdir(f"{self.Dirs["Data"]}/train")
                artists["test"] = os.listdir(f"{self.Dirs["Data"]}/test")
        
            for typeData in self.Dirs["Data"]: 
                dirs = os.listdir(f"{self.Dirs["Data"]}/{typeData}")
                for artist in f"{self.Dirs["Data"]}/{typeData}/{artists}":
                    if artist in dirs:
                        for imageName in f"{dirs}/{artist}":
                            shutil.copy(f"{dirs}/{artist}/{imageName}", self.Dirs[typeData])      
                    else:
                        Warning(f"{artist} is not found in {typeData}")
                        
        def getAndSplitData(test_size: float,
                            random_state: int = 42,
                            ) -> None:
            ImagePaths = dict()
            artistNames = os.listdir(self.Dirs["Data"])
            for artistName in artistNames:
                ImagePaths[artistName] = os.listdir(f"{artistNames}/{artistName}")
                
            TrainTestData = dict()
            for artistName in artistNames:
                train, test = train_test_split(ImagePaths[artistName],
                                                test_size=test_size,
                                                random_state=random_state
                                            )
                TrainTestData[artistName] = {"Train": train, "Test": test}
                
            self._makeDirs(artistNames, "Train")
            self._makeDirs(artistNames, "Test")
            
            for typeData in ["Test", "Train"]:    
                for artistName in artistNames:
                    for imageName in TrainTestData[artistName][typeData]:
                        shutil.copy(f"{self.Dirs["Data"]}/{artistName}/{imageName}",
                                    f"{self.Dirs[typeData]}/{artistName}"
                                    )
        
        contextFunction = {
            "TrainTest": getTrainTestData,
            "Split": getAndSplitData
        }
        return contextFunction[contextKeys]
                    
    def clearDirs(self,
                   isTrain: bool = True,
                   isTest: bool = True
                   ) -> None:
        if isTrain: shutil.rmtree(self.Dirs["Train"])
        if isTest: shutil.rmtree(self.Dirs["Test"])
        self._isDir(isClear=True)
        
    def scaleImage(self,
                   imagePath: str,
                   scaleFunctions: list = [cv2.GaussianBlur],
                   args: list = [(5,5)],
                   kwargs: list = [{"sigmaX": 36, "sigmaY": 36}],
                   ) -> None:
        
        image =  cv2.imread(imagePath, 0)
        
        for idx, scaleFunction in enumerate(scaleFunctions):
            image = scaleFunction(image, **args[idx], **kwargs[idx])
            
        isWritten = cv2.imwrite(imagePath, image)            

                    
    