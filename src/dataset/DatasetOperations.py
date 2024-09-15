import yaml
import shutil
import os


class DatasetOperations:
    def __init__(self, user: str) -> None:
        
        self.cfg = None
        with open('config.yaml', 'r') as config:
            self.cfg = yaml.safe_load(config)[user]
            
        self._isDir()
    
    def _isDir(self) -> bool:
        if not os.path.isdir(self.cfg["Dataset"]):
            ValueError("dataset directory is not found")
            
        if not os.path.isdir(self.cfg["Train"]):
            Warning("train directory was not created earlier")
            os.makedirs(self.cfg["Train"])
        
        if not os.path.isdir(self.cfg["Test"]):
            Warning("test directory was not created earlier")
            os.makedirs(self.cfg["Test"])
    
    def getDataset(self,
                    artists: dict[list] = {"train": [], "test": []},
                    all: bool = False) -> None:
        
        datasetDir = self.cfg["Dataset"]
        
        if all: 
            artists["train"] = os.listdir(f"{datasetDir}/train")
            artists["test"] = os.listdir(f"{datasetDir}/test")
        
        for typeData in self.cfg["Dataset"]: 
            dirs = os.listdir(f"{datasetDir}/{typeData}")
            for artist in f"{datasetDir}/{typeData}/{artists}":
                if artist in dirs:
                    for image_path in f"{dirs}/{artist}":
                        shutil.copy(image_path, self.cfg["Train"])
                        
                else:
                    Warning(f"{artist} is not found in {typeData}")
                    
    def clearDirs(self,
                   isTrain: bool = True,
                   isTest: bool = True
                   ) -> None:
        
        if isTrain: shutil.rmtree(self.cfg["Train"])
        if isTest: shutil.rmtree(self.cfg["Test"])
                    

                    
    