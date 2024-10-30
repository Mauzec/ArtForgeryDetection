import yaml
import shutil
import os
import cv2
from sklearn.model_selection import train_test_split
    
class DataOperations:
    
    def __init__(self, user: str) -> None:
        
        self.Dirs = None
        self.decoder = None
        with open('config.yaml', 'r') as config:
            cfg = yaml.safe_load(config)[user]
            self.Dirs: dict = {
                "Data": cfg["Dataset"]["Input"],
                "Train": cfg["Dataset"]["Output"]["Train"],
                "Test": cfg["Dataset"]["Output"]["Test"]}
        self.__isDir()
        self.decoder = None
    
    def __isDir(self,
               isClear: bool = False,
               ) -> bool:
        if not os.path.isdir(self.Dirs["Data"]):
            ValueError("dataset directory is not found")
        for typeData in ["Train", "Test"]:
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
    
    def getData(self, contextKey: str = ["TrainTest", "Split", "FitPredict", "Bin", "Inference"]):
        def getTrainTestData(getTrain: bool = True,
                             getTest: bool = True
                            ) -> None:
            artists = dict.fromkeys(["Train", "Test"], [])
            if getTrain:
                artists["Train"] = os.listdir(f"{self.Dirs["Data"]}/{contextKey}/Train")
            if getTest:    
                artists["Test"] = os.listdir(f"{self.Dirs["Data"]}/{contextKey}/Test")
        
            for typeData in ["Train", "Test"]: 
                dirs = os.listdir(f"{self.Dirs["Data"]}/{contextKey}/{typeData}")
                for artist in artists[typeData]:
                    artistDir = f"{self.Dirs["Data"]}/{contextKey}/{typeData}/{artist}"
                    if artist in dirs:
                        self.__makeDirs([artist], typeData=typeData)
                        for imageName in os.listdir(artistDir):
                            shutil.copy(f"{artistDir}/{imageName}", f"{self.Dirs[typeData]}/{artist}")      
                    else:
                        Warning(f"{artist} is not found in {typeData}")
                        
        def getAndSplitData(test_size: float,
                            random_state: int = 42,
                            ) -> None:
            ImagePaths = dict()
            artistNames = os.listdir(f"{self.Dirs["Data"]}/{contextKey}")
            for artistName in artistNames:
                ImagePaths[artistName] = os.listdir(f"{self.Dirs["Data"]}/{contextKey}/{artistName}")
                
            TrainTestData = dict()
            for artistName in artistNames:
                train, test = train_test_split(ImagePaths[artistName],
                                                test_size=test_size,
                                                random_state=random_state
                                            )
                TrainTestData[artistName] = {"Train": train, "Test": test}
                
            self.__makeDirs(artistNames, "Train")
            self.__makeDirs(artistNames, "Test")
            
            for typeData in ["Test", "Train"]:    
                for artistName in artistNames:
                    for imageName in TrainTestData[artistName][typeData]:
                        shutil.copy(f"{self.Dirs["Data"]}/{contextKey}/{artistName}/{imageName}",
                                    f"{self.Dirs[typeData]}/{artistName}"
                                    )
                        
        def getFitPredictData(encoded: bool = False) -> dict:
            Data = dict.fromkeys(["Train", "Test"], None)
            for typeData in ["Train", "Test"]:
                imageNames = list()
                classImages = list()
                for artistName in os.listdir(self.Dirs[typeData]):
                    for imageName in os.listdir(f"{self.Dirs[typeData]}/{artistName}"):
                        imageNames.append(f"{self.Dirs[typeData]}/{artistName}/{imageName}")
                        classImages.append(artistName)
                        
                Data[typeData] = (imageNames, classImages)

            if encoded:
                classes =  set(Data["Train"][1])
                encoder = dict.fromkeys(classes)
                self.decoder = dict()
                for idx, key in enumerate(classes):
                    print(idx, key)
                    encoder[key] = idx
                    self.decoder[idx] = key
                    
                for typeData in ["Train", "Test"]:
                    for idx in range(len(Data[typeData][1])):
                        Data[typeData][1][idx] = encoder[Data[typeData][1][idx]]
                        
                           
            return Data
        
        def getBinData(getTrain: bool = True,
                       getTest: bool = True) -> None:
            DataTypes = list()
            if getTrain:
                self.__makeDirs(["Artist", "OtherArtist"], "Train")
                DataTypes.append("Train")
            if getTest:
                self.__makeDirs(["Artist", "OtherArtist"], "Test")
                DataTypes.append("Test")
            
            for typeData in DataTypes:
                for artist in os.listdir(f"{self.Dirs["Data"]}/{contextKey}/{typeData}"):
                    for image in os.listdir(f"{self.Dirs["Data"]}/{contextKey}/{typeData}/{artist}"):
                        shutil.copy(f"{self.Dirs["Data"]}/{contextKey}/{typeData}/{artist}/{image}",
                                    f"{self.Dirs[typeData]}/{artist}")
            
        
        def getInference() -> list[str]:
            self.__makeDirs(["Inference"], "Test")
            for image in os.listdir(f"{self.Dirs["Data"]}/{contextKey}"):
                shutil.copy(f"{self.Dirs["Data"]}/{contextKey}/{image}",
                            f"{self.Dirs["Test"]}/{"Inference"}")
        
        contextFunction = {
            "TrainTest": getTrainTestData,
            "Split": getAndSplitData,
            "FitPredict": getFitPredictData,
            "Bin": getBinData,
            "Inference": getInference
        }
        return contextFunction[contextKey]
                    
    def clearDirs(self,
                   isTrain: bool = True,
                   isTest: bool = True
                   ) -> None:
        if isTrain: shutil.rmtree(self.Dirs["Train"])
        if isTest: shutil.rmtree(self.Dirs["Test"])
        self.__isDir(isClear=True)
        
    def __scaleImage(self,
                   imagePath: str,
                   scaleFunctions: list = [cv2.GaussianBlur],
                   args: list = [[(5,5)]],
                   kwargs: list = [{"sigmaX": 36, "sigmaY": 36}],
                   ) -> None:
        
        image =  cv2.imread(imagePath, 0)
        
        for idx, scaleFunction in enumerate(scaleFunctions):
            image = scaleFunction(image, *args[idx], **kwargs[idx])
            
        isWritten = cv2.imwrite(imagePath, image)
        
    def scaleImages(self,
                    ImagePaths: list,
                    scaleFunctions: list = [cv2.GaussianBlur],
                    args: list = [[(5,5)]],
                    kwargs: list = [{"sigmaX": 36, "sigmaY": 36}]
                    ) -> None:
        for ImagePath in ImagePaths:
            self.__scaleImage(
                ImagePaths,
                scaleFunctions,
                args,
                kwargs
            )
        
    def decode(self, EncodedList: list) -> list:
        print(self.decoder)
        DecodedList = [None] * len(EncodedList)
        for idx, code in enumerate(EncodedList):
            DecodedList[idx] = self.decoder[code]   
            
        return DecodedList           

                    
    