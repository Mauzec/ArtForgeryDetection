import os
import shutil
import cv2
import random
from platform import platform as pf
from PIL import Image

is_win = pf().startswith('Win')

PATH = "C:\\home_screen\\programming\\algoritm and data structure\\Dataset" if is_win else "/Users/maus/Downloads/images"
cwd = os.getcwd()

class DatasetOperations:
        
    @staticmethod
    def get_mona_original(PATH: str=PATH) -> None:
        shutil.copy(f"{PATH}\\Mona\\high_resolution\\mona_original.png" if is_win else f"{PATH}/Mona/high_resolution/mona_original.png",
                    f"{cwd}\\dataset\\train\\artist" if is_win else f"{cwd}/dataset/train/artist"
                    )
        
    @staticmethod
    def get_mona_younger(PATH: str=PATH,
                          resolution: str = "low") -> None:
        
        for mona_name in os.listdir(f"{PATH}\\Mona\\{resolution}_resolution" if is_win else f"{PATH}/Mona/{resolution}_resolution"):
            if "younger" in mona_name:
                shutil.copy(f"{PATH}\\Mona\\{resolution}_resolution\\{mona_name}" if is_win else f"{PATH}/Mona/{resolution}_resolution/{mona_name}",
                            f"{cwd}\\dataset\\test\\artist" if is_win else f"{cwd}/dataset/test/artist"
                            )
                
    @staticmethod
    def get_mona_replica(PATH: str=PATH,
                          resolution: str = "low") -> None:
        
        for mona_name in os.listdir(f"{PATH}\\Mona\\{resolution}_resolution" if is_win else f"{PATH}/Mona/{resolution}_resolution"):
            if "replica" in mona_name:
                shutil.copy(f"{PATH}\\Mona\\{resolution}_resolution\\{mona_name}" if is_win else f"{PATH}/Mona/{resolution}_resolution/{mona_name}",
                            f"{cwd}\\dataset\\test\\artist" if is_win else f"{cwd}/dataset/test/artist"
                            )
                
    
    @staticmethod
    def get_mona_test(PATH: str=PATH,
                          resolution: str = "low") -> None:
        
        shutil.copy(f"{PATH}\\Mona\\high_resolution\\mona_original.png" if is_win else f"{PATH}/Mona/{resolution}_resolution/mona_original.png",
            f"{cwd}\\dataset\\test\\artist" if is_win else f"{cwd}/dataset/test/artist"
            )
        
        for mona_name in os.listdir(f"{PATH}\\Mona\\{resolution}_resolution" if is_win else f"{PATH}/Mona/{resolution}_resolution"):
            if not mona_name == "mona_original.png": 
                shutil.copy(f"{PATH}\\Mona\\{resolution}_resolution\\{mona_name}" if is_win else f"{PATH}/Mona/{resolution}_resolution/{mona_name}",
                            f"{cwd}\\dataset\\test\\other_artist" if is_win else f"{cwd}/dataset/test/other_artist"
                            )
        
        
        
    @staticmethod
    def get_work_train_dataset(PATH: str=PATH,
                               percentage_train: int=50,
                               ) -> None:
        if not os.path.isdir(PATH):
            raise NameError("No such directory " + PATH)
        
        names_folder = {
            'Joshua_Reynolds',
            'Raphael',
            'Rembrandt',
            'Thomas_Lawrence',
            'Titian'
        }
        
        k_train = 0
        for folder in names_folder:
            files = [file for root, dirs, files in os.walk(os.path.join(f"{PATH}\\Images" if is_win else f"{PATH}/Images", folder)) for file in files]
            end_for_train = round(len(files) * percentage_train / 100)
            k_train += DatasetOperations.get_pictures_folder(PATH, start=0, end=end_for_train, folder=folder, for_using="train")
            
        print(f"скопировано {k_train} картинок в train")
            
    
    @staticmethod
    def get_pictures_folder(PATH: str=PATH,
                            start: int=0,
                            end: int=0,
                            all: bool=False,
                            folder: str = "",
                            for_using: str="") -> int:
        
        
        DESTINATION_PATH = f"{cwd}\\dataset\\{for_using}\\other_artist" if is_win else f"{cwd}/dataset/{for_using}/other_artist"
        k=0
        for subdir, dirs, files in os.walk(os.path.join(f"{PATH}\\Images" if is_win else f"{PATH}/Images", folder)):
                if all:
                    for file in files:
                        file_path = os.path.join(subdir, file)

                        shutil.copy(file_path, DESTINATION_PATH)
                        k += 1
                else:
                    for file in files[start:end]:
                        file_path = os.path.join(subdir, file)

                        shutil.copy(file_path, DESTINATION_PATH)
                        k += 1
        return k
    
    @staticmethod
    def clear() -> None:   
        for role in ["train", "test"]:
            for artist in os.listdir(f"{cwd}\\dataset\\{role}"):
                path = f"{cwd}\\dataset\\{role}\\{artist}" if is_win else f"{cwd}/dataset/{role}/{artist}" 
                shutil.rmtree(path)
                
    @staticmethod
    def scale_all():
        for for_using in ['train', 'test']:
            for type_artist in os.listdir(f"{cwd}\\dataset\\{for_using}"):
                for image in os.listdir(f"{cwd}\\dataset\\{for_using}\\{type_artist}"):
                    DatasetOperations.scale_image(f"{cwd}\\dataset\\{for_using}\\{type_artist}\\{image}")
    
    @staticmethod
    def scale_image(image_path: cv2.typing.MatLike) -> cv2.typing.MatLike:
        image = cv2.imread(image_path, 0)
        image = cv2.GaussianBlur(image, (5,5), sigmaX=36, sigmaY=36)
        height, width = image.shape
        isWritten = cv2.imwrite(image_path, image)
        return image_path
    
    @staticmethod
    def get_image_path(PATH: str = f"{PATH}\\Images") -> dict:
        artists = dict()
        
        for dir in os.listdir(PATH):
            artists[dir] = list()
            for file in os.listdir(f"{PATH}\\{dir}"):
                artists[dir].append(f"{PATH}\\{dir}\\{file}")
            
        return artists
    
    @staticmethod
    def split_dataset(ratio: float = 0.8) -> None:
        DatasetOperations.clear()
        image_path = DatasetOperations.get_image_path()
        
        for artist, images in image_path.items():
            random.shuffle(images)
            images = images[:10]
            dataset = dict()
            dataset['train'] = images[:round(0.8 * len(images))]
            dataset['test'] = images[round(0.8 * len(images)):]
            
            for for_using in ['train', 'test']:
            
                using_path = f"{cwd}\\dataset\\{for_using}\\{artist}" if is_win else f"{cwd}/dataset/{for_using}/{artist}"

                if not os.path.exists(using_path):
                    os.makedirs(using_path)

                for file_path in dataset[for_using]:
                    shutil.copy(file_path, using_path)
                    
    @staticmethod
    def resize_dataset():
        for for_using in ['train', 'test']:
            for type_artist in os.listdir(f"{cwd}\\dataset\\{for_using}"):
                for image in os.listdir(f"{cwd}\\dataset\\{for_using}\\{type_artist}"):
                    DatasetOperations.resize_image(f"{cwd}\\dataset\\{for_using}\\{type_artist}\\{image}")
    
    @staticmethod
    def resize_image(image_path: str, new_width: int = 308):
        original_image = Image.open(image_path)
        width, height = original_image.size

        new_height = int(height * (new_width / width))

        resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_image.save(image_path)
                                