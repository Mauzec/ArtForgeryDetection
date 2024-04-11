import os
import shutil
from platform import platform as pf
is_win = pf().startswith('Win')

# it's hard code motherf**cker
# itll must be changed
PATH = "C:\\home_screen\\programming\\algoritm and data structure\\images" if is_win else "/Users/maus/Downloads/images"
cwd = os.getcwd() + ("\\BoVW" if is_win else "/BoVW")

class Dataset_operations:
    @staticmethod
    def get_images(start: int, end: int, for_using: str) -> None:
        DESTINATION_PATH = f"{cwd}\\dataset\\{for_using}\\other_artist" if is_win else f"{cwd}/dataset/{for_using}/other_artist"
        for path in [PATH, DESTINATION_PATH]:
            if not os.path.isdir(path):
                    raise NameError("No such directory " + path)

        names_folder = os.listdir(PATH)
        k = 0
        for folder in names_folder:
            for subdir, dirs, files in os.walk(os.path.join(PATH, folder)):
                for file in files[start:end]:
                    file_path = os.path.join(subdir, file)

                    shutil.copy(file_path, DESTINATION_PATH)
                    k += 1

        print(f"скопировано {k} картинок")
        
    @staticmethod
    def get_mona() -> None:
        shutil.copy(f"{cwd}\\mona\\mona_original.png" if is_win else f"{cwd}/mona/mona_original.png",
                    f"{cwd}\\dataset\\train\\artist" if is_win else f"{cwd}/dataset/train/artist"
                    )
        shutil.copy(f"{cwd}\\mona\\/mona_younger_1.JPG" if is_win else f"{cwd}/mona/mona_younger_1.JPG",
                    f"{cwd}\\dataset\\test\\artist" if is_win else f"{cwd}/dataset/test/artist"
                    )
        shutil.copy(f"{cwd}\\mona\\/mona_younger_2.JPG" if is_win else f"{cwd}/mona/mona_younger_2.JPG",
                    f"{cwd}\\dataset\\test\\artist" if is_win else f"{cwd}/dataset/test/artist"
                    )

    @staticmethod
    def clear() -> None:   
        for role in ["train", "test"]:
            for artist in ["artist", "other_artist"]:
                path = f"{cwd}\\dataset\\{role}\\{artist}" if is_win else f"{cwd}/dataset/{role}/{artist}" 
                shutil.rmtree(path)
                os.mkdir(path)
                