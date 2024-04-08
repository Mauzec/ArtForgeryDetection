import os
import shutil

PATH = "C:\\home_screen\\programming\\algoritm and data structure\\images"

class Dataset_operations:
    @staticmethod
    def get_images(start: int, end: int, for_using: str) -> None:
        DESTINATION_PATH = f"C:\\home_screen\\programming\\algoritm and data structure\\ArtForgeryDetection\\BoVW\\dataset\\{for_using}\\other_artist"
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
        shutil.copy(f"C:\\home_screen\\programming\\algoritm and data structure\\ArtForgeryDetection\\BoVW\\mona\\mona_original.png",
                    f"C:\\home_screen\\programming\\algoritm and data structure\\ArtForgeryDetection\\BoVW\\dataset\\train\\artist"
                    )
        shutil.copy(f"C:\\home_screen\\programming\\algoritm and data structure\\ArtForgeryDetection\\BoVW\\mona\\mona_younger_2.JPG",
                    f"C:\\home_screen\\programming\\algoritm and data structure\\ArtForgeryDetection\\BoVW\\dataset\\test\\artist"
                    )
        shutil.copy(f"C:\\home_screen\\programming\\algoritm and data structure\\ArtForgeryDetection\\BoVW\\mona\\mona_younger_1.JPG",
                    f"C:\\home_screen\\programming\\algoritm and data structure\\ArtForgeryDetection\\BoVW\\dataset\\test\\artist"
                    )

    @staticmethod
    def clear() -> None:   
        for role in ["train", "test"]:
            for artist in ["artist", "other_artist"]:
                path = f"C:\\home_screen\\programming\\algoritm and data structure\\ArtForgeryDetection\\BoVW\\dataset\\{role}\\{artist}"
                shutil.rmtree(path)
                os.mkdir(path)
                