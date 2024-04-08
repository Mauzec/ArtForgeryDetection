import os
import shutil

PATH = "C:/home_screen\programming/algoritm and data structure/images"

def get_images(start: int, end: int, for_using: str) -> None:
    DESTINATION_PATH = f"C:/home_screen/programming/algoritm and data structure/ArtForgeryDetection/bag-of-visual-word\dataset/{for_using}/other_artist"
    if not os.path.isdir(PATH):
            raise NameError("No such directory " + PATH)
        
    names_folder = os.listdir(PATH)
    k = 0
    for folder in names_folder:
        for subdir, dirs, files in os.walk(os.path.join(PATH, folder)):
            for file in files[start:end]:
                file_path = os.path.join(subdir, file)
                
                shutil.copy(file_path, DESTINATION_PATH)
                k += 1
                
    print(f"скопировано {k} картинок")
                
def get_mona() -> None:
    shutil.copy(f"C:\home_screen\\programming\\algoritm and data structure\\ArtForgeryDetection\\bag-of-visual-word\mona\mona_original.png",
                f"C:\\home_screen\programming\\algoritm and data structure\\ArtForgeryDetection\\bag-of-visual-word\dataset\\train\\artist"
                )
    shutil.copy(f"C:\home_screen\programming\\algoritm and data structure\\ArtForgeryDetection\\bag-of-visual-word\mona\mona_younger_replica.JPG",
                f"C:\\home_screen\programming\\algoritm and data structure\\ArtForgeryDetection\\bag-of-visual-word\dataset\\test\\artist"
                )
    shutil.copy(f"C:\home_screen\programming\\algoritm and data structure\\ArtForgeryDetection\\bag-of-visual-word\mona\mona_younger.JPG",
                f"C:\\home_screen\programming\\algoritm and data structure\\ArtForgeryDetection\\bag-of-visual-word\dataset\\test\\artist"
                )
    
    
if __name__ == "__main__":
    start, end = tuple(map(int, input().split()))
    # for_using = input()
    get_images(start, end)
                