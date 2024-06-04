from BoVW.BoVW import BoVW
from GUI.GUI import GUI
import tkinter as tk
from CustomDescriptors.SiftDescriptor import SIFT

if __name__ == "__main__":
    bovw = BoVW(descriptor=SIFT, number_words=500)
    root = tk.Tk()
    gui = GUI(root, bovw)
    gui.start()
        