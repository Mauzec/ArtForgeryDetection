from BoVW.BoVW import BoVW
from GUI.GUI import GUI
import tkinter as tk
from ResnetDescriptor.Resnet import ResnetDescruptor as Resnet

if __name__ == "__main__":
    bovw = BoVW(descriptor=Resnet, number_words=500)
    root = tk.Tk()
    gui = GUI(root, bovw)
    gui.start()
        