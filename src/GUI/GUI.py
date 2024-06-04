import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter.messagebox import showerror, showinfo
from tkinter.simpledialog import askstring
import os

class GUI:
    def __init__(self, root: tk.Tk, BovW):
        self.pictures = {"artist": None, "may_artist": None}
        self.BovW = BovW
        self.root = root
        self.root.title("Проектная работа")
        self.root.geometry('900x600')
        root.configure(background="#FDF4E3")
        
        self.root.grid_rowconfigure(0, weight=1, minsize=300)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.image_frame1 = tk.Frame(self.root, borderwidth=2, relief="solid", background="#F8F4FF")
        self.image_frame1.grid(row=0, column=0, padx=10, sticky="nsew")
        self.image_frame1.pack_propagate(False) # Предотвращает растягивание фрейма

        self.image_frame2 = tk.Frame(self.root, borderwidth=2, relief="solid", background="#F8F4FF")
        self.image_frame2.grid(row=0, column=1, padx=10, sticky="nsew")
        self.image_frame2.pack_propagate(False)

        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        # Create a menu for "Изображения"
        self.image_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Изображения", menu=self.image_menu)
        self.image_menu.add_command(label="Оригинальная картина", command=lambda: self.load_image(self.image_frame1, "artist"))
        self.image_menu.add_command(label="Проверяемая картина", command=lambda: self.load_image(self.image_frame2, "may_artist"))

        # Create a menu for "датасет"
        self.dataset_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="датасет", menu=self.dataset_menu)
        self.dataset_menu.add_command(label="обучить модель", command=self.train)
        self.dataset_menu.add_command(label="Использовать предобученную модель", command=self.download_model)
        
        self.menu.add_command(label="проверить картину", command=self.check_picture)
        
        self.image_label1 = tk.Label(self.image_frame1, text="Оригинальная картина", font=("Arial", 12), anchor="center", background="#F8F4FF")
        self.image_label1.pack(expand=True)
        
        self.image_label2 = tk.Label(self.image_frame2, text="Проверяемая картина", font=("Arial", 12), anchor="center", background="#F8F4FF")
        self.image_label2.pack(expand=True)
    
    def load_image(self, frame, type: str, file_path: str = None):
        if file_path is None: file_path = filedialog.askopenfilename()
        
        for widget in frame.winfo_children():
            widget.destroy()
        
        if file_path:
            image = Image.open(file_path)
            self.pictures[type] = file_path
            image.thumbnail((400, int(image.size[1] / image.size[0] * 400)))
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(frame, image=photo, borderwidth=0, relief="flat")
            label.image = photo
            label.pack(expand=True)
            
    def train(self):
        showinfo(title="датасет", message="датасет должен состоять из 2-х набора данных: artist, other_artist. \
                 После того, как вы его загрузите, вы должны выбрать оригинальное изображение. Затем начнется обучение. \
                     После обучения, вы можете выбрать картину, которую хотите проверить.")
        
        directory_path = filedialog.askdirectory()
        self.BovW.add_train_dataset(directory_path)
        image = [os.path.join(f"{directory_path}\\artist", f) for f in os.listdir(f"{directory_path}/artist")][0]
        self.load_image(self.image_frame1, "artist", file_path=image)
        
        self.BovW.training_model()
    
    def download_model(self):
        showinfo(title="датасет", message="Модели, которые вы хотите использовать должны находиться в одной папке c main.py. \
            Они должны называтья: code_book_file_name.npy, modelSVM.jolib, name_classes.json, std_scaler.jolib")
        self.BovW.download_model()
        
    def check_picture(self):
        if self.pictures["may_artist"] is None: self.load_image(self.image_frame2, "may_artist")
        answer = self.BovW.classification_image(self.pictures["may_artist"])
        if answer[0] == "artist": showinfo(title="ответ", message="это картина того же художника")
        else: showinfo(title="ответ", message="это картина другого художника")
        
    def start(self):
        self.root.mainloop()
