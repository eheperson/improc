import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from utils import Base
from proc import Proc
import os

"""

# place a label on the root window
    message = tk.Label(root, text="Hello, World!")
    message.pack()



"""


class Window:
    def __init__(self, title="enivicivokki", w=1024, h=780):
        self._title = title
        buttonHeight = 20
        self.processor = Proc()
        #
        window = tk.Tk()
        window.title(title)
        window.resizable(False, False)
        # window.attributes('-alpha', 0.5)
        #
        # get the screen dimensions
        screenWidth = window.winfo_screenwidth()
        screenHeight = window.winfo_screenheight()
        # find the center point
        centerX = int(screenWidth/2 - w / 2)
        centerY = int(screenHeight/2 - h / 2)
        # set the window geometry
        windowGeometry = f'{w}x{h}+{centerX}+{centerY}' # '1280x720+100+100'
        window.geometry(windowGeometry)
        #
        self._window = window
        self._w = w
        self._h = h
        #
        frame = tk.Frame(window)
        frame.pack()
        #
        bottomframe = tk.Frame(window)
        bottomframe.pack( side = tk.TOP)

        loadButton = tk.Button(
            frame, 
            text ="Load Image",
            width=self._w, 
            command=self.loadImage)
        loadButton.pack()
        #
        plotHistogramButton = tk.Button(
            frame, 
            text ="Plot Histogram",
            width=w, 
            command=self.plotHistogram)
        plotHistogramButton.pack()
        #
        histEquButton = tk.Button(
            frame, 
            text ="Histogram Equalization",
            width=w, 
            command=self.histEqualizer)
        histEquButton.pack()
        #
        histMatchButton = tk.Button(
            window, 
            text ="Histogram MAtching",
            width=w, 
            command=self.histMatch)
        histMatchButton.pack()
        #
        samplerButton = tk.Button(
            window, 
            text ="Sampling Test",
            width=w, 
            command=self.sampler)
        samplerButton.pack()
        #
        simpleQuantizerButton = tk.Button(
            window, 
            text ="Simple Quantization Test",
            width=w, 
            command=self.simpleQuantizer)
        simpleQuantizerButton.pack()
        #
        knnQuantizerButton = tk.Button(
            window, 
            text ="KNN Quantization Test",
            width=w, 
            command=self.knnQuantizer)
        knnQuantizerButton.pack()

        bgImagePath = os.path.join(Base.resourcesDir, "bg.jpg")
        bgImage = self.processor.readImage(fileName=bgImagePath)
        img = self.processor.image_resize(bgImage,width=1024)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im) 
        self.imgPane = tk.Label(image=imgtk)
        self.imgPane.image = imgtk
        self.imgPane.pack()

    def loadImage(self):
        path = filedialog.askopenfilename(initialdir = Base.resourcesDir,
                                          title = "Select a File",
                                          filetypes = (("all files",
                         
                                                       "*.*"),("Text files",
                                                        "*.txt*")
                                                       ))
        if len(path) > 0:
            img = self.processor.readImage(fileName=path)
            self._image = img
            im = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im) 
            self.imgPane.image = imgtk
            self.imgPane.pack()
            self.imgPane.config(image=imgtk)
            # imgPane.configure(image=imgtk)
            # imgPane.image = imgtk

    def plotHistogram(self):
        histogram = self.processor.plotHist()
        self.window2(histogram, "Histogram Of Image")
        pass

    def histEqualizer(self):
        equalized, plot=self.processor.histogramEqualizer()
        self.window2(equalized, "Equalized Image")
        self.window2(plot, "Histogram Plots")

    def histMatch(self):
        path = filedialog.askopenfilename(initialdir = Base.resourcesDir,
                                          title = "Select a File",
                                          filetypes = (("all files",
                         
                                                       "*.*"),("Text files",
                                                        "*.txt*")
                                                       ))
        if len(path) > 0:
            refImage = self.processor.readImage(fileName=path, replace=False)
            baseImage, refImage, matched, pltImage = self.processor.histMatch(baseImage=self._image, refImage=refImage)
            self.window2(baseImage, "Base Image")
            self.window2(refImage, "Referance Image")
            self.window2(matched, "Histogram Matching Result")
            self.window2(pltImage, "Histogram Plots")

    def sampler(self):
        result = self.processor.sampler(1)
        #
        w = result.shape[1]
        print(result.shape[0], result.shape[1])
        window2 = tk.Toplevel(self._window)
        window2.title("Image Quantization")
        im = Image.fromarray(result)
        imgtk = ImageTk.PhotoImage(image=im) 
        imgPane = tk.Label(window2, image=imgtk)
        imgPane.image = imgtk
        imgPane.pack(side="top", padx=10, pady=10)

        def updateWindow(samplingRatio):
            print(samplingRatio)
            result = self.processor.sampler(int(samplingRatio))
            im = Image.fromarray(result)
            imgtk = ImageTk.PhotoImage(image=im) 
            imgPane.image = imgtk
            imgPane.configure(image=imgtk)
            
        scaler = tk.Scale(
            window2, 
            label='Sampler', 
            from_=1, to=99, 
            orient=tk.HORIZONTAL, 
            length=w, 
            showvalue=1,
            tickinterval=2, 
            resolution=2, 
            command=updateWindow
        )
        scaler.pack()

    def simpleQuantizer(self):
        result = self.processor.quantizer(1)
        #
        w = result.shape[1]
        window2 = tk.Toplevel(self._window)
        window2.title("Image Quantization")
        im = Image.fromarray(result)
        imgtk = ImageTk.PhotoImage(image=im) 
        imgPane = tk.Label(window2, image=imgtk)
        imgPane.image = imgtk
        imgPane.pack(side="top", padx=10, pady=10)

        def updateWindow(quant):
            print(quant)
            result = self.processor.quantizer(int(quant))
            im = Image.fromarray(result)
            imgtk = ImageTk.PhotoImage(image=im) 
            imgPane.image = imgtk
            imgPane.configure(image=imgtk)
            
        scaler = tk.Scale(
            window2, 
            label='Quantizer', 
            from_=1, to=99, 
            orient=tk.HORIZONTAL, 
            length=w, 
            showvalue=1,
            tickinterval=2, 
            resolution=2, 
            command=updateWindow
        )
        scaler.pack()


    def knnQuantizer(self):
        pass

    def display(self):
        self._window.mainloop()

    def window2(self, img, title):
        window2 = tk.Toplevel(self._window)
        window2.title(title)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im) 
        imgPane = tk.Label(window2, image=imgtk)
        imgPane.image = imgtk
        imgPane.pack(side="top", padx=10, pady=10)


if __name__ == "__main__":
    w = Window()
    w.display()



