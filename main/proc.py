import matplotlib
matplotlib.use('TkAgg')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

from utils import Base


class Proc:
    def __init__(self, title="Test", fileName=None):
        self.fileName = fileName
        self.title = title
        #
        self._original = None
        self._image = None
        self._height = None
        self._width = None
        self._channel = None
        #   
        if fileName is not None:
            self.imagePath = os.path.join(Base.resourcesDir, fileName)
            self.readImage()

    @property
    def image(self):
        """
            getter method for last processed result of image object
        """
        return self._image

    @property
    def original(self):
        """
            getter method for  image object
        """
        return self._original

    @property
    def h(self):
        """
            getter method for image height
        """
        return self._height

    @property
    def w(self):
        """
            getter method for image width
        """
        return self._width

    @property
    def c(self):
        """
            getter method for image width
        """
        return self._channel

    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized


    def readImage(self, fileName=None, replace=True):
        """
            metho to read image and set width and height
        """
        img = cv2.imread(fileName, cv2.IMREAD_COLOR)
        if replace:
            self._image = img
            self._original = img
            self._height = img.shape[0]
            self._width = img.shape[1]
            self._channel = img.shape[2]
        # self.resize(scale = 0.4) 

        img = self.image_resize(img, height=560)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def display(self):
        cv2.imshow(self.title, self._image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _setDim(self, img):
        """
            setter function for dimensions of image
        """
        self._height = img.shape[0]
        self._width = img.shape[1]
        self._channel = img.shape[2]


    def start(self):
        cv2.namedWindow("Display")
        cv2.createButton("Back",self.display,None,cv2.QT_PUSH_BUTTON,0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def displaySampler(self, img=None):
        """
            method to display image
        """
        cv2.destroyAllWindows()
        cv2.imshow(self.title, self._image)
        cv2.createTrackbar('sampler', self.title, 1, 16, self.sampler)
        # cv2.createTrackbar('quantizer', self.title, 1, 100, self.quantizer)
        # # self.sampler(1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def displayQuantizer(self, img=None):
        """
            method to display image
        """
        cv2.destroyAllWindows()
        cv2.imshow(self.title, self._image)
        # cv2.createTrackbar('sampler', self.title, 1, 16, self.sampler)
        cv2.createTrackbar('quantizer', self.title, 1, 100, self.quantizer)
        # # self.sampler(1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def resize(self, scale=1):
        """
            method to resize image
        """
        img = cv2.resize(self._image, fx=scale, fy=scale, dsize = None)
        self._image = img
        self._setDim(img)

        return img

    def histogramEqualizer(self):

        fig = plt.figure()
        orgImage = self._image
        
        # method 1
        # convert from RGB color-space to YCrCb
        # ycrcb_img = cv2.cvtColor(orgImage, cv2.COLOR_BGR2YCrCb)
        # # equalize the histogram of the Y channel
        # ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        # # convert back to RGB color-space from YCrCb
        # equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        # cv2.imshow('Histogram Equalized Image', equalized_img)
        # cv2.imshow('Original Image', self._image)
        
        # method 2
        # convert it to grayscale
        img_yuv = cv2.cvtColor(orgImage,cv2.COLOR_BGR2YUV)
        # apply histogram equalization 
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        equalized_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        # display both images (original and equalized)
        # cv2.imshow("equalizeHist", np.hstack((orgImage, equalized_img)))

        img = orgImage
        if(len(img.shape) == 3):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histogram = cv2.calcHist([img],[i],None,[256],[0,256])
                
                plt.subplot(1, 2, 1)
                plt.plot(histogram,color = col, label="Base Image")
                plt.xlim([0,256])
        else:
            histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
            plt.plot(histogram, color='k')

        img = equalized_img
        if(len(img.shape) == 3):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histogram = cv2.calcHist([img],[i],None,[256],[0,256])
                
                plt.subplot(1, 2, 2)
                plt.plot(histogram,color = col, label="Reference Image")
                plt.xlim([0,256])
        else:
            histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
            plt.plot(histogram, color='k')

        # redraw the canvas
        fig.canvas.draw()

        # convert canvas to image
        pltImage = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        pltImage  = pltImage.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # img is rgb, convert to opencv's default bgr
        equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)
        pltImage = cv2.cvtColor(pltImage, cv2.COLOR_BGR2RGB)
        return equalized_img, pltImage


    def histMatch(self, baseImage, refImage):
        # determine if we are performing multichannel histogram matching
        # and then perform histogram matching itself

        multi = True if baseImage.shape[-1] > 1 else False
        matched = exposure.match_histograms(baseImage, refImage, multichannel=multi)

        fig = plt.figure()

        if(len(baseImage.shape) == 3):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histogram = cv2.calcHist([baseImage],[i],None,[256],[0,256])
                
                plt.subplot(1, 3, 1)
                plt.plot(histogram,color = col, label="Base Image")
                plt.xlim([0,256])
        else:
            histogram = cv2.calcHist([baseImage], [0], None, [256], [0, 256])
            plt.plot(histogram, color='k')

        if(len(refImage.shape) == 3):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histogram = cv2.calcHist([refImage],[i],None,[256],[0,256])
                
                plt.subplot(1, 3, 2)
                plt.plot(histogram,color = col, label="Reference Image")
                plt.xlim([0,256])
        else:
            histogram = cv2.calcHist([refImage], [0], None, [256], [0, 256])
            plt.plot(histogram, color='k')

        if(len(matched.shape) == 3):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histogram = cv2.calcHist([matched],[i],None,[256],[0,256])
                
                plt.subplot(1, 3, 3)
                plt.plot(histogram,color = col, label="Result of Matching")
                plt.xlim([0,256])
        else:
            histogram = cv2.calcHist([matched], [0], None, [256], [0, 256])
            plt.plot(histogram, color='k')

        # redraw the canvas
        fig.canvas.draw()
        # convert canvas to image
        pltImage = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        pltImage  = pltImage.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        pltImage = cv2.cvtColor(pltImage,cv2.COLOR_RGB2BGR)

        plt.close(fig)

        pltImage = cv2.cvtColor(pltImage, cv2.COLOR_BGR2RGB)

        return baseImage, refImage, matched, pltImage

    def gray(self, image=None):
        """
            method to convert image to grayscale
        """
        if image is None:
            img = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
            self._image = img
            self._setDim(img)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return img

    def quantizer(self,quant):
        """
            simple quantizer method for image object
        """

        # img = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        img = self._image
        quantizer = np.zeros((self.w, self.h, 3), np.uint(8))
        quantizer = np.uint8(img/quant) * quant

        quantizer = cv2.cvtColor(quantizer, cv2.COLOR_BGR2RGB)
        return quantizer


    def sampler(self,ratio):
        """
            simple sampler method for image object
        """
        # img = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        img = self._image
        img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
        img = img[0:-1:ratio, 0:-1:ratio]
        img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)
        #

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def plotHist(self):
        img = self._image
        fig = plt.figure()
        if(len(img.shape) == 3):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histogram = cv2.calcHist([img],[i],None,[256],[0,256])
                plt.plot(histogram,color = col, label="Histogram")
                plt.xlim([0,256])
        else:
            histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
            plt.plot(histogram, color='k')
            
        # redraw the canvas
        fig.canvas.draw()

        # convert canvas to image
        pltImage = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        pltImage  = pltImage.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        pltImage = cv2.cvtColor(pltImage,cv2.COLOR_RGB2BGR)

        plt.close(fig)

        pltImage = cv2.cvtColor(pltImage, cv2.COLOR_BGR2RGB)
        return pltImage