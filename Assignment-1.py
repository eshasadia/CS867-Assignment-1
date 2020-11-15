import numpy as np
from matplotlib import pyplot as plt
import cv2
def RgbExclusion( image ,n):
    image[:, :, n] = 0 # empty blue channel
    displayImage(image, "Color Excluded Image")

def displayImage(image,title):
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

 def Convolution(image, kernel):

        kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
        output = np.zeros_like(image)  # convolution output
        # Add zero padding to the input image
        image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
        image_padded[1:-1, 1:-1] = image

        # Loop over every pixel of the image and implement convolution operation (element wise multiplication and summation).
        # You can use two loops. The result is stored in the variable output.

        for x in range(image.shape[0]):  # Loop over every pixel of the image
            for y in range(image.shape[1]):
                # element-wise multiplication and summation
                output[x, y] = (kernel * image_padded[x:x + 3, y:y + 3]).sum()

        return output


def plot_helper(x, y, sigma):
    temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -1 / (np.pi * sigma ** 4) * (1 - temp) * np.exp(-temp)

N = 100
half_N = N // 2

x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]

