import numpy as np
import cv2

def blur_and_th(image,epsilon=1e7):
    # apply a median blur to the image and then subtract the blurred
    # image from the original image to approximate the foreground
    blur = cv2.medianBlur(image, 5)
    foreground = image.astype("float") - blur

    # threshold the foreground image by setting any pixels with a
    # value greater than zero to zero
    foreground[foreground > 0] = 0

    # apply min/max scaling to bring the pixel intensities to the
    # range [0, 1]
    minVal = np.min(foreground)
    maxVal = np.max(foreground)
    foreground = (foreground - minVal) / (maxVal - minVal + epsilon)
    # return the foreground-approximated image
    return foreground


