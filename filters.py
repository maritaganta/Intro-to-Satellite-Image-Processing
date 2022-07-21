import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('gamma.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.imshow('fire', image)
cv2.waitKey(0)


# Create a Gaussian blurred image
gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)

cv2.imshow('fire', gray_blur)
cv2.waitKey(0)

# High-pass filter

# 3x3 sobel filters for edge detection
sobel_x = np.array([[ -1, 0, 1],
                   [ -2, 0, 2],
                   [ -1, 0, 1]])


sobel_y = np.array([[ -1, -2, -1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])


# Filter the orginal and blurred grayscale images using filter2D
filtered = cv2.filter2D(gray, -1, sobel_x)
cv2.imshow('fire', filtered)
cv2.waitKey(0)
filtered_blurred = cv2.filter2D(gray_blur, -1, sobel_y)
cv2.imshow('fire', filtered_blurred)
cv2.waitKey(0)

retval, binary_image = cv2.threshold(filtered_blurred, 50, 255, cv2.THRESH_BINARY)
cv2.imshow('fire', binary_image)
cv2.waitKey(0)
# Try Canny using "wide" and "tight" thresholds

wide = cv2.Canny(gray, 30, 100)
tight = cv2.Canny(gray, 220, 250)

cv2.imshow('fire', wide)
cv2.waitKey(0)

cv2.imshow('fire', tight)
cv2.waitKey(0)

# Create a custom kernel

# 3x3 array for edge detection
sobel_y = np.array([[ -1, -2, -1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])

sobel_x = np.array([[ -1/8, 0, 1/8],
                   [ -1/4, 0, 1/4],
                   [ -1/8, 0, 1/8]])
# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
filtered_image_x = cv2.filter2D(gray, -1, sobel_x)
filtered_image_y = cv2.filter2D(gray, -1, sobel_y)

cv2.imshow('fire', filtered_image_x)
cv2.waitKey(0)

cv2.imshow('fire', filtered_image_y)
cv2.waitKey(0)

retval, binary_image = cv2.threshold(filtered_blurred, 50, 255, cv2.THRESH_BINARY)


def ft_image(norm_image):
    '''This function takes in a normalized, grayscale image
       and returns a frequency spectrum transform of that image. '''
    f = np.fft.fft2(norm_image)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20 * np.log(np.abs(fshift))

    return frequency_tx


freq_fire = ft_image(gray/255.0)

cv2.imshow('fire', freq_fire)
cv2.waitKey(0)




#morphological operations
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(gray,kernel,iterations = 1)
dilation = cv2.dilate(gray,kernel,iterations = 1)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow('fire', closing)
cv2.waitKey(0)
















