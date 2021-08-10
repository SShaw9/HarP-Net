import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Paths

test_img_path = r'../HarPData/Training/Sagittal/x/7025_x_l_sagittal_62'


def inspect_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(path, type(img), img.shape)
    print(img[0])
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # img = cv2.imshow('Test Image', img)
    # plt.imread(test_img_path)
    # plt.imshow()

    return img


def normalise_img(img):
    if not type(img) == 'numpy.ndarray':
        np.array(img)
    dest = np.zeros(img.shape)
    img = cv2.normalize(img, dest, 0, 255, cv2.NORM_MINMAX)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    return img


def flip_img(img):
    if not type(img) == 'numpy.ndarray':
        np.array(img)
    r_img = cv2.rotate(img, cv2.ROTATE_180)
    # plt.imshow(r_img)
    # plt.show()

    return r_img

test_img = inspect_image(test_img_path)
# normalise_img(test_img)
flip_img(test_img)
