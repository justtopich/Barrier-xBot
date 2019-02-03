import os
import cv2
import numpy as np

def color_rgb_filter(image, min):
    """
    Позволяет выделять цвета не переводя в HSV формат
    за счёт чего не снижается производительность
    :param image:
    :return:
    """
    try:
        B,G,R,_ = cv2.split(image)
    except ValueError:
        B,G,R = cv2.split(image)

    # R = np.where(R < 160, 0, R)
    # G = np.where(G < 160, 0, G)
    # B = np.where(B < 160, 0, B)
    #
    # R = np.where(G == 0, 0, R)
    # R = np.where(B == 0, 0, R)
    # G = np.where(R == 0, 0, G)
    # B = np.where(R == 0, 0, B)

    R = np.where(R < min, 0, R)
    # R[R > 179] = 255
    R = np.where(G < min, 0, R)
    R = np.where(B < min, 0, R)

    return cv2.merge([R])

PATH_TO_IMG_DIR ='.'
PATH_TO_OUT_DIR = './out/'
files = [os.path.join(PATH_TO_IMG_DIR, img) for img in os.listdir(PATH_TO_IMG_DIR) if not img.endswith('py')]
images = [file for file in files if os.path.isfile(file)]
# print(images)
for n, img in enumerate(images):
    image_np = cv2.imread(img, cv2.IMREAD_COLOR)
    image_np = color_rgb_filter(image_np,180)
    
    # cv2.imshow('window', image_np)
    # cv2.waitKey(1)
    # input('?')
    cv2.imwrite(os.path.join(PATH_TO_OUT_DIR, f"{n}.jpg"), image_np)