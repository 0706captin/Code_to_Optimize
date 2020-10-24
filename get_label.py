# from PIL import Image
# import numpy as np
# import cv2
# from time import *
# import os
#
#
# def get_optic_disk_label(directory1_name, directory2_name):
#     for filename in os.listdir(directory1_name):
#         count = 0
#         img = cv2.imread(directory1_name + "/" + filename)
#         im = Image.fromarray(img)
#         (width, height) = im.size
#         im_array = np.asarray(im)
#         im_array = np.array(im_array)
#         for i in range(0, height - 1):
#             for j in range(0, width - 1):
#                 if im_array[i][j] <= 50:
#                     im_array[i][j] = 128
#         im_2 = Image.fromarray(im_array)
#         im_3 = im_2.convert('L')
#         threshold = 185
#         table = []
#         for item in range(256):
#             if item < threshold:
#                 table.append(0)
#             else:
#                 table.append(1)
#         bim = im_3.point(table, '1')
#         bim.save(directory2_name + "/" + filename)
#         count += 1
#         print('Successful! ----->', filename)
#
#
# get_optic_disk_label("/home/chenxiaojing/1_dataset/segmentation/train/mask", "/home/chenxiaojing/1_dataset/segmentation/train/optic_disk_label")
#

from PIL import Image
import numpy as np
import cv2
from time import *
import os


def get_optic_disk_label(directory1_name, directory2_name):
    for filename in os.listdir(directory1_name):
        im = Image.open(directory1_name + "/" + filename)
        (width, height) = im.size
        im_array = np.asarray(im)
        im_array = np.array(im_array)
        for i in range(0, height - 1):
            for j in range(0, width - 1):
                if im_array[i][j] <= 50:
                    im_array[i][j] = 128
        im_2 = Image.fromarray(im_array)
        im_3 = im_2.convert('L')
        threshold = 185
        table = []
        for item in range(256):
            if item < threshold:
                table.append(0)
            else:
                table.append(1)
        bim = im_3.point(table, '1')
        bim.save(directory2_name + "/" + filename)
        print('Successful! ----->', filename)


get_optic_disk_label("/home/chenxiaojing/1_dataset/segmentation/train/mask", "/home/chenxiaojing/1_dataset/segmentation/train/optic_disk_label")