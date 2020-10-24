import torch
import numpy as np
from utils.origin import tensor2im
from PIL import Image

'''
添加扰动函数：
把图片归一化到0-1之后，给每个像素的r邻域(r由radio决定，r=0为自身，r=1外扩一圈以此类推)添加扰动
如果添加扰动后像素值大于1或小于0则做截断处理。

radio为邻域半径，bis为扰动大小。

'''
def voice(img2, row, col, radio,bis):
    count = 0
    row_up, col_left, row_buttom, col_right = row - radio, col - radio, row + radio, col + radio
    # print(1,row_up,col_left,row_buttom,col_right)
    row_up, row_buttom, col_left, col_right = torch.tensor([row_up, row_buttom, col_left, col_right]).clamp(min=0,
                                                                                                            max=480)

    a = torch.zeros(1, 480, 480)
    a[:, row_up:row_buttom + 1, col_left:col_right + 1] = bis
    img3 = img2 + a
    img3 = torch.tensor(img3).clamp(min=0, max=1)
    count = (row_buttom - row_up + 1) * (col_right - col_left + 1)

    img4 = tensor2im(img3)
    img5 = Image.fromarray(np.uint8(img4))

#     return img5,count
    return img5