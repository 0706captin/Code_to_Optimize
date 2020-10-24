import torch
import torch.utils.data as DATA
from PIL import Image
from utils.newload import Glaucoma_Dataset
from utils.criterion import dice_coeff3
from utils.visualise import plot_img_and_mask
import matplotlib.pyplot as plt
from models.CENet import CE_Net_
from config import config
from numpy import *
import os
import pandas as pd
import numpy as np
import csv

opt = config()

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn

def save_iou_name(score, name):
    dict_data = {'name': name, 'score': score}
    dataframe = pd.DataFrame.from_dict(dict_data,orient='index')
    # dataframe = pd.DataFrame(data, columns=['name', 'score'])  # columns自定义列的索引值
    csv_file = ('/home/chenxiaojing/PycharmProjects/CENet/results/csv_file/0909test_score.csv')
    dataframe.to_csv(csv_file)

if __name__ == '__main__':

    # net = UNet(n_channels=3, n_classes=1)
    # 1.导入网络模型
    net2 = CE_Net_(opt.num_classes, 3)

    # 2.

    # 3.加载训练好的模型
    net2.load_state_dict(torch.load('/home/chenxiaojing/PycharmProjects/CENet/checkpoints/0909v1_CENET_STUDY/CP300_k1.pth'))

    # 4. 放入gpu
    net2.cuda()

    # 默认操作
    net2.eval()

    # 5. 开始测试、
    # 5.1 封装数据集
    dataset_test = Glaucoma_Dataset(root='/home/chenxiaojing/PycharmProjects/CENet/v6', phase='test')
    # 5.2 加载数据集
    test_loader = DATA.DataLoader(
        dataset_test,
        batch_size=1,  # 测试时，batchsize是1
        drop_last=False,  # 是false
        shuffle=False,  # 是false
        num_workers=1)

    # 开始测试（算分）
    score_ave = 0

    for step, [img, label,name] in enumerate(test_loader):
        # 数据放入 cuda()
        img = img.cuda()
        label = label.float().cuda()

        # 图像放入网络 的到输出
        output = net2(img)
        # 归一化
        output = normPRED(output)
        # 二值化
        output = (output > 0.5).float()
        print(output)

        # 计算 dice
        score = dice_coeff3(output, label)
        name2 = ''.join(name)
        name3 = name2.replace('/home/chenxiaojing/PycharmProjects/B4_attUnetv2/v6/test/','')

        # 打印score，显示图像
        print('step{}'.format(step), ':', score.item(), '\t', name3)
        # 计算总分
        score_ave += score.item()

        data = open("/home/chenxiaojing/PycharmProjects/CENet/results/csv_file/0909test_score.txt","a+")
        print(name3,'\t',score.item(),file = data)
        data.close()
        # 打印图像并保存
        plot_img_and_mask(step, img, output, label, name3.replace('.jpg',''))
        plt.show()

    # 打印平均分
    print("score_ave:", score_ave / (step + 1))




