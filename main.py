# 原图分割
import os
from time import *
from utils.add_preturbation import voice
import torch
from utils.segmentor import segment
from utils.origin import *
import pandas as pd
from config import config

opt = config()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## 原图所在路径
directory = opt.dir_origin_pic
## 扰动图所在路径
directory2 = opt.dir_bias_pic
## 结果保存路径
directory3 = opt.dir_save

output1 = torch.zeros(480, 480)
output2 = torch.zeros(480, 480)
save_img = torch.zeros(1, 480, 480)
save_img = save_img.numpy()
label_influ_1 = torch.zeros(1, 480, 480)
label_influ_1 = label_influ_1.numpy()
label_influ_2 = torch.zeros(1, 480, 480)
label_influ_2 = label_influ_2.numpy()
list_dir = os.listdir(directory)
list_dir.sort()
epsilon = 0.05
# df2 = pd.DataFrame([], columns=['img_name', 'target_pixel', 'epsilon', 'influ_pixel_num', 'influ_pixel'])

begin_time = time()

for filename in list_dir:

    # img2：原图转换后 origin_pic to transform
    img2 = origin(directory, filename)
    print(directory + filename)

    # voice函数表示给img2(原图)添加上ε = 0 的扰动
    img5 = voice(img2, 0, 0, 0, 0)
    img5.save(directory2 + "test/" + filename)

    # output1:原图经过分割器之后得到的矩阵
    output1 = segment(directory2, filename)

    for row in range(0, 480, 1):
        for col in range(0, 480, 1):
            num = 0
            # voice函数表示给img2(原图)添加上ε = 0.1 的扰动
            img5 = voice(img2, row, col, 0, epsilon)
            img5.save(directory2 + "test/" + filename)

            # 送入分割器得到output2：扰动图经过分割器之后得到的矩阵
            output2 = segment(directory2, filename)

            # 异或运算
            c = output1 ^ output2
            q, k = np.where(c == 1)
            # i = list((np.where(c == 1)))[0].tolist()
            # j = list((np.where(c == 1)))[1].tolist()
            # if (len(i) != 0):
            #     data = {'img_name': (directory + filename).replace(directory, ""), 'target_pixel': (row, col),
            #             'epsilon': epsilon, 'influ_pixel_num': len(i), 'influ_pixel': (i, j)}
            #     df2 = df2.append(data, ignore_index=True)
            # #                 print("img_name:",(directory+filename).replace(directory,""),"current_pixel:",(row,col),"epsilon:",epsilon,"influ_num:",len(i),"influ_pixel:",(i,j))
            # df2.to_csv('/home/chenxiaojing/PycharmProjects/CENet/0926save_result/sen_pixel_first.txt', sep='\t', index=False)

            num = q.size

            # save_img用于保存异或运算后output2单个像素影响别的像素个数矩阵
            save_img[0][row][col] = num

    # 取影响像素个数1作为阈值(标绿显示)
    label_influ_1 = np.int64(save_img >= 1)
    label_influ_2 = np.int64(save_img >= 2)

    im_save = origin(directory, filename)
    im_save_array = tensor2im(im_save)
    im_save_array2 = tensor2im(im_save)
    row_p, col_p = np.where(label_influ_1[0] == 1)
    row_p2, col_p2 = np.where(label_influ_2[0] == 1)

    data = open(directory3 + "pic_pixels_num1.txt", "a+")
    print(filename, file=data)
    data.close()

    # 将save_img中超过阈值的像素取出，在原图上用绿色像素点显示
    for row_p, col_p in zip(row_p, col_p):
        data = open(directory3 + "pic_pixels_num1.txt", "a+")
        print((row_p, col_p), file=data)
        data.close()
        im_save_array[:][row_p][col_p] = 0, 255, 0
    im_save_img = Image.fromarray(np.uint8(im_save_array))
    im_save_img.save(directory3 + filename[0:5] + ".png")

    data = open(directory3 + "pic_pixels_num2.txt", "a+")
    print(filename, file=data)
    data.close()

    for row_p2, col_p2 in zip(row_p2, col_p2):
        data = open(directory3 + "pic_pixels_num2.txt", "a+")
        print((row_p2, col_p2), file=data)
        data.close()
        im_save_array2[:][row_p2][col_p2] = 0, 255, 0
    im_save_img2 = Image.fromarray(np.uint8(im_save_array2))
    im_save_img2.save(directory3 + filename[0:5] + "2" + ".png")

end_time = time()
run_time = end_time - begin_time
print("total_tun_time is:", run_time)
# df2.to_csv("/home/chenxiaojing/PycharmProjects/CENet/0926save_result/sen_pixel.csv")
# df2.to_csv('/home/chenxiaojing/PycharmProjects/CENet/0926save_result/sen_pixel.txt',sep='\t',index=False)