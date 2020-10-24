from PIL import Image
from utils.origin import *
from models.CENet import CE_Net_

'''
分割器函数
transform2是对label做转换

transform是对原图做转换
'''
transform2 = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),  # torch.Size([1, 160, 160])
    ])

transform = transforms.Compose([
    transforms.Resize((480,480)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     #在ToTensor的结果下，依次减去0.5并除以0.5 则得到归一到[-1,1]的结果
])

#载入模型
net2 = CE_Net_(1,3)
net2.load_state_dict(torch.load('/home/chenxiaojing/PycharmProjects/CENet/checkpoints/0915v1/CP300_k1.pth'))


def segment(directory2, filename):
    label = Image.open(directory2 + "test_mask/" + filename)
    label_ten = transform2(label)
    label_ten = label_ten.unsqueeze(dim=0)

    img = Image.open(directory2 + "test/" + filename)  # 打开图片
    img = transform(img)  # 归一化到 [0.0,1.0]

    net2.cuda().eval()
    # 数据放入 cuda()
    img = img.unsqueeze(0).cuda()
    label = label_ten.float().unsqueeze(0).cuda()

    # 图像放入网络 的到输出
    output = net2(img)

    # 归一化
    output = normPRED(output)

    # 二值化
    output = (output > 0.5).float()

    output2 = output.view(480, 480)
    output3 = output2.int().cpu().numpy() #取二值化矩阵

    return output3
