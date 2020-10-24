import torch.utils.data as DATA
from PIL import Image
import torchvision.transforms as transforms
import os

class Dataset(DATA.Dataset):
    def __init__(self, root):
        super(Dataset, self).__init__()
        self.base_folder = 'root'

        test = 'test'

        test_root = os.path.join(root, test)

        test_imgs = os.listdir(test_root)  # test_imgs  #输出的是文件名，包含.jpg——用于测试的原图
        test_imgs.sort()
        self.imgs = [os.path.join(test_root, img) for img in test_imgs]  # `os.path.join`是将两个路径名字粘贴在一起
        #  数据转换操作
        # 原图的转换
        self.transform1 = transforms.Compose([
            transforms.Resize((480, 480)),  # 把原图全部转换成800*800的大小
            transforms.ToTensor(),  # 把原图转化为Tensor, [0-255]转化为[0-1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化 原图转化为[-1 1] image=(图像-平均值)/std
        ])

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        '''
        img = self.imgs[index]  # 此时的img，label只是对应图片的路径，还没有打开文件
        # print(index)
        # doing this so that it is consistent with all other datasets to return a PIL image
        # 打开图片
        # print(img)
        img = Image.open(img)
        img = img.convert('RGB')

        # 数据转换
        img = self.transform1(img)
        #       print(img.shape)
        return img

    def __len__(self):
        '''
        返回数据集中所有的的图片个数
        '''
        return len(self.imgs)