import torch.utils.data as DATA
from PIL import Image
import torchvision.transforms as transforms
import os

class Glaucoma_Dataset(DATA.Dataset):
    def __init__(self, root, phase):
        '''
        目标：获取所有图片地址，并根据训练、验证、测试集划分数据
        '''
        super(Glaucoma_Dataset, self).__init__()
        self.base_folder = 'root'
        self.phase = phase

        # 判断语句，区分训练集和测试集，获取路径
        if self.phase == 'train':

            train = 'train'
            label = 'train_mask'

            train_root = os.path.join(root, train)
            label_root = os.path.join(root, label)
            # print('train_root:', train_root)

            train_imgs = os.listdir(train_root)  # train_imgs  #输出的是文件名，包含.jpg——用于训练的原图
            label_imgs = os.listdir(label_root)  # label_imgs  #输出的是文件名，包含.jpg——用于训练的做好标签的图
            # print('train_imgs:', train_imgs)

            self.imgs = [os.path.join(train_root, img) for img in train_imgs]  # `os.path.join`是将两个路径名字粘贴在一起
            self.labels = [os.path.join(label_root, img) for img in label_imgs]
            self.name = self.imgs

        else:
            test = 'test'
            label = 'test_mask'

            test_root = os.path.join(root, test)
            label_root = os.path.join(root, label)

            test_imgs = os.listdir(test_root)  # train_imgs  #输出的是文件名，包含.jpg——用于训练的原图
            label_imgs = os.listdir(label_root)  # label_imgs  #输出的是文件名，包含.jpg——用于训练的做好标签的图

            self.imgs = [os.path.join(test_root, img) for img in test_imgs]  # `os.path.join`是将两个路径名字粘贴在一起
            self.labels = [os.path.join(label_root, img) for img in label_imgs]
            self.name = self.imgs

        #  数据转换操作
        # 原图的转换
        self.transform1 = transforms.Compose([
            #             transforms.Pad(300,0),         #(1600, 1600)
            transforms.Resize((320, 320)),  # (160, 160)
            #             transforms.CenterCrop(160),    #(160, 160)
            transforms.ToTensor(),  # torch.Size([3, 160, 160])
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # label的转换
        self.transform2 = transforms.Compose([
            #             transforms.Pad(300,0),         #(1600, 1600)
            transforms.Resize((320, 320)),  # (160, 160)
            #             transforms.CenterCrop(160),    #(160, 160)
            transforms.ToTensor(),  # torch.Size([1, 160, 160])
        ])

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        '''
        img, label, name = self.imgs[index], self.labels[index], self.name[index]  # 此时的img，label只是对应图片的路径，还没有打开文件
        # doing this so that it is consistent with all other datasets to return a PIL image

        # 打开图片
        img = Image.open(img)
        #         img = img.resize((160, 160))

        # 打开label 图片
        label = Image.open(label)
        #         label = label.resize((160, 160))

        # 数据转换
        img = self.transform1(img)
        label = self.transform2(label)

        return img, label, name

    def __len__(self):
        '''
        返回数据集中所有的的图片个数
        '''
        return len(self.imgs)

if __name__ == '__main__':
    # packaging data
    dataset_train = Glaucoma_Dataset(root='/home/chenxiaojing/PycharmProjects/B4_attUnetv2/v6', phase='train')
    # load data
    train_loader = DATA.DataLoader(
        dataset_train,
        batch_size=2,
        drop_last=True,
        shuffle=True,
        num_workers=1)

    for epoch in range(2):
        print('Starting epoch {}/{}.'.format(epoch + 1, 2))
        for step, [img, label] in enumerate(train_loader):
            print(img.shape)
            break
