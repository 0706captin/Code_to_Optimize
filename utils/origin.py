import numpy as np
import torch
from torchvision import transforms
from PIL import Image
def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def normPRED(d):
    ma = torch.max(d)  #求输入tensor中的最大值
    mi = torch.min(d)  #求输入tensor中的最小值

    dn = (d - mi) / (ma - mi)   #定义dn=输入tensor-最小值/最大值-最小值 这里的规范化作用？类似于normalize归一到[-1,1]吗？

    return dn   #返回规范化之后的结果

def origin(direcotry,filename):
    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    im = Image.open(direcotry+filename)
    return transform(im)


def tensor2im(image_tensor, imtype=np.uint8, normalize = True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)