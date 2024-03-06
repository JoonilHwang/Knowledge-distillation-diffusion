from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from skimage import io
from skimage.transform import resize
import pdb
import numpy as np
import cv2
from matplotlib import pyplot as plt

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        img_path = self.image_paths[index]
        image = None
        
        #transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(p=p),
            #transforms.Resize(self.image_size),
            #transforms.ToTensor()
        #])
        
        #try:
        #    image = Image.open(img_path)
        #except BaseException as e:
        #    print(img_path)

        #if not image.mode == 'RGB':
        #    image = image.convert('F')

        #image = transform(image)

        #if self.to_normal:
        #    image = (image - 0.5) * 2.
        #    image.clamp_(-1., 1.)
        
        image = io.imread(img_path)

        image_resize = cv2.resize(image,(self.image_size[0],self.image_size[1]))
        #print(img_path)
        #image_resize = np.fromfile(img_path,dtype=np.float32).reshape(256,256)
        image_resize_norm = image_resize
        image_extend = np.expand_dims(image_resize_norm,axis=0)
        image_concat = np.concatenate((image_extend,image_extend,image_extend),axis=0) # For the 3 channel transformation
        
        image_name = Path(img_path).stem
        return image_concat, image_name


