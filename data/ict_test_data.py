from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
from os.path import join
import json
import torch
import numpy as np
import random
import data.distortion as distortion
import torch.nn.functional as F
import cv2
import torch.distributed as dist

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class McDataset(Dataset):
    def __init__(self, json_path, real_name, fake_name, train = False, perturb = 'CS#0.4', transform=None, size = 112):
        self.transform = transform
        self.base_dir = json_path

        self.train = train
        self.size = size
        self.perturb = perturb
        self.real_name = real_name
        self.fake_name = fake_name
        self.real_paths = json.load(open(join(self.base_dir, 'paths_of_' + self.real_name + '.json')))
        self.fake_paths = json.load(open(join(self.base_dir, 'paths_of_' + self.fake_name + '.json')))

        self.num = len(self.real_paths) + len(self.fake_paths)
        if dist.get_rank() == 0:
            print ("{2}: Real {0}, Fake {1}, Total {3}, PERTURB {4}"\
                    .format(len(self.real_paths), len(self.fake_paths), fake_name, self.num, self.perturb))
        
    def __len__(self):
        return self.num 

    def __getitem__(self, index):
        if index < len(self.real_paths): ### real
            label = 1
            path = self.real_paths[index]
            
        else: ### fake
            label = 0
            item = index - len(self.real_paths)
            path = self.fake_paths[item]

        img = Image.open(path)
        
        img = self.perturbation(img, self.perturb)
        img = self.transform(img)

        return img, label

    def perturbation(self, im, mode = 'None'):
        if mode == 'None':
            return im
        type, level = mode.split('#')
        level = float(level)
        im = im.resize((256,256))
        im = np.asarray(im)
        im = np.flip(im, 2)
        if type == 'CS':
            im = distortion.color_saturation(im, level)
        elif type == 'CC':
            im = distortion.color_contrast(im, level)
        elif type == 'BW':
            im = distortion.block_wise(im, int(level))
        elif type == 'GNC':
            im = distortion.gaussian_noise_color(im, level)
        elif type == 'GB':
            im = distortion.gaussian_blur(im, int(level))
        elif type == 'JPEG':
            im = distortion.jpeg_compression(im, int(level))
        elif type == 'VC':
            im = distortion.video_compression(im, int(level))
        elif type == 'REAL_JPEG':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
            result, encimg = cv2.imencode('.jpg', im, encode_param)
            im = cv2.imdecode(encimg, 1)
        else:
            print ('Error mode:', mode)
            exit(0)
        im = Image.fromarray(np.flip(im, 2))
        return im
        
