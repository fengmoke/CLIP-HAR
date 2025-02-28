import os
import sys
sys.path.append("D:\Study\Postgraduate\Deep_learning\paper\CLIP\Code\TS2ACT-main")
from clip_adapter import build_custom_clip
from PIL import Image
import torch
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
# from clip.clip import *
from clip.CODER import *
import open_clip
# from mobileclip.modules.common.mobileone import reparameterize_model

class Get_dataset_aug(Dataset):  # 得到增强数据集
    def __init__(self,
                 X,
                 y,
                 split = 'train',
                 clip_len = 500):
        super(Dataset, self).__init__()
        self.clip_len = clip_len
        self.split = split
        self.len = X.shape[1]
        self.train_augmenter = (
            TimeWarp()
            # + Crop(size=clip_len) 
            + Quantize(n_levels=[10, 20, 30])
            + Drift(max_drift=(0.1, 0.5)) @ 0.8
            + Reverse() @ 0.5
        )
        self.test_augmenter = (
            Crop(size=clip_len)
        )
        self.X = X.astype(np.float64)
        self.y = y.astype(np.int32)
    def __getitem__(self, index):
        x = self.X[index][np.newaxis,:,:]
        y = self.y[index]
        start1 = random.randint(0, self.len-self.clip_len) if self.len > self.clip_len else 0
        start2 = random.randint(0, self.len-self.clip_len) if self.len > self.clip_len else 0
        if self.split == 'TS2ACT':
            return self.train_augmenter.augment(x)[0,start1:start1+self.clip_len],\
                self.train_augmenter.augment(x)[0,start2:start2+self.clip_len],y
        else:
            return x[0,start2:start2+self.clip_len],y
    def __len__(self):
        return len(self.y)

class Get_dataset(Dataset):
    def __init__(self,
                 X,
                 y,
                 split = 'train',
                 clip_len = 128):
        super(Dataset, self).__init__()
        self.clip_len = clip_len
        self.split = split
        self.len = X.shape[1]
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).view(-1).long()
    def __getitem__(self, index):
        if self.split == 'TS2ACT':
            start1 = random.randint(0, self.len-self.clip_len) if self.len > self.clip_len else 0
            start2 = random.randint(0, self.len-self.clip_len) if self.len > self.clip_len else 0
            return  self.X[index][start1:start1+self.clip_len], \
                    self.X[index][start2:start2+self.clip_len], \
                    self.y[index]
        else:
            start = random.randint(0, self.len-self.clip_len) if self.len > self.clip_len else 0
            return self.X[index][start:start+self.clip_len], self.y[index]
    def __len__(self):
        return len(self.y)


def UCI_aug(shot = 10 ,dataset_dir='TS2ACT-main/dataset/data/UCI',name="10-shot",clip_len = 500):
    path = os.path.join(dataset_dir, name)
    xtrain = np.load(os.path.join(path,"xtrain.npy"), allow_pickle=True).astype('float64')
    xvalid = np.load(os.path.join(path,"xvalid.npy"), allow_pickle=True).astype('float64')
    xtest = np.load(os.path.join(path,"xtest.npy"), allow_pickle=True).astype('float64')
    ytrain = np.load(os.path.join(path,"ytrain.npy"), allow_pickle=True).astype('int64')
    yvalid = np.load(os.path.join(path,"yvalid.npy"), allow_pickle=True).astype('int64')
    ytest = np.load(os.path.join(path,"ytest.npy"), allow_pickle=True).astype('int64')
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxvalid shape: %s\nxtest shape: %s\nytrain shape: %s\nyvalid shape: %s\nytest shape: %s'%(xtrain.shape, xvalid.shape, xtest.shape, ytrain.shape, yvalid.shape, ytest.shape))
    return Get_dataset(xtrain, ytrain, clip_len=clip_len), Get_dataset(xvalid, yvalid, 'TS2ACT', clip_len=clip_len), Get_dataset(xtest, ytest, 'TS2ACT', clip_len=clip_len)
def test(N):
    max_image = 50 if N is None else N
    
    print('正在处理图片数据集...')
    
    # os.chdir('/home/xiakang/xiak/TS')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model = ''
    model, preprocess = load(clip_model)
    pretrained also accepts local paths

    model = model.to(device)
    model.eval()
    
    labels = ['Walking','Walking Upstairs','Walking Downstairs','Sitting','Standing','Laying']
    labels2 = [ 'an action of walking',
                'an action of walking upstairs',
                'an action of walking downstairs',
                'an action of sitting',
                'an action of standing',
                'an action of laying']
    text = tokenize(labels2).to(device)
    with torch.no_grad(): text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    clip_model = clip_model.replace("/", "_")
    torch.save(text_features,f"TS2ACT-main/dataset/data/UCI/UCI_text_no_adapter_{clip_model}.pth")
    save = []
    for label in labels:
        images = []
        for i in range(max_image):
            images.append(preprocess(Image.open('TS2ACT-main/dataset/data/UCI/'+label+'/' + 
                str(i+1)+".jpg")))
        images = torch.tensor(np.stack(images)).to(device)
    
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            save.append(image_features)
    image_features = torch.stack(save,dim=0)
    torch.save(image_features,f"TS2ACT-main/dataset/data/UCI/UCI_image_no_adapter_{clip_model}.pth")
    print(image_features.shape)


def test_adapter(adapter_num, clip_model, N):
    max_image = 50 if N is None else N
    labels = ['Walking','Walking Upstairs','Walking Downstairs','Sitting','Standing','Laying']
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = build_custom_clip('UCI', labels, clip_model, adapter_num)
    clip_model = clip_model.replace("/", "_")
    model_state_dict = torch.load(f'TS2ACT-main/adapter/UCI_best_clip_adapter{adapter_num}_{clip_model}.pt')
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    save_image = []
    save_text = []
    for label in labels:
        images = []
        for i in range(max_image):
            images.append(preprocess(Image.open('TS2ACT-main/dataset/data/UCI/'+label+'/' + 
                str(i+1)+".jpg")))
        images = torch.tensor(np.stack(images)).to(device)
    
        with torch.no_grad():
            _, image_features, text_features = model(images)
            save_image.append(image_features)
            save_text.append(text_features)
    image_features = torch.stack(save_image,dim=0)
    if N is not None:
        torch.save(image_features,f"TS2ACT-main/dataset/data/UCI/N/UCI_image_{clip_model}_adapter{adapter_num}_{N}.pth")
        torch.save(text_features,f"TS2ACT-main/dataset/data/UCI/N/UCI_text_{clip_model}_adapter{adapter_num}.pth")
    else:
        torch.save(image_features,f"TS2ACT-main/dataset/data/UCI/UCI_image_{clip_model}_adapter{adapter_num}.pth")
        torch.save(text_features,f"TS2ACT-main/dataset/data/UCI/UCI_text_{clip_model}_adapter{adapter_num}.pth")
    print(image_features.shape)
    print(text_features.shape)
if __name__ == '__main__':
    test(None)
    # print(open_clip.list_pretrained())
    # test_adapter(0, 'ViT-B/32', None)
    # N_list = [1, 5, 10, 20, 30, 40, 50]
    # for i in N_list:
    #     test_adapter(0, 'ViT-B/32', i)
    # test()
