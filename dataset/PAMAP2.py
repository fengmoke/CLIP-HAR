import os
import sys
from clip_adapter import build_custom_clip
import torch
from PIL import Image
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from clip.clip import *
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
# from tip_adapter import *


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

class Get_dataset_aug(Dataset):
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
                 clip_len = 500):
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

def PAMAP_aug(clip_len = 500 ,dataset_dir='TS2ACT-main/dataset/data/PAMAP2',name="1-shot"):
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



def test():
    # os.chdir('/home/xiakang/xiak/TS')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prompt = 'an action of'
    labels = ['laying','sitting','standing','walking','running','cycling','Nordic walking','ascending stairs','descending stairs','vacuum cleaning','ironing','rope jumping',]
    prompt_labels = [  prompt + 'laying',
                prompt + 'sitting',
                prompt + 'standing',
                prompt + 'walking',
                prompt + 'running',
                prompt + 'cycling',
                prompt + 'Nordic walking',
                prompt + 'ascending stairs',
                prompt + 'descending stairs',
                prompt + 'vacuum cleaning',
                prompt + 'ironing',
                prompt + 'rope jumping',
            ]
    text = tokenize(prompt_labels).to(device)
    clip_model = 'RN101'
    model, preprocess = load(clip_model)
    model = model.to(device)
    model.eval()
    with torch.no_grad(): text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    clip_model = clip_model.replace("/", "_")
    torch.save(text_features,f"TS2ACT-main/dataset/data/PAMAP2/PAMAP2_text_no_adapter_{clip_model}.pth")
    save = []
    for label in labels:
        images = []
        for i in range(50):
            images.append(preprocess(Image.open('TS2ACT-main/dataset/data/PAMAP2/'+label+'/' + 
                str(i+1)+".jpg")))
        images = torch.tensor(np.stack(images)).to(device)
    
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            save.append(image_features)
    image_features = torch.stack(save,dim=0)
    torch.save(image_features,f"TS2ACT-main/dataset/data/PAMAP2/PAMAP2_image_no_adapter_{clip_model}.pth")
    print(image_features.shape)



def get_meta_adapter_feature(clip_model= 'RN101'):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prompt = 'an action of'
    labels = ['laying','sitting','standing','walking','running','cycling','Nordic walking','ascending stairs','descending stairs','vacuum cleaning','ironing','rope jumping',]
    prompt_labels = [  prompt + 'laying',
                prompt + 'sitting',
                prompt + 'standing',
                prompt + 'walking',
                prompt + 'running',
                prompt + 'cycling',
                prompt + 'Nordic walking',
                prompt + 'ascending stairs',
                prompt + 'descending stairs',
                prompt + 'vacuum cleaning',
                prompt + 'ironing',
                prompt + 'rope jumping',
            ]
    text = tokenize(prompt_labels).to(device)
    model, preprocess = load(clip_model)
    model = model.to(device)
    model.eval()
    train_data = []
    for n, text_label in enumerate(labels):
        selected_index = random.sample(list(range(50)), k=40)
        assert len(selected_index) == len(set(selected_index))
        data = []
        for i in range(50):
            data.append((preprocess(Image.open(f'TS2ACT-main/dataset/data/PAMAP2/'+text_label+'/' +
                str(i+1)+".jpg")), n))
            # if i in selected_index:
            #     train_data.append(data[i])
            # else:
            #     val_data.append(data[i])
            train_data.append(data[i])
    train_dataset = CustomDataset(train_data)
    train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=8, shuffle=False,
                pin_memory=False)
    clip_weights = clip_classifier(labels, prompt, model)
    cache_keys = build_cache_model(model, train_dataloader, True)
    train_features, train_labels = pre_load_features(model, train_dataloader, True)
    run_meta_adapter(cache_keys, train_features, train_labels, clip_weights, model, train_dataloader)
    

def test_adapter(adapter_num, clip_model):
    labels = ['laying','sitting','standing','walking','running','cycling','Nordic walking','ascending stairs','descending stairs','vacuum cleaning','ironing','rope jumping']
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if adapter_num < 8:
        model, preprocess = build_custom_clip('PAMAP2', labels, clip_model, adapter_num)
        clip_model = clip_model.replace("/", "_")
        model_state_dict = torch.load(f'TS2ACT-main/adapter/PAMAP2_best_clip_adapter{adapter_num}_{clip_model}.pt')
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()
        save_image = []
        save_text = []
        for label in labels:
            images = []
            for i in range(50):
                images.append(preprocess(Image.open('TS2ACT-main/dataset/data/PAMAP2/'+label+'/' + 
                    str(i+1)+".jpg")))
            images = torch.tensor(np.stack(images)).to(device)
        
            with torch.no_grad():
                _, image_features, text_features = model(images)
                save_image.append(image_features)
                save_text.append(text_features)
        image_features = torch.stack(save_image,dim=0)
    else:
        prompt = 'an action of'
        labels = ['laying','sitting','standing','walking','running','cycling','Nordic walking','ascending stairs','descending stairs','vacuum cleaning','ironing','rope jumping',]
        prompt_labels = [  prompt + 'laying',
                    prompt + 'sitting',
                    prompt + 'standing',
                    prompt + 'walking',
                    prompt + 'running',
                    prompt + 'cycling',
                    prompt + 'Nordic walking',
                    prompt + 'ascending stairs',
                    prompt + 'descending stairs',
                    prompt + 'vacuum cleaning',
                    prompt + 'ironing',
                    prompt + 'rope jumping',
                ]
        text = tokenize(prompt_labels).to(device)
        model, preprocess = load(clip_model)
        model = model.to(device)
        model.eval()
        visual_prompt = build_custom_clip('PAMAP2', labels, clip_model, adapter_num)[-1]
        visual_prompt_state_dict = torch.load(f'TS2ACT-main/adapter/PAMAP2_best_clip_adapter{adapter_num}_{clip_model}.pt')
        visual_prompt.load_state_dict(visual_prompt_state_dict)
        visual_prompt = visual_prompt.to(device)
        visual_prompt.eval()
        with torch.no_grad(): text_features = model.encode_text(text)
        save_image = []
        for label in labels:
            images = []
            for i in range(50):
                images.append(preprocess(Image.open('TS2ACT-main/dataset/data/PAMAP2/'+label+'/' + 
                    str(i+1)+".jpg")))
            images = torch.tensor(np.stack(images)).to(device)
        
            with torch.no_grad():
                images = visual_prompt(images)
                image_features = model.encode_image(images)
                save_image.append(image_features)
        image_features = torch.stack(save_image,dim=0)
    torch.save(image_features,f"TS2ACT-main/dataset/data/PAMAP2/PAMAP2_image_{clip_model}_adapter{adapter_num}.pth")
    torch.save(text_features,f"TS2ACT-main/dataset/data/PAMAP2/PAMAP2_text_{clip_model}_adapter{adapter_num}.pth")
    print(image_features.shape)
    print(text_features.shape)

if __name__ == '__main__':
    # test()
    test_adapter(0, 'RN101')
    # get_meta_adapter_feature()
    # N_list = [1, 5, 10, 20, 30, 40, 50]
    # for i in N_list:
    #     test_adapter(0, 'RN101', i)
