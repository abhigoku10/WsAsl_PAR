import os
import sys
import time
sys.path.append(os.getcwd())
from PIL import Image
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import glob as glob

def default_loader(path):
    return Image.open(path).convert('RGB')
class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform = None, test = False , loader = default_loader):
        images = []
        if test == False :
            labels = open(label).readlines()
            for line in labels:
                items = line.split()
                img_name = items.pop(0)
                if os.path.isfile(os.path.join(root, img_name)):
                    cur_label = tuple([int(v) for v in items])
                    images.append((img_name, cur_label))
                else:
                    print(os.path.join(root, img_name) + 'Not Found.')
        else : 
            labels = open(label).readlines()
            for line in labels:
                items = line.split()
                img_name = items.pop(0)
                if os.path.isfile(os.path.join(root, img_name)):
                    # cur_label = tuple([int(v) for v in items])
                    images.append((img_name))
                else:
                    print(os.path.join(root, img_name) + 'Not Found.')

        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader
        self.test  = test 

    def __getitem__(self, index):
        if self.test == False : 
            preprocess_time = time.time()
            img_name, label = self.images[index]
            img = self.loader(os.path.join(self.root, img_name))
            imsize = img.size
            raw_img = img.copy()
            if self.transform is not None:
                img = self.transform(img)
            preprocess_end_time = time.time() - preprocess_time
            # return img, torch.Tensor(label)
            return img,torch.Tensor(label),img_name,imsize,preprocess_end_time
        else :
            preprocess_time = time.time()
            img_name = self.images[index]
            img = self.loader(os.path.join(self.root, img_name))
            imsize = img.size
            raw_img = img.copy()
            if self.transform is not None:
                img = self.transform(img)
            preprocess_end_time = time.time() - preprocess_time
            # return img
            return img,img_name,imsize,preprocess_end_time


    def __len__(self):
        return len(self.images)


attr_nums = {}
attr_nums['pa100k'] = 26
attr_nums['rap'] = 51
attr_nums['peta'] = 35

description = {}
description['pa100k'] = ['Female',
                        'AgeOver60',
                        'Age18-60',
                        'AgeLess18',
                        'Front',
                        'Side',
                        'Back',
                        'Hat',
                        'Glasses',
                        'HandBag',
                        'ShoulderBag',
                        'Backpack',
                        'HoldObjectsInFront',
                        'ShortSleeve',
                        'LongSleeve',
                        'UpperStride',
                        'UpperLogo',
                        'UpperPlaid',
                        'UpperSplice',
                        'LowerStripe',
                        'LowerPattern',
                        'LongCoat',
                        'Trousers',
                        'Shorts',
                        'Skirt&Dress',
                        'boots']

description['peta'] = ['Age16-30',
                        'Age31-45',
                        'Age46-60',
                        'AgeAbove61',
                        'Backpack',
                        'CarryingOther',
                        'Casual lower',
                        'Casual upper',
                        'Formal lower',
                        'Formal upper',
                        'Hat',
                        'Jacket',
                        'Jeans',
                        'Leather Shoes',
                        'Logo',
                        'Long hair',
                        'Male',
                        'Messenger Bag',
                        'Muffler',
                        'No accessory',
                        'No carrying',
                        'Plaid',
                        'PlasticBags',
                        'Sandals',
                        'Shoes',
                        'Shorts',
                        'Short Sleeve',
                        'Skirt',
                        'Sneaker',
                        'Stripes',
                        'Sunglasses',
                        'Trousers',
                        'Tshirt',
                        'UpperOther',
                        'V-Neck']

description['rap'] = ['Female',
                        'AgeLess16',
                        'Age17-30',
                        'Age31-45',
                        'BodyFat',
                        'BodyNormal',
                        'BodyThin',
                        'Customer',
                        'Clerk',
                        'BaldHead',
                        'LongHair',
                        'BlackHair',
                        'Hat',
                        'Glasses',
                        'Muffler',
                        'Shirt',
                        'Sweater',
                        'Vest',
                        'TShirt',
                        'Cotton',
                        'Jacket',
                        'Suit-Up',
                        'Tight',
                        'ShortSleeve',
                        'LongTrousers',
                        'Skirt',
                        'ShortSkirt',
                        'Dress',
                        'Jeans',
                        'TightTrousers',
                        'LeatherShoes',
                        'SportShoes',
                        'Boots',
                        'ClothShoes',
                        'CasualShoes',
                        'Backpack',
                        'SSBag',
                        'HandBag',
                        'Box',
                        'PlasticBag',
                        'PaperBag',
                        'HandTrunk',
                        'OtherAttchment',
                        'Calling',
                        'Talking',
                        'Gathering',
                        'Holding',
                        'Pusing',
                        'Pulling',
                        'CarryingbyArm',
                        'CarryingbyHand']




def Get_Dataset(experiment, approach):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(hue=.05, saturation=.05),
        # transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.ToTensor(),
        normalize
        ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
        ])

    ###
    ### Provide the data path where images are present 
    data_path = "/home/abhilash.sk/zzz_projects/HoneywellVA/persondata/RAP_dataset/"
    root = os.getcwd()
    rap_trainlist_path= "/home/abhilash.sk/zzz_projects/HoneywellVA/iccv19_attribute/data_list/rap/train.txt"
    rap_vallist_path = "/home/abhilash.sk/zzz_projects/HoneywellVA/iccv19_attribute/data_list/rap/val.txt"
    rap_testlist_path = "/home/abhilash.sk/zzz_projects/HoneywellVA/iccv19_attribute/data_list/rap/test.txt"

    if experiment == 'pa100k':
        train_dataset = MultiLabelDataset(root='data_path',
                    label='train_list_path', transform=transform_train)
        val_dataset = MultiLabelDataset(root='data_path',
                    label='val_list_path', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['pa100k'], description['pa100k']

    elif experiment == 'rap':
        train_dataset = MultiLabelDataset(root=data_path,
                    label=rap_trainlist_path, transform=transform_train)
        val_dataset = MultiLabelDataset(root=data_path,
                    label=rap_vallist_path, transform=transform_test)
        test_dataset = MultiLabelDataset(root=data_path,
                    label=rap_testlist_path, transform=transform_test, test = True)

        return train_dataset, val_dataset, attr_nums['rap'], description['rap'],test_dataset
    elif experiment == 'peta':
        train_dataset = MultiLabelDataset(root='data_path',
                    label='train_list_path', transform=transform_train)
        val_dataset = MultiLabelDataset(root='data_path',
                    label='val_list_path', transform=transform_test)
        test_dataset = MultiLabelDataset(root='data_path',
                    label='test_list_path', transform=transform_test, test = True)

        return train_dataset, val_dataset, attr_nums['peta'], description['peta'],test_dataset