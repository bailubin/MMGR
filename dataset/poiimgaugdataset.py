import os
from PIL import Image
from torch_geometric.data import Dataset, Data
import torch
from torchvision import transforms
import random


class PoiImageAugDataset(Dataset):
    def __init__(self, path, edge_w=False):
        super(PoiImageAugDataset, self).__init__()
        self.img_path = os.path.join(path, 'sub_shanghai_img')
        self.poi_path = os.path.join(path, 'sub_shanghai_emb')
        self.edge_path = os.path.join(path, 'sub_shanghai_edge')
        self.edge_w = edge_w

        if self.edge_w:
            self.edge_w_path = os.path.join(path, 'wuhanemb', 'edgeweight')

        self.train_files=os.listdir(self.img_path)


    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.train_files[index])
        poi_name = os.path.join(self.poi_path, self.train_files[index][:-4] + 'tensor')
        edge_name = os.path.join(self.edge_path, self.train_files[index][:-3] + 'tensor')
        img = Image.open(img_name)

        img1 = self.random_aug(img)
        img2 = self.random_aug(img)

        # POI embedding
        second_class_emb_list = torch.load(poi_name, map_location='cpu').float()
        ws = torch.load(edge_name, map_location='cpu').long()

        if self.edge_w:
            edge_w_name = os.path.join(self.edge_w_path, self.train_files[index][:-3] + 'tensor')
            edge_weight = torch.load(edge_w_name, map_location='cpu').float()
            return Data(x=second_class_emb_list, edge_index=ws, edge_attr=edge_weight), img1, img2
        else:
            return Data(x=second_class_emb_list, edge_index=ws), img1, img2

    def random_aug(self, img):
        RandomResizeCrop = transforms.RandomResizedCrop(224, scale=(0.2, 1.))
        ColorJitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        RandomGrayscale = transforms.RandomGrayscale(p=0.2)
        RandomHorizontalFlip = transforms.RandomHorizontalFlip()
        toTensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        img = RandomResizeCrop(img)
        if random.random() > 0.2:
            img = ColorJitter(img)
        img = RandomGrayscale(img)
        img = RandomHorizontalFlip(img)
        img = toTensor(img)
        img = normalize(img)
        return img

    def __len__(self):
        return len(self.train_files)
