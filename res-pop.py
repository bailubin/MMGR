import os
import numpy as np
import torch
import torchvision
import argparse
import torch.nn.functional as F
import torch.nn as nn
from modules import ImageEncoder
import math
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/blb/blbdata/img-poi-moco')
parser.add_argument('--lr', type=float, default=5e-2)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--resnet', type=str, default='resnet50')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--total_epoch', type=int, default=40)
parser.add_argument('--model_path', type=str, default='./ckpt')
parser.add_argument('--train_record', type=str, default='res-gcn-wuhanlinear.txt')
parser.add_argument('--schedule', default=[20, 30], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', type=bool, default=False)


class ImagePopDataset(Dataset):
    def __init__(self, path, mode='train'):
        super(ImagePopDataset, self).__init__()
        self.img_path = os.path.join(path, 'img')
        self.pop_path = os.path.join(path, 'pop1km')
        self.mode=mode

        if mode == 'train':
            filelist = os.path.join(path, 'train-wuhan.txt')
        elif mode == 'test':
            filelist = os.path.join(path, 'val-wuhan.txt')

        self.train_files = []
        with open(filelist, 'r') as lines:
            for line in lines:
                self.train_files.append(line.rstrip('\n'))

    def __getitem__(self, index):
        # get image
        img_name = os.path.join(self.img_path, self.train_files[index][:-3] + 'png')
        img = Image.open(img_name)
        img = self.random_aug(img)

        # pop num
        filename = self.train_files[index][:-3] + 'txt'
        pop_f = open(os.path.join(self.pop_path, filename))
        pop = float(pop_f.readline())
        pop_f.close()

        return img, torch.tensor([pop]).float()

    def random_aug(self, img):
        RandomResizeCrop = transforms.RandomResizedCrop(224, scale=(0.2, 1.))
        RandomHorizontalFlip = transforms.RandomHorizontalFlip()
        toTensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if self.mode=='train':
            img = RandomResizeCrop(img)
            img = RandomHorizontalFlip(img)
            img = toTensor(img)
            img = normalize(img)
        else:
            img = toTensor(img)
            img = normalize(img)
        return img

    def __len__(self):
        return len(self.train_files)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch == milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train(model, train_loader):
    # keep normalization same
    model.eval()
    loss_all = 0
    loss_step = 0
    for step, pkg in enumerate(train_loader):
        img, y = pkg[0], pkg[1]
        img = img.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_estimated = F.relu(model.encoder(img))
        loss = F.mse_loss(y_estimated, y).float()

        loss.backward()
        loss_all += loss.item()
        loss_step += loss.item()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss_step / 50}")
            loss_step = 0
    return loss_all / len(train_loader.dataset)


def test(model, loader):
    model.eval()
    yps = []
    yts = []
    for pkg in loader:
        img, y = pkg[0], pkg[1]
        img = img.to(device)

        y_estimated = F.relu(model.encoder(img)).cpu().detach().flatten().numpy()
        y = y.cpu().detach().flatten().numpy()

        yps.append(y_estimated)
        yts.append(y)
    yps = np.concatenate(yps, axis=0)
    yts = np.concatenate(yts, axis=0)
    mae = metrics.mean_absolute_error(yts, yps)
    mse = metrics.mean_squared_error(yts, yps, squared=False)
    r2 = r2_score(yts, yps)
    return mae, mse, r2


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda")

    ckpts = ['wuhan_img_120.tar']
    train_records = ['test.txt']
    sckpts = ['test.tar']
    for ckpt, train_record, sckpt in zip(ckpts, train_records, sckpts):
        # train records
        model_fp = os.path.join(args.model_path, ckpt)
        record_path = os.path.join('./train-records', train_record)
        record_file = open(record_path, 'a')
        record_file.write(
            'res-pop model test, ckpt:{}, batchsize:{}, lr:{}\n'.format(model_fp, args.batch_size, args.lr))
        record_file.close()

        # dataset
        train_dataset = ImagePopDataset(path=args.data_path, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_dataset = ImagePopDataset(path=args.data_path, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

        # image encoder
        resnet = torchvision.models.resnet50(pretrained=False, num_classes=args.projection_dim)
        dim_mlp = resnet.fc.weight.shape[1]
        img_encoder = ImageEncoder(resnet, args.projection_dim, dim_mlp)
        img_encoder.load_state_dict(torch.load(model_fp, map_location=device))
        img_encoder.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, int(dim_mlp / 2), bias=True),
                                               nn.ReLU(),
                                               nn.Linear(int(dim_mlp / 2), 1, bias=True))
        img_encoder = img_encoder.to(device)

        for name, param in img_encoder.named_parameters():
            print(name)
            if 'fc' not in name:
                param.requires_grad = False

        optimizer = torch.optim.Adam(img_encoder.parameters(), lr=args.lr)

        # train and test
        best_r2 = 0
        for epoch in range(args.total_epoch):
            optimizer = adjust_learning_rate(optimizer, epoch, args)
            loss = train(img_encoder, train_loader)
            if epoch >= 5:
                test_error, mse, r2 = test(img_encoder, test_loader)

                print('Epoch: {:03d}, Loss: {:.7f}, Test MAE: {:.7f}, mse:{:.7f}, r2:{:.7f}'.
                      format(epoch, loss, test_error, mse, r2))
                record_file = open(record_path, 'a')
                record_file.write('Epoch: {:03d}, Loss: {:.7f}, Test MAE: {:.7f}, mse:{:.7f}, r2:{:.7f}\n'.
                                  format(epoch, loss, test_error, mse, r2))
                record_file.close()
