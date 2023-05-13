import os
import torch
import torchvision
import argparse
import torch_geometric.data
import math
import torch.nn as nn
from modules import *

from dataset import PoiImageAugDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/blb/blbdata/img-poi-moco')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--resnet', type=str, default='resnet50')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--total_epoch', type=int, default=120)
parser.add_argument('--model_path', type=str, default='./ckpt')
parser.add_argument('--schedule', default=[90, 110], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', type=bool, default=False)
parser.add_argument('--train_record', type=str, default='gcn-record.txt')


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.total_epoch))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch == milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    train_dataset = PoiImageAugDataset(path=args.data_path, poi_f='poi12-100')
    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=6)

    # image encoder
    resnet = torchvision.models.resnet50(pretrained=False, num_classes=args.projection_dim)
    resnet.fc = nn.Linear(resnet.fc.weight.shape[1], args.projection_dim, bias=True)
    resnet.fc.weight.data.normal_(mean=0.0, std=0.01)
    resnet.fc.bias.data.zero_()
    dim_mlp = resnet.fc.weight.shape[1]
    img_encoder = ImageEncoder(resnet, args.projection_dim, dim_mlp).to(device)
    img_encoder.train()

    # poi encoder
    poi_encoder = GCNSet2SetNoGCNNet(args.projection_dim, dim_mlp).to(device)
    poi_encoder.train()

    # optimizer
    img_optimizer = torch.optim.Adam(img_encoder.parameters(), lr=args.lr)
    poi_optimizer = torch.optim.Adam(poi_encoder.parameters(), lr=args.lr)

    # loss
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    # train records
    record_path = os.path.join('./train-records', args.train_record)
    img_lmd = 0.5

    for epoch in range(1, args.total_epoch + 1):
        loss_step1 = 0
        loss_step2 = 0
        img_optimizer = adjust_learning_rate(img_optimizer, epoch, args)
        poi_optimizer = adjust_learning_rate(poi_optimizer, epoch, args)
        for step, data in enumerate(train_loader):

            img_optimizer.zero_grad()
            poi_optimizer.zero_grad()

            poi, img1, img2 = data[0], data[1], data[2]
            if img1.shape[0] != args.batch_size:
                continue
            poi = poi.to(device)
            img1 = img1.to(device)
            img2 = img2.to(device)

            z_img1, z_img2 = img_encoder(img1, img2)
            z_poi = poi_encoder(poi)

            loss2 = criterion(z_img1, z_img2)
            loss1 = criterion(z_img1, z_poi)
            loss = loss1 * (1 - img_lmd) + loss2 * img_lmd
            loss.backward()

            img_optimizer.step()
            poi_optimizer.step()

            loss_step1 += loss1.item()
            loss_step2 += loss2.item()

            if step % 50 == 0:
                print(
                    f"epoch [{epoch}/{args.total_epoch}]\t Step [{step}/{len(train_loader)}]\t Loss1: {loss_step1 / 50}\t Loss2: {loss_step2 / 50}")
                record_file = open(record_path, 'a')
                record_file.write(
                    f"Step [{step}/{len(train_loader)}]\t Loss1: {loss_step1 / 50}\t Loss2: {loss_step2 / 50}\n")
                record_file.close()
                loss_step1 = 0
                loss_step2 = 0

        if epoch % 20 == 0 or epoch == args.total_epoch:
            out_img = os.path.join(args.model_path, "wuhan_img_{}.tar".format(epoch))
            out_poi = os.path.join(args.model_path, "wuhan_poi_{}.tar".format(epoch))
            torch.save(img_encoder.state_dict(), out_img)
            torch.save(poi_encoder.state_dict(), out_poi)
